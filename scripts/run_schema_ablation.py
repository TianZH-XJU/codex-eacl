#!/usr/bin/env python3
"""A2 schema ablation runner (single model per process).

Rows sampled from v2 track-clean search_dev pools (NOT old pilot rows, NOT test).
Representations (pre-registered in semantic_schemas.py):
- legacy_noul (Laya only): no criteria (historical ablation reference)
- semantic_noul: explicit false/true semantics
- choice: neutral two-option Choice
Records raw P(true)/choice probs for AUROC/AUPRC + agreement + label mix.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LAYA_ROOT = ROOT / "external" / "laya"
sys.path.insert(0, str(LAYA_ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def prevent_sleep():
    try:
        return bool(ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED))
    except Exception:
        return False


def allow_sleep():
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
    except Exception:
        pass


def git_sha(path):
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                              check=True, capture_output=True, text=True).stdout.strip()
    except Exception:
        return None


def _code_rows(limit, seed, split):
    from group_oof import function_group_id
    rng = random.Random(seed)
    manifest = json.loads(Path("results/v2/data_manifest/codexglue_function_groups.json").read_text(encoding="utf-8"))
    js_all = json.loads(Path("data/raw/codexglue_defect/function.json").read_text(encoding="utf-8"))
    members = manifest.get("members")
    if members is None:  # legacy manifest: rebuild from official train indices only
        train_idx = {int(x) for x in Path("data/raw/codexglue_defect/train.txt").read_text(encoding="utf-8").split()}
        g2idx = {}
        for i, r in enumerate(js_all):
            if i in train_idx:
                g2idx.setdefault(function_group_id(r["func"]), []).append(i)
        members = g2idx
    groups = list(manifest["splits"][split])
    rng.shuffle(groups)
    rows = []
    for g in groups:
        for i in sorted(members.get(g, [])):
            rows.append({"key": f"idx:{i}", "text": js_all[i]["func"],
                         "gold": bool(js_all[i]["target"]), "group_id": g,
                         "meta": {"idx": i, "project": js_all[i].get("project", "?")}})
            if len(rows) >= limit:
                return rows
    return rows


def _findver_rows(limit, seed, split):
    rng = random.Random(seed)
    manifest = json.loads(Path("results/v2/data_manifest/findver_report_groups.json").read_text(encoding="utf-8"))
    return _findver_rows_from_ids(
        [eid for rep in manifest["splits"][split] for eid in
         [r["example_id"] for r in json.loads(Path("external/FinDVer/data/testmini.json").read_text(encoding="utf-8")) if r["report"] == rep]],
        rng, limit)


def _findver_rows_from_ids(ids, rng, limit):
    data = {r["example_id"]: r for r in
            json.loads(Path("external/FinDVer/data/testmini.json").read_text(encoding="utf-8"))}
    ids = list(ids)
    rng.shuffle(ids)
    rows = []
    for eid in ids[:limit]:
        r = data[eid]
        rep = json.loads((ROOT / f"external/FinDVer/financial_reports/{r['report']}").read_text(encoding="utf-8"))
        ev = "\n".join(rep["context"][i]["context"] for i in r["relevant_context"])
        rows.append({"key": eid, "text": f"Claim: {r['statement']}\n\nEvidence:\n{ev}",
                     "gold": bool(r["entailment_label"]), "group_id": f"rp:{r['report']}"})
    return rows


def resolve_rows(task, limit, seed, pool="v1"):
    """pool is reserved for future pools; default = v2 search_dev (matches all A2 runs)."""
    if task == "code_defect":
        split = "search_train" if pool == "v2train" else "search_dev"
        return _code_rows(limit, seed, split)
    split = "search_train" if pool == "v2train" else "search_dev"
    return _findver_rows(limit, seed, split)


def normalize_choice(raw, valid):
    """Normalize a raw choice string for comparison. Returns (normalized|None, reason)."""
    if raw is None:
        return None, "missing"
    norm = str(raw).strip().lower().replace("_", " ")
    norm = " ".join(norm.split())
    if norm in valid:
        return norm, "ok"
    return None, f"unknown choice {raw!r}"


def build_question(task, rep):
    import semantic_schemas as S
    key = f"{task.split('_')[0]}_{rep}"
    mapping = {"code_legacy_noul": None, "findver_legacy_noul": None,
               "code_semantic_noul": S.CODE_DEFECT_NOUL, "findver_semantic_noul": S.FINDVER_NOUL,
               "code_choice": S.CODE_DEFECT_CHOICE, "findver_choice": S.FINDVER_CHOICE}
    base = mapping.get(key)
    if base is None:  # legacy: instructions only, no criteria
        instr = {"code": "Is this function vulnerable?",
                 "findver": "Is the claim supported by the evidence?"}[task.split("_")[0]]
        return {"type": "noul", "instructions": instr}
    return dict(base)


ALLOWED_BUDGETS = {"head_max_len": (1, 192), "max_len": (1, 512)}


def parse_predict_kwargs(raw, model):
    """Parse k=v,k=v or equivalent JSON. Raises SystemExit on misuse."""
    if model != "laya":
        raise SystemExit("--predict-kwargs is Laya-only (Vela has no such per-call budget API)")
    raw = (raw or "").strip()
    try:
        parsed = dict(json.loads(raw)) if raw.startswith("{") else None
    except ValueError:
        parsed = None
    if parsed is None:
        parsed = {}
        for part in raw.split(","):
            if "=" not in part:
                raise SystemExit(f"invalid --predict-kwargs entry (want k=v): {part!r}")
            k, v = part.split("=", 1)
            try:
                parsed[k.strip()] = int(v.strip())
            except ValueError:
                raise SystemExit(f"non-integer budget value: {part!r}")
    for k, v in parsed.items():
        if k not in ALLOWED_BUDGETS:
            raise SystemExit(f"unsupported budget key: {k}")
        lo, hi = ALLOWED_BUDGETS[k]
        if isinstance(v, bool) or not isinstance(v, int) or not (lo <= v <= hi):
            raise SystemExit(f"budget {k}={v} out of pinned in-range [{lo},{hi}] (larger = OOD stress, separate)")
    return parsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["laya", "vela"])
    ap.add_argument("--task", required=True, choices=["code_defect", "findver"])
    ap.add_argument("--reps", default=None)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--predict-kwargs", default=None,
                    help='Budget overrides for Laya only: k=v,k=v (e.g. head_max_len=96) or '
                         'equivalent JSON. Rejected for Vela and for out-of-range values.')
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_available():
        raise RuntimeError("CPU-only run refused: CUDA is available")

    default_reps = {"laya": ["legacy_noul", "semantic_noul", "choice"],
                    "vela": ["semantic_noul", "choice"]}[args.model]
    reps = args.reps.split(",") if args.reps else default_reps
    budget_kw = parse_predict_kwargs(args.predict_kwargs, args.model) if args.predict_kwargs else {}
    if args.model == "laya" and "legacy_noul" not in reps:
        pass
    if args.model == "vela" and "legacy_noul" in reps:
        raise SystemExit("vela has no legacy_noul rep (plan A2: semantic + choice only)")

    rows = resolve_rows(args.task, args.limit, args.seed)

    if args.model == "laya":
        sys.path.insert(0, str(LAYA_ROOT / "research" / "benchmarks" / "tournament"))
        import laya  # noqa: E402
        agent = laya.load("english", device="cpu")
        def call(state, q):
            return agent.predict(state, {"q": q}, **budget_kw)["answers"]["q"], 1
    else:
        from transformers import AutoModel
        agent = AutoModel.from_pretrained(str(ROOT / "models/vela"), trust_remote_code=True)
        def call(state, q):
            return agent.system_one(state, {"q": q})["answers"]["q"], 1

    proc = psutil.Process()
    import os
    peak = proc.memory_info().rss
    sleep_on = prevent_sleep()
    results = []
    try:
        for pos, row in enumerate(rows):
            for rep in reps:
                q = build_question(args.task, rep)
                t0 = time.perf_counter()
                err, ans, n_calls = None, None, None
                try:
                    ans, n_calls = call({"message": row["text"]}, q)
                except Exception as e:  # noqa: BLE001
                    err = f"{type(e).__name__}: {e}"
                ms = (time.perf_counter() - t0) * 1000.0
                peak = max(peak, proc.memory_info().rss)
                pred = score = raw_choice = norm_choice = None
                if err is None and ans is not None:
                    if ans.get("type") == "noul":
                        score = float(ans.get("noul", 0.0))
                        pred = score >= 0.5
                    else:
                        probs = ans.get("probabilities", {}) or {}
                        top = max(probs, key=lambda k: probs[k]) if probs else None
                        score = float(probs.get(top, 0.0)) if top else 0.0
                        raw_choice = ans.get("choice")
                        valid = {"secure", "vulnerable"} if args.task == "code_defect" else {"supported", "not supported"}
                        norm_choice, reason = normalize_choice(raw_choice, valid)
                        if norm_choice is None:
                            err = f"choice-parse: {reason}"
                        else:
                            pos_label = "vulnerable" if args.task == "code_defect" else "supported"
                            pred = (norm_choice == pos_label)
                results.append({"position": pos, "row_key": row["key"], "group_id": row["group_id"],
                                "rep": rep, "gold": bool(row["gold"]), "pred": pred, "score": score,
                                "raw_choice": raw_choice, "norm_choice": norm_choice,
                                "correct": bool(pred == bool(row["gold"])) if pred is not None else False,
                                "latency_ms": round(ms, 1), "calls": n_calls, "error": err})
                print(f"{pos+1}/{len(rows)} {rep} err={err}", flush=True)
    finally:
        allow_sleep()

    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "model": args.model,
              "task": args.task, "reps": reps, "seed": args.seed, "n_rows": len(rows),
              "predict_kwargs": budget_kw or None,
              "device": "cpu", "threads": args.threads, "laya_commit": git_sha(LAYA_ROOT),
              "project_git_sha": git_sha(ROOT),
              "windows_sleep_prevention_active": sleep_on,
              "peak_rss_mb": round(peak / 2**20, 1),
              "results": results,
              "status": "passed" if all(r["error"] is None for r in results) else "partial_failure"}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for rep in reps:
        sub = [r for r in results if r["rep"] == rep and r["error"] is None]
        acc = sum(r["correct"] for r in sub) / len(sub) if sub else None
        print(rep, "acc", round(acc, 4) if acc is not None else None, "n", len(sub))


if __name__ == "__main__":
    raise SystemExit(main())
