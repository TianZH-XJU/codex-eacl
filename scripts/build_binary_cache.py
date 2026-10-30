#!/usr/bin/env python3
"""Gate 1 pilot cache for binary judgment tasks (SINGLE-first cascade).

Task adapters:
  code_defect : state = function source; single = noul vulnerable?;
                switch = binary choice secure/vulnerable; wide = rerun w/ budget.
  findver     : state = claim + relevant_context evidence; single = noul
                supported?; switch = binary choice supported/not supported;
                wide = rerun w/ budget.
Emits the same row schema as the ProcessBench pilot so
evaluate_strategy_complementarity.py runs unchanged. No test rows.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
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
sys.path.insert(0, str(LAYA_ROOT / "research" / "benchmarks" / "tournament"))

import laya  # noqa: E402

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


TASKS = {
    "code_defect": {
        "single_q": {"type": "noul", "instructions": "Is this function vulnerable?"},
        "switch_q": {"type": "choice", "instructions": "Is this function secure or vulnerable?",
                     "criteria": {"secure": None, "vulnerable": None}},
        "pos_label": "vulnerable", "render": lambda c: (c or "").replace(" ", "_"),
    },
    "findver": {
        "single_q": {"type": "noul", "instructions": "Is the claim supported by the evidence?"},
        "switch_q": {"type": "choice", "instructions": "Is the claim supported by the evidence?",
                     "criteria": {"supported": None, "not supported": None}},
        "pos_label": "supported", "render": lambda c: (c or "").replace(" ", "_"),
    },
}


def load_rows(task, raw_dir, findver_root, split, limit, seed):
    rng = random.Random(seed)
    if task == "code_defect":
        js_all = json.loads(Path(raw_dir, "function.json").read_text(encoding="utf-8"))
        idx = [int(x) for x in Path(raw_dir, f"{split}.txt").read_text(encoding="utf-8").split()]
        rng.shuffle(idx)
        out = []
        for i in idx[:limit]:
            out.append({"key": str(i), "text": js_all[i]["func"],
                        "gold_bool": bool(js_all[i]["target"]),
                        "gold_render": "vulnerable" if js_all[i]["target"] else "secure",
                        "meta": {"idx": i, "func_len": len(js_all[i]["func"]),
                                 "project": js_all[i].get("project", "?")}})
        return out
    rows = json.loads(Path(findver_root, "data", "testmini.json").read_text(encoding="utf-8"))
    manifest = json.loads(Path("results/findver/controller_split_manifest.json").read_text(encoding="utf-8"))
    ids = manifest["dev_example_ids"] if split == "dev" else manifest["train_example_ids"]
    by_id = {r["example_id"]: r for r in rows}
    rng.shuffle(ids)
    out = []
    for eid in ids[:limit]:
        r = by_id[eid]
        rep = json.loads(Path(findver_root, "financial_reports", r["report"]).read_text(encoding="utf-8"))
        ev = "\n".join(rep["context"][i]["context"] for i in r["relevant_context"])
        out.append({"key": eid, "text": f"Claim: {r['statement']}\n\nEvidence:\n{ev}",
                    "gold_bool": bool(r["entailment_label"]),
                    "gold_render": "supported" if r["entailment_label"] else "not_supported",
                    "meta": {"subset": r["subset"], "statement_len": len(r["statement"]),
                             "evidence_len": len(ev)}})
    return out


def row_features(answer):
    probs = (answer.get("probabilities") or {}) if answer else {}
    sp = sorted(probs.values(), reverse=True)
    top1 = sp[0] if sp else None
    top2 = sp[1] if len(sp) > 1 else None
    ent = -sum(p * math.log(p) for p in probs.values() if p and p > 0) if probs else None
    return {
        "answer_confidence": (answer.get("answer_confidence") if answer else None),
        "entropy_confidence": (answer.get("confidence") if answer else None),
        "act_probability": ((answer.get("action") or {}).get("act_probability") if answer else None,
        "top1_prob": top1, "top2_prob": top2,
        "margin": (top1 - top2) if (top1 is not None and top2 is not None) else None,
        "entropy": ent, "noul_p_true": (answer.get("noul") if answer and "noul" in answer else None),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["code_defect", "findver"])
    ap.add_argument("--split", default="dev")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--raw-dir", default="data/raw/codexglue_defect")
    ap.add_argument("--findver-root", default="external/FinDVer")
    ap.add_argument("--model", default="english")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_available():
        raise RuntimeError("CPU-only run refused: CUDA is available")

    cfg = TASKS[args.task]
    out_dir = Path(args.out_dir or {"code_defect": "results/codexglue_defect", "findver": "results/findver"}[args.task])
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.task, args.raw_dir, args.findver_root, "valid" if args.task == "code_defect" else args.split, args.limit, args.seed)

    agent = laya.load(args.model, device=args.device)
    agent.predict({"message": "warm up"}, {"q": dict(cfg["single_q"])})
    proc = psutil.Process(os.getpid())
    peak_rss = proc.memory_info().rss
    sleep_on = prevent_sleep()

    def run(qtype, text, wide=False):
        t0 = time.perf_counter()
        err, ans, calls = None, None, None
        try:
            q = dict(cfg[single_q_key(qtype)])
            kw = {"head_max_len": 512, "max_len": 1024} if wide else {}
            out = agent.predict({"message": text}, {"q": q}, **kw)
            ans = out["answers"]["q"]
            calls = 1
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
        return ans, calls, (time.perf_counter() - t0) * 1000.0, err

    def single_q_key(qt):
        return {"single": "single_q", "switch": "switch_q", "wide": "switch_q"}[qt]

    path = out_dir / f"pilot_{args.split}_cache.jsonl"
    try:
        with path.open("w", encoding="utf-8") as f:
            for pos, item in enumerate(rows):
                # SINGLE (noul).
                s_ans, s_calls, s_ms, s_err = run("single", item["text"])
                s_choice = cfg["render"](None)
                if s_ans is not None:
                    s_choice = "vulnerable" if s_ans.get("noul", 0) >= 0.5 else ("secure" if args.task == "code_defect" else "not_supported")
                    if args.task == "findver" and s_ans.get("noul", 0) >= 0.5:
                        s_choice = "supported"
                s_correct = int(s_choice == item["gold_render"]) if s_err is None else 0
                s_feat = row_features(s_ans)
                for name in ("single", "switch", "wide"):
                    if name == "single":
                        ans, n_calls, choice, correct, err, ms = s_ans, s_calls, s_choice, s_correct, s_err, s_ms
                    else:
                        ans, n_calls, ms, err = run(name, item["text"], wide=(name == "wide"))
                        if err is None:
                            choice = cfg["render"](ans.get("choice"))
                            correct = int(choice == item["gold_render"])
                        else:
                            choice, correct, n_calls = None, 0, None
                    cum_ms = s_ms + (0.0 if name == "single" else ms)
                    cum_calls = (s_calls or 0) + (0 if name == "single" else (n_calls or 0))
                    rec = {
                        "dataset": args.task, "split": f"pilot_{args.split}", "seed": args.seed,
                        "position": pos, "row_id": item["key"], "gold_render": item["gold_render"],
                        "gold_bool": item["gold_bool"], "n_candidates": 2,
                        "strategy": name, "choice": choice, "correct": bool(correct),
                        "latency_ms": round(ms, 1), "calls": n_calls, "error": err,
                        "strategy_features": row_features(ans),
                        "tournament": {"rounds": 0, "n": 2},
                        "initial_choice": s_choice, "initial_answer_confidence": s_feat["answer_confidence"],
                        "initial_entropy_confidence": s_feat["entropy_confidence"],
                        "initial_top1_prob": s_feat["top1_prob"], "initial_top2_prob": s_feat["top2_prob"],
                        "initial_margin": s_feat["margin"], "initial_entropy": s_feat["entropy"],
                        "initial_act_probability": s_feat["act_probability"],
                        "initial_noul_p_true": s_feat["noul_p_true"],
                        "initial_latency_ms": round(s_ms, 1), "initial_calls": s_calls,
                        "initial_error": s_err, "initial_n_options": 2,
                        "initial_text_len": len(item["text"]),
                        "incremental_latency_ms": round(0.0 if name == "single" else ms, 1),
                        "cumulative_latency_ms": round(cum_ms, 1), "cumulative_calls": cum_calls,
                        "meta": item["meta"],
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                peak_rss = max(peak_rss, proc.memory_info().rss)
                print(f"{args.task} {pos+1}/{len(rows)} single_ok={bool(s_correct)}", flush=True)
    finally:
        allow_sleep()

    summary = {"created_utc": datetime.now(timezone.utc).isoformat(), "dataset": args.task,
               "split": args.split, "seed": args.seed, "n": len(rows),
               "strategies": ["single", "switch", "wide"], "model": args.model,
               "device": args.device, "threads": args.threads,
               "laya_commit": git_sha(LAYA_ROOT), "project_git_sha": git_sha(ROOT),
               "windows_sleep_prevention_active": sleep_on,
               "peak_rss_mb": round(peak_rss / (1024 * 1024), 1), "status": "passed"}
    (out_dir / f"pilot_{args.split}_cache.summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("n", "status", "peak_rss_mb")}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
