#!/usr/bin/env python3
"""Stage 1B/1C: Laya-vs-Vela paired API smoke + Vela resource staircase.

ONE model per process (never co-resident). Canonical fixed rows (no test):
- processbench_choice: first train-split row of the frozen manifest
- code_noul: fixed valid idx
- findver_choice: first controller-dev example
Staircase mode builds synthetic text to approx token targets (actual usage
input_tokens recorded from the response).
"""
from __future__ import annotations

import argparse
import ctypes
import json
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


def load_laya():
    sys.path.insert(0, str(LAYA_ROOT / "research" / "benchmarks" / "tournament"))
    import laya  # noqa: E402
    return laya.load("english", device="cpu")


def load_vela():
    from transformers import AutoModel
    snap = ROOT / "models" / "vela"
    return AutoModel.from_pretrained(str(snap), trust_remote_code=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["laya", "vela"])
    ap.add_argument("--task", required=True,
                    choices=["processbench_choice", "code_noul", "findver_choice", "staircase"])
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_available():
        raise RuntimeError("CPU-only run refused: CUDA is available")

    proc = psutil.Process()
    rss_before = proc.memory_info().rss
    t_load0 = time.perf_counter()
    agent = load_laya() if args.model == "laya" else load_vela()
    load_s = time.perf_counter() - t_load0
    rss_after_load = proc.memory_info().rss
    peak = rss_after_load

    def build_inputs():
        if args.task == "processbench_choice":
            import json as js
            manifest = js.loads((ROOT / "results/processbench/split_manifest.json").read_text(encoding="utf-8"))
            rid = next(r["id"] for r in manifest["rows"]["train"] if r["file"] == "gsm8k.json")
            row = next(r for r in js.loads((ROOT / "data/raw/processbench/gsm8k.json").read_text(encoding="utf-8")) if r["id"] == rid)
            cands = [f"step {k}" for k in range(len(row["steps"]))] + ["ALL_STEPS_CORRECT"]
            state = {"message": row["problem"] + "\n\nReasoning steps:\n" + "\n".join(f"{k+1}. {s}" for k, s in enumerate(row["steps"]))}
            q = {"type": "choice", "instructions": "Which step contains the earliest error? Choose the sentinel if all steps are correct.",
                 "criteria": {c.replace("_", " "): None for c in [c.replace(" ", "_") for c in cands]}}
            return [("pb_choice", state, {"q": q}, row)]
        if args.task == "code_noul":
            import json as js
            js_all = {r.get("idx", -1): r for r in js.loads((ROOT / "data/raw/codexglue_defect/function.json").read_text(encoding="utf-8"))}
            valid = [int(x) for x in (ROOT / "data/raw/codexglue_defect/valid.txt").read_text(encoding="utf-8").split()]
            r = js_all[valid[0]]
            return [("code_noul", {"message": r["func"]},
                     {"q": {"type": "noul", "instructions": "Is this function vulnerable?"}}, r)]
        if args.task == "findver_choice":
            import json as js
            manifest = js.loads((ROOT / "results/findver/controller_split_manifest.json").read_text(encoding="utf-8"))
            eid = manifest["dev_example_ids"][0]
            data = {r["example_id"]: r for r in js.loads((ROOT / "external/FinDVer/data/testmini.json").read_text(encoding="utf-8"))}
            r = data[eid]
            rep = js.loads((ROOT / f"external/FinDVer/financial_reports/{r['report']}").read_text(encoding="utf-8"))
            ev = "\n".join(rep["context"][i]["context"] for i in r["relevant_context"])
            state = {"message": f"Claim: {r['statement']}\n\nEvidence context:\n{ev}"}
            q = {"type": "choice", "instructions": "Is the claim supported by the evidence?",
                 "criteria": {"supported": None, "not supported": None}}
            return [("findver_choice", state, {"q": q}, r)]
        sent = "The quick brown fox jumps over the lazy dog. "
        nrep = max(1, args.tokens // 12)
        return [(f"staircase_{args.tokens}", {"message": sent * nrep},
                 {"q": {"type": "noul", "instructions": "Is this text about animals?"}}, None)]

    cases = build_inputs()
    sleep_on = prevent_sleep()
    results = []
    try:
        for name, state, questions, ref in cases:
            t0 = time.perf_counter()
            err, out = None, None
            try:
                out = agent.system_one(state, questions) if args.model == "vela" else agent.predict(state, questions)
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {e}"
            ms = (time.perf_counter() - t0) * 1000.0
            peak = max(peak, proc.memory_info().rss)
            answers = (out.get("answers", {}) if out else {})
            usage = (out.get("usage", {}) if out else {})
            shape_ok = bool(answers) and all(
                isinstance(v, dict) and v.get("type") in ("choice", "noul", "score") for v in answers.values())
            results.append({"case": name, "latency_ms": round(ms, 1), "error": err,
                            "shape_ok": shape_ok, "usage": usage,
                            "answer_keys": {k: sorted(v.keys()) for k, v in answers.items()}})
            print(f"{name} shape_ok={shape_ok} ms={ms:.0f} usage={usage} err={err}", flush=True)
    finally:
        allow_sleep()

    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "model": args.model,
              "task": args.task, "device": "cpu", "threads": args.threads,
              "laya_commit": git_sha(LAYA_ROOT), "vela_snapshot": str((ROOT / "models/vela").resolve()),
              "load_seconds": round(load_s, 1),
              "rss_before_load_mb": round(rss_before / 2**20, 1),
              "rss_after_load_mb": round(rss_after_load / 2**20, 1),
              "peak_rss_mb": round(peak / 2**20, 1),
              "windows_sleep_prevention_active": sleep_on,
              "project_git_sha": git_sha(ROOT),
              "results": results,
              "status": "passed" if all(r["shape_ok"] for r in results) else "partial_failure"}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "peak_rss_mb", "load_seconds")}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
