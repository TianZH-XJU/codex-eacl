#!/usr/bin/env python3
"""Gate 0: CodeXGLUE defect Noul vs binary-Choice adapter check (small Laya run).

Runs on official VALID rows only. Compares Laya noul ("is this function
vulnerable?") with a neutral two-item choice ("secure" vs "vulnerable").
Records structured outputs, agreement, and P(true) behavior. No conclusions
drawn here; this validates the adapters before any routing experiment.
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


CHOICE_CRITERIA = {"secure": None, "vulnerable": None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw/codexglue_defect")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--model", default="english")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default="results/codexglue_defect/adapter_smoke.json")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_available():
        raise RuntimeError("CPU-only run refused: CUDA is available")

    import random
    valid_idx = [int(x) for x in Path(args.raw_dir, "valid.txt").read_text(encoding="utf-8").split()]
    rng = random.Random(args.seed)
    rng.shuffle(valid_idx)
    chosen = valid_idx[:args.limit]
    js_all = json.loads(Path(args.raw_dir, "function.json").read_text(encoding="utf-8"))

    agent = laya.load(args.model, device=args.device)
    warm = {"type": "noul", "instructions": "Is this function vulnerable?"}
    agent.predict({"message": "int f() { return 0; }"}, {"q": warm})

    proc = psutil.Process()
    peak = proc.memory_info().rss
    sleep_on = prevent_sleep()
    rows = []
    try:
        for pos, idx in enumerate(chosen):
            func, target = js_all[idx]["func"], js_all[idx]["target"]
            state = {"message": func}
            t0 = time.perf_counter()
            n_err = None
            try:
                n_out = agent.predict(state, {"q": {"type": "noul", "instructions": "Is this function vulnerable?"}})
                n_ans = n_out["answers"]["q"]
            except Exception as e:  # noqa: BLE001
                n_err, n_ans = f"{type(e).__name__}: {e}", None
            n_ms = (time.perf_counter() - t0) * 1000.0
            t0 = time.perf_counter()
            c_err = None
            try:
                c_out = agent.predict(state, {"q": {"type": "choice",
                    "instructions": "Is this function secure or vulnerable?",
                    "criteria": dict(CHOICE_CRITERIA)}})
                c_ans = c_out["answers"]["q"]
            except Exception as e:  # noqa: BLE001
                c_err, c_ans = f"{type(e).__name__}: {e}", None
            c_ms = (time.perf_counter() - t0) * 1000.0
            n_p = (n_ans.get("noul") if n_ans else None)
            c_choice = (c_ans.get("choice") if c_ans else None)
            agree = (n_p is not None and c_choice is not None
                     and ((n_p >= 0.5) == (c_choice.replace(" ", "_") == "vulnerable")))
            rows.append({"idx": idx, "target": target, "func_len": len(func),
                         "noul_p_true": n_p, "noul_error": n_err, "noul_ms": round(n_ms, 1),
                         "choice": c_choice, "choice_error": c_err, "choice_ms": round(c_ms, 1),
                         "agree": agree})
            peak = max(peak, proc.memory_info().rss)
            print(f"{pos+1}/{len(chosen)} idx={idx} target={target} agree={agree}", flush=True)
    finally:
        allow_sleep()

    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "task": "codexglue_defect",
              "split": "valid", "seed": args.seed, "n": len(rows),
              "model": args.model, "device": args.device, "threads": args.threads,
              "laya_commit": git_sha(LAYA_ROOT), "project_git_sha": git_sha(ROOT),
              "windows_sleep_prevention_active": sleep_on,
              "peak_rss_mb": round(peak / (1024 * 1024), 1),
              "structured_ok": all(r["noul_error"] is None and r["choice_error"] is None for r in rows),
              "agreement_rate": round(sum(r["agree"] for r in rows) / len(rows), 4) if rows else None,
              "rows": rows, "status": "passed" if all(r["noul_error"] is None and r["choice_error"] is None for r in rows) else "partial_failure"}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("structured_ok", "agreement_rate", "status")}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
