#!/usr/bin/env python3
"""Determinism check: same genome, same rows, two passes in ONE process.

Answers whether menu-restricted accuracy on n=30 is repeatable within a
process. Prior Vela runs scored the same genome 0.80 (exhaustive) vs 0.833
(random) across processes; this isolates within-process flips from
cross-process ones. Saves per-row pass1/pass2 choices. No test rows.
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
sys.path.insert(0, str(ROOT / "scripts"))
from build_menu_bank import load_pool  # noqa: E402
from run_evo_menu_pilot import build_menus  # noqa: E402

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def prevent_sleep():
    try:
        return bool(ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED))
    except Exception:
        return False


def allow_sleep():
    try:
        ctypes.windll.SetThreadExecutionState(ES_CONTINUOUS)
    except Exception:
        pass


def git_sha(path):
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                              check=True, capture_output=True, text=True).stdout.strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="banking77", choices=["banking77", "clinc150"])
    ap.add_argument("--model", default="vela", choices=["laya", "vela"])
    ap.add_argument("--n-rows", type=int, default=30)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_available():
        raise RuntimeError("CPU-only run refused: CUDA is available")

    genome = {"retriever": "tfidf-char", "k": 10, "norm": "raw", "order": "shuffled"}
    train = load_pool(args.dataset, "search_train", 2000, args.seed)
    dev_rows = load_pool(args.dataset, "search_dev", None, args.seed + 1)[:args.n_rows]
    menus = build_menus(genome, train, dev_rows)

    if args.model == "laya":
        sys.path.insert(0, str(ROOT / "external" / "laya" / "research" / "benchmarks" / "tournament"))
        import laya  # noqa: E402
        agent = laya.load("english", device="cpu")

        def call(text, criteria):
            return agent.predict({"message": text},
                                 {"q": {"type": "choice",
                                        "instructions": "Which intent does this message express?",
                                        "criteria": criteria}})["answers"]["q"]
    else:
        from transformers import AutoModel
        agent = AutoModel.from_pretrained(str(ROOT / "models/vela"), trust_remote_code=True)

        def call(text, criteria):
            return agent.system_one(text, {"q": {"type": "choice",
                "instructions": "Which intent does this message express?",
                "criteria": criteria}})["answers"]["q"]

    proc = psutil.Process()
    peak = proc.memory_info().rss
    sleep_on = prevent_sleep()
    rows = []
    try:
        for row in menus:
            q = {c.replace("_", " "): None for c in row["menu"]}
            gold = row["label"].replace(" ", "_").lower()
            choices = []
            for _ in range(2):
                t0 = time.perf_counter()
                err, choice = None, None
                try:
                    ans = call(row["text"], q)
                    choice = str(ans.get("choice", "")).replace(" ", "_").lower()
                except Exception as e:  # noqa: BLE001
                    err = f"{type(e).__name__}: {e}"
                choices.append({"choice": choice, "error": err,
                                "ms": round((time.perf_counter() - t0) * 1000.0, 1)})
            peak = max(peak, proc.memory_info().rss)
            rows.append({"key": row["key"], "gold": gold, "menu": row["menu"],
                         "pass1": choices[0], "pass2": choices[1],
                         "flip": bool(choices[0]["choice"] != choices[1]["choice"]),
                         "correct1": bool(choices[0]["choice"] == gold),
                         "correct2": bool(choices[1]["choice"] == gold)})
    finally:
        allow_sleep()

    n = len(rows)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "task": args.dataset,
              "model": args.model, "genome": genome, "n": n, "seed": args.seed,
              "device": "cpu", "threads": args.threads, "project_git_sha": git_sha(ROOT),
              "windows_sleep_prevention_active": sleep_on, "peak_rss_mb": round(peak / 2**20, 1),
              "acc_pass1": round(sum(r["correct1"] for r in rows) / n, 4),
              "acc_pass2": round(sum(r["correct2"] for r in rows) / n, 4),
              "n_flips": sum(r["flip"] for r in rows),
              "rows": rows,
              "status": "passed" if all(r["pass1"]["error"] is None and r["pass2"]["error"] is None for r in rows) else "partial_failure"}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("acc_pass1", "acc_pass2", "n_flips", "status")}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
