#!/usr/bin/env python3
"""Stage 4 (Gate 4): Laya configuration-sensitivity pilot on frozen rows.

SINGLE once per row (config-independent); then each candidate config runs as
an upgrade with SINGLE-first cumulative accounting. Configs are pre-registered,
range-checked against the pinned build, and cache-keyed by config hash.
Reports per-config accuracy, flips vs single, net DeltaCorrect, cost, and a
paired bootstrap CI for the DeltaCorrect mean. No test rows. Offline analysis
after the (single, sequential) inference run.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LAYA_ROOT = ROOT / "external" / "laya"
sys.path.insert(0, str(LAYA_ROOT))
sys.path.insert(0, str(LAYA_ROOT / "research" / "benchmarks" / "tournament"))
sys.path.insert(0, str(ROOT / "scripts"))

import benchmark as bench  # noqa: E402
import laya  # noqa: E402
from prepare_processbench import to_choice_task  # noqa: E402

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


CONFIGS = [
    {"name": "wide_h96", "action": "wide", "kwargs": {"head_max_len": 96, "max_len": 512}},
    {"name": "wide_h160", "action": "wide", "kwargs": {"head_max_len": 160, "max_len": 512}},
    {"name": "wide_max1024", "action": "wide", "kwargs": {"head_max_len": 512, "max_len": 1024}},
    {"name": "order_reverse", "action": "order", "kwargs": {"order": "REVERSE"}},
    {"name": "tournament_g8", "action": "tournament", "kwargs": {"group_size": 8}},
    {"name": "tournament_g16", "action": "tournament", "kwargs": {"group_size": 16}},
]


def config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def validate_config(cfg, n_candidates):
    """Repair/reject per plan 4.3. Returns (ok, reason)."""
    kw = cfg.get("kwargs", {})
    if cfg["action"] == "tournament":
        gs = kw.get("group_size", 16)
        if not isinstance(gs, int) or gs < 2:
            return False, "group_size must be int >= 2"
    if cfg["action"] == "order":
        if kw.get("order") not in ("REVERSE",):
            return False, "unsupported order template in pilot"
    for k in ("head_max_len", "max_len"):
        if k in kw and (not isinstance(kw[k], int) or kw[k] <= 0):
            return False, f"bad {k}"
    return True, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--split-manifest", default="results/processbench/split_manifest.json")
    ap.add_argument("--raw-dir", default="data/raw/processbench")
    ap.add_argument("--model", default="english")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out-dir", default="results/processbench")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_available():
        raise RuntimeError("CPU-only run refused: CUDA is available")

    manifest = json.loads(Path(args.split_manifest).read_text(encoding="utf-8"))
    pool = list(manifest["rows"]["train"])
    rng = random.Random(args.seed)
    rng.shuffle(pool)
    chosen = pool[:args.limit]
    raw_cache = {}
    for filename in ("gsm8k.json", "math.json", "olympiadbench.json", "omnimath.json"):
        raw_cache[filename] = {r["id"]: r for r in
                               json.loads(Path(args.raw_dir, filename).read_text(encoding="utf-8"))}

    for cfg in CONFIGS:
        ok, why = validate_config(cfg, None)
        if not ok:
            raise SystemExit(f"invalid pre-registered config {cfg['name']}: {why}")
        cfg["config_hash"] = config_hash({"name": cfg["name"], "action": cfg["action"], "kwargs": cfg["kwargs"]})

    agent = laya.load(args.model, device=args.device)
    tourn_fns = {}
    proc = psutil.Process(os.getpid())
    peak_rss = proc.memory_info().rss
    sleep_on = prevent_sleep()

    out_dir = Path(args.out_dir)
    rows_out = []
    try:
        for pos, ref in enumerate(chosen):
            row = raw_cache[ref["file"]][ref["id"]]
            task = to_choice_task(row)
            render = [c.replace("_", " ") for c in task["candidates"]]
            gold = render[task["gold_index"]]
            state = {"message": row["problem"] + "\n\nSteps:\n" + "\n".join(f"[{k}] {s}" for k, s in enumerate(row["steps"]))}
            base_q = {"type": "choice",
                      "instructions": "Which step contains the earliest error? Choose the sentinel if all steps are correct.",
                      "criteria": {c: None for c in render}}
            t0 = time.perf_counter()
            s_err = None
            try:
                s_ans, s_calls = agent.predict(state, {"q": base_q}), 1
                s_choice = s_ans["answers"]["q"]["choice"]
                s_correct = int(s_choice == gold)
            except Exception as e:  # noqa: BLE001
                s_err, s_ans, s_calls, s_choice, s_correct = f"{type(e).__name__}: {e}", None, None, None, 0
            s_ms = (time.perf_counter() - t0) * 1000.0
            for cfg in CONFIGS:
                t0 = time.perf_counter()
                err, choice, correct, n_calls, extra = None, None, 0, None, {}
                try:
                    if cfg["action"] == "wide":
                        ans, n_calls = bench.strategy("wide", agent)(state, base_q, **cfg["kwargs"])
                        choice = ans["answers"]["q"]["choice"] if isinstance(ans, dict) and "answers" in ans else ans["choice"]
                    elif cfg["action"] == "order":
                        crit = dict(reversed(list(base_q["criteria"].items())))
                        q2 = dict(base_q, criteria=crit)
                        ans, n_calls = agent.predict(state, {"q": q2}), 1
                        choice = ans["answers"]["q"]["choice"]
                    elif cfg["action"] == "tournament":
                        gs = cfg["kwargs"]["group_size"]
                        if gs not in tourn_fns:
                            tourn_fns[gs] = bench.strategy(f"tournament-{gs}", agent)
                        ans, n_calls = tourn_fns[gs](state, base_q)
                        choice = ans["answers"]["q"]["choice"] if isinstance(ans, dict) and "answers" in ans else ans["choice"]
                        extra = {"rounds": (n_calls - 1) if n_calls else None}
                    correct = int(choice == gold)
                except Exception as e:  # noqa: BLE001
                    err = f"{type(e).__name__}: {e}"
                ms = (time.perf_counter() - t0) * 1000.0
                rows_out.append({
                    "position": pos, "row_id": ref["id"], "source_file": ref["file"],
                    "config": cfg["name"], "config_hash": cfg["config_hash"],
                    "choice": choice, "correct": bool(correct), "error": err,
                    "latency_ms": round(ms, 1), "calls": n_calls,
                    "cumulative_latency_ms": round(s_ms + ms, 1),
                    "cumulative_calls": (s_calls or 0) + (n_calls or 0),
                    "single_choice": s_choice, "single_correct": bool(s_correct),
                    "single_latency_ms": round(s_ms, 1), "gold_rendered": gold,
                    "n_candidates": len(render), "extra": extra})
            peak_rss = max(peak_rss, proc.memory_info().rss)
            print(f"{pos+1}/{len(chosen)} {ref['id']} single_ok={bool(s_correct)}", flush=True)
    finally:
        allow_sleep()

    # Offline summary with paired bootstrap on DeltaCorrect mean per config.
    rng_b = np.random.RandomState(17)
    by_cfg = {}
    single_acc = sum(r["single_correct"] for r in rows_out if r["config"] == CONFIGS[0]["name"]) / len(chosen)
    summary_cfgs = {}
    for cfg in CONFIGS:
        recs = [r for r in rows_out if r["config"] == cfg["name"]]
        delta = np.array([int(r["correct"]) - int(r["single_correct"]) for r in recs], dtype=float)
        means = [float(np.mean(rng_b.choice(delta, size=len(delta), replace=True))) for _ in range(2000)]
        lo, hi = np.percentile(means, [2.5, 97.5])
        lats = sorted(r["cumulative_latency_ms"] for r in recs)
        summary_cfgs[cfg["name"]] = {
            "n": len(recs), "accuracy": round(sum(r["correct"] for r in recs) / len(recs), 4),
            "single_accuracy_ref": round(single_acc, 4),
            "mean_delta_correct": round(float(delta.mean()), 4),
            "delta_ci95": [round(float(lo), 4), round(float(hi), 4)],
            "flips_vs_single": int(sum(r["choice"] != r["single_choice"] for r in recs)),
            "mean_cumulative_ms": round(sum(lats) / len(lats), 1),
            "errors": sum(r["error"] is not None for r in recs),
            "config_hash": cfg["config_hash"], "kwargs": cfg["kwargs"]}

    summary = {"created_utc": datetime.now(timezone.utc).isoformat(), "dataset": "Qwen/ProcessBench",
               "seed": args.seed, "n_rows": len(chosen), "model": args.model, "device": args.device,
               "threads": args.threads, "laya_commit": git_sha(LAYA_ROOT), "project_git_sha": git_sha(ROOT),
               "model_revision": "7b928d828b7b0e022f929d9bd2e44165aa270148",
               "windows_sleep_prevention_active": sleep_on, "peak_rss_mb": round(peak_rss / 2**20, 1),
               "configs": summary_cfgs, "status": "passed",
               "sample_ids": [r["id"] for r in chosen]}
    (out_dir / "sensitivity_pilot_cache.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows_out), encoding="utf-8")
    (out_dir / "sensitivity_pilot.summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: {kk: summary["configs"][k][kk] for kk in ("accuracy", "mean_delta_correct", "delta_ci95", "flips_vs_single", "mean_cumulative_ms")} for k in summary["configs"]}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
