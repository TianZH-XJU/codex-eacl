#!/usr/bin/env python3
"""Gate 1 pilot cache for ProcessBench math Choice (SINGLE-first cascade).

Samples seeded train/dev pilot subsets from the frozen problem-grouped split
manifest, runs SINGLE then WIDE and TOURNAMENT per row, and records prediction,
correctness (official exact-match rule), latency, calls, tournament rounds,
and controller-visible SINGLE features. No test rows. Single process, CPU.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
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
sys.path.insert(0, str(ROOT / "scripts"))

import benchmark as bench  # noqa: E402
import laya  # noqa: E402
from prepare_processbench import (  # noqa: E402
    ALL_CORRECT_SENTINEL, REPO_ID as PB_REPO, REVISION as PB_REVISION,
    to_choice_task,
)

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


def row_features(answer):
    probs = (answer.get("probabilities") or {}) if answer else {}
    sp = sorted(probs.values(), reverse=True)
    top1 = sp[0] if sp else None
    top2 = sp[1] if len(sp) > 1 else None
    ent = None
    if probs:
        ent = -sum(p * math.log(p) for p in probs.values() if p and p > 0)
    return {
        "answer_confidence": (answer.get("answer_confidence") if answer else None),
        "entropy_confidence": (answer.get("confidence") if answer else None),
        "act_probability": ((answer.get("action") or {}).get("act_probability") if answer else None),
        "probabilities": probs,
        "top1_prob": top1, "top2_prob": top2,
        "margin": (top1 - top2) if (top1 is not None and top2 is not None) else None,
        "entropy": ent,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-manifest", default="results/processbench/split_manifest.json")
    ap.add_argument("--raw-dir", default="data/raw/processbench")
    ap.add_argument("--train-limit", type=int, default=120)
    ap.add_argument("--dev-limit", type=int, default=60)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--strategies", default="single,wide,tournament")
    ap.add_argument("--model", default="english")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out-dir", default="results/processbench")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_available():
        raise RuntimeError("CPU-only run refused: CUDA is available")

    import random
    manifest = json.loads(Path(args.split_manifest).read_text(encoding="utf-8"))
    rng = random.Random(args.seed)
    subsets = {}
    for split, limit in (("train", args.train_limit), ("dev", args.dev_limit)):
        pool = list(manifest["rows"][split])
        rng.shuffle(pool)
        subsets[split] = pool[:limit]

    raw_cache = {}
    for filename in ("gsm8k.json", "math.json", "olympiadbench.json", "omnimath.json"):
        raw_cache[filename] = {r["id"]: r for r in
                               json.loads(Path(args.raw_dir, filename).read_text(encoding="utf-8"))}

    agent = laya.load(args.model, device=args.device)
    requested = [s.strip() for s in args.strategies.split(",") if s.strip()]
    record_strategies = list(dict.fromkeys(["single"] + requested))
    answer_fns = {name: bench.strategy(name, agent) for name in record_strategies}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    proc = psutil.Process(os.getpid())
    peak_rss = proc.memory_info().rss
    sleep_on = prevent_sleep()

    totals = {}
    try:
        for split in ("train", "dev"):
            path = out_dir / f"pilot_{split}_cache.jsonl"
            with path.open("w", encoding="utf-8") as f:
                for pos, ref in enumerate(subsets[split]):
                    row = raw_cache[ref["file"]][ref["id"]]
                    task = to_choice_task(row)
                    render = [c.replace("_", " ") for c in task["candidates"]]
                    gold_rendered = render[task["gold_index"]]
                    state = {"message": row["problem"] + "\n\nSteps:\n" + "\n".join(
                        f"[{k}] {s}" for k, s in enumerate(row["steps"]))}
                    question = {"type": "choice",
                                "instructions": "Which step contains the earliest error? Choose the sentinel if all steps are correct.",
                                "criteria": {c: None for c in render}}
                    # SINGLE first.
                    t0 = time.perf_counter()
                    s_err = None
                    try:
                        s_ans, s_calls = answer_fns["single"](state, question)
                        s_choice = s_ans["choice"]
                        s_correct = int(s_choice == gold_rendered)
                    except Exception as exc:  # noqa: BLE001
                        s_err = f"{type(exc).__name__}: {exc}"
                        s_ans, s_calls, s_choice, s_correct = None, None, None, 0
                    s_ms = (time.perf_counter() - t0) * 1000.0
                    s_feat = row_features(s_ans)
                    for name in record_strategies:
                        if name == "single":
                            ans, n_calls, choice, correct, err, strat_ms = s_ans, s_calls, s_choice, s_correct, s_err, s_ms
                            tinfo = {"rounds": 0, "n": len(render)}
                        else:
                            t0 = time.perf_counter()
                            err = None
                            try:
                                ans, n_calls = answer_fns[name](state, question)
                                choice = ans["choice"]
                                correct = int(choice == gold_rendered)
                            except Exception as exc:  # noqa: BLE001
                                err = f"{type(exc).__name__}: {exc}"
                                ans, n_calls, choice, correct = None, None, None, 0
                            strat_ms = (time.perf_counter() - t0) * 1000.0
                            tinfo = None
                            if name == "tournament" and err is None:
                                # bench.strategy returns calls = rounds + 1 final call.
                                tinfo = {"rounds": (n_calls - 1) if n_calls else None, "n": len(render)}
                        cum_ms = s_ms + (0.0 if name == "single" else strat_ms)
                        cum_calls = (s_calls or 0) + (0 if name == "single" else (n_calls or 0))
                        rec = {
                            "dataset": "processbench", "split": f"pilot_{split}", "seed": args.seed,
                            "position": pos, "row_id": ref["id"], "source_file": ref["file"],
                            "gold_rendered": gold_rendered, "gold_index": task["gold_index"],
                            "n_candidates": len(render), "n_steps": task["n_steps"],
                            "strategy": name, "choice": choice, "correct": bool(correct),
                            "latency_ms": round(strat_ms, 1), "calls": n_calls, "error": err,
                            "strategy_features": row_features(ans),
                            "tournament": tinfo,
                            "initial_choice": s_feat and s_choice, "initial_answer_confidence": s_feat["answer_confidence"],
                            "initial_entropy_confidence": s_feat["entropy_confidence"],
                            "initial_top1_prob": s_feat["top1_prob"], "initial_top2_prob": s_feat["top2_prob"],
                            "initial_margin": s_feat["margin"], "initial_entropy": s_feat["entropy"],
                            "initial_act_probability": s_feat["act_probability"],
                            "initial_latency_ms": round(s_ms, 1), "initial_calls": s_calls,
                            "initial_error": s_err, "initial_n_options": len(render),
                            "initial_text_len": len(state["message"]),
                            "incremental_latency_ms": round(0.0 if name == "single" else strat_ms, 1),
                            "cumulative_latency_ms": round(cum_ms, 1), "cumulative_calls": cum_calls,
                        }
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    peak_rss = max(peak_rss, proc.memory_info().rss)
                    print(f"{split} {pos+1}/{len(subsets[split])} {ref['id']} single_ok={bool(s_correct)}", flush=True)
            totals[split] = len(subsets[split])
    finally:
        allow_sleep()

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "Qwen/ProcessBench", "data_revision": PB_REVISION,
        "split_manifest_seed": manifest.get("seed"), "pilot_seed": args.seed,
        "train_limit": args.train_limit, "dev_limit": args.dev_limit,
        "n_train": totals.get("train", 0), "n_dev": totals.get("dev", 0),
        "strategies": record_strategies, "model": args.model, "device": args.device,
        "threads": args.threads, "laya_commit": git_sha(LAYA_ROOT),
        "project_git_sha": git_sha(ROOT), "model_revision": "7b928d828b7b0e022f929d9bd2e44165aa270148",
        "windows_sleep_prevention_active": sleep_on,
        "peak_rss_mb": round(peak_rss / (1024 * 1024), 1),
        "sample_ids": {k: [r["id"] for r in v] for k, v in subsets.items()},
        "status": "passed",
    }
    (out_dir / "pilot_cache.summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("n_train", "n_dev", "status", "peak_rss_mb",
                                             "windows_sleep_prevention_active")}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
