#!/usr/bin/env python3
"""Build a per-example action cache for the NEAT controller.

Protocol: SINGLE is always executed first and supplies the controller-visible
features. For every strategy, the recorded latency/cost is cumulative under a
SINGLE-first cascade:
  SINGLE       -> cost = SINGLE
  WIDE         -> cost = SINGLE + WIDE
  TOURNAMENT   -> cost = SINGLE + TOURNAMENT

The cache stores both the initial SINGLE features and each strategy's outcome.
"""
from __future__ import annotations

import atexit
import argparse
import ctypes
import json
import math
import os
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAYA_ROOT = ROOT / "external" / "laya"
sys.path.insert(0, str(LAYA_ROOT))
sys.path.insert(0, str(LAYA_ROOT / "research" / "benchmarks" / "tournament"))

import psutil  # noqa: E402
import torch  # noqa: E402

import benchmark as bench  # noqa: E402
import laya  # noqa: E402

def prevent_system_sleep():
    """Prevent Windows system sleep during long local inference; restore at exit."""
    if os.name != "nt":
        return False
    es_continuous, es_system_required = 0x80000000, 0x00000001
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    set_state = kernel.SetThreadExecutionState
    set_state.argtypes, set_state.restype = [ctypes.c_uint], ctypes.c_uint
    if set_state(es_continuous | es_system_required) == 0:
        raise ctypes.WinError(ctypes.get_last_error())
    atexit.register(lambda: set_state(es_continuous))
    return True


DATASET_TO_REPO = {
    "banking77": "mteb/banking77",
    "clinc150": "clinc/clinc_oos",
    "massive": "SetFit/amazon_massive_intent_en-US",
}


def extract_features(answer):
    probabilities = answer.get("probabilities", {}) if answer else {}
    sorted_probs = sorted(probabilities.values(), reverse=True)
    top1 = sorted_probs[0] if sorted_probs else None
    top2 = sorted_probs[1] if len(sorted_probs) > 1 else None
    margin = (top1 - top2) if (top1 is not None and top2 is not None) else None
    entropy = None
    if probabilities:
        entropy = -sum(p * math.log(p) for p in probabilities.values() if p and p > 0)
    return {
        "choice": answer.get("choice") if answer else None,
        "answer_confidence": answer.get("answer_confidence") if answer else None,
        "entropy_confidence": answer.get("confidence") if answer else None,
        "act_probability": (answer.get("action") or {}).get("act_probability") if answer else None,
        "probabilities": probabilities,
        "top1_prob": top1,
        "top2_prob": top2,
        "margin": margin,
        "entropy": entropy,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(bench.DATASETS))
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=0, help="0 means all rows")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--strategies", default="single,wide,tournament")
    ap.add_argument("--model", default="english")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--manifest", default=None, help="Fixed split manifest JSON")
    ap.add_argument("--manifest-subset", default="dev", choices=["train_1shot", "train_3shot", "train_5shot", "dev", "train_all", "confirmation"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    sleep_prevention_active = prevent_system_sleep()
    if torch.cuda.is_available():
        raise RuntimeError("CPU-only run refused: CUDA is available")

    if args.manifest:
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        args.seed = manifest.get("seed", args.seed)
        source_split = manifest.get("source_split", "train")
        load, _instructions = bench.DATASETS[args.dataset]
        _names, all_rows = load(source_split)
        names = _names
        if args.manifest_subset == "train_1shot":
            indices = manifest["nested_train_indices"]["1"]
        elif args.manifest_subset == "train_3shot":
            indices = manifest["nested_train_indices"]["3"]
        elif args.manifest_subset == "train_5shot":
            indices = manifest["nested_train_indices"]["5"]
        elif args.manifest_subset == "train_all":
            indices = manifest["train_indices_all"]
        elif args.manifest_subset == "confirmation":
            indices = manifest["confirmation_indices"]
        else:
            indices = manifest["dev_indices"]
        rows = [all_rows[i] for i in indices]
        if args.limit and args.limit > 0 and args.limit < len(rows):
            rows = rows[:args.limit]
        args.split = args.manifest_subset
    else:
        load, _instructions = bench.DATASETS[args.dataset]
        names, rows = load(args.split)
        if args.limit and args.limit > 0 and args.limit < len(rows):
            rows = random.Random(args.seed).sample(rows, args.limit)

    question = {"type": "choice", "instructions": bench.DATASETS[args.dataset][1],
                "criteria": {name.replace("_", " "): None for name in names}}
    agent = laya.load(args.model, device=args.device)
    few = dict(question, criteria=dict(list(question["criteria"].items())[:8]))
    for text, _gold in rows[:3]:
        agent.predict({"message": text}, {"q": few})

    requested = [s.strip() for s in args.strategies.split(",") if s.strip()]
    record_strategies = list(dict.fromkeys(["single"] + requested))
    answer_fns = {name: bench.strategy(name, agent) for name in record_strategies}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    proc = psutil.Process(os.getpid())
    peak_rss = proc.memory_info().rss

    records = []
    started_run = time.perf_counter()
    with out_path.open("w", encoding="utf-8") as f:
        for idx, (text, gold) in enumerate(rows):
            gold_norm = gold.replace("_", " ")

            # 1. SINGLE first.
            t0 = time.perf_counter()
            single_error = None
            try:
                single_answer, single_calls = answer_fns["single"]({"message": text}, question)
                single_choice = single_answer["choice"]
                single_correct = int(single_choice == gold_norm)
            except Exception as exc:  # noqa: BLE001
                single_error = f"{type(exc).__name__}: {exc}"
                single_answer = None
                single_calls = None
                single_choice = None
                single_correct = 0
            single_latency_ms = (time.perf_counter() - t0) * 1000.0
            single_features = extract_features(single_answer)

            for name in record_strategies:
                if name == "single":
                    answer = single_answer
                    n_calls = single_calls
                    choice = single_choice
                    correct = single_correct
                    error = single_error
                    strategy_latency_ms = single_latency_ms
                else:
                    t0 = time.perf_counter()
                    error = None
                    try:
                        answer, n_calls = answer_fns[name]({"message": text}, question)
                        choice = answer["choice"]
                        correct = int(choice == gold_norm)
                    except Exception as exc:  # noqa: BLE001
                        error = f"{type(exc).__name__}: {exc}"
                        answer = None
                        choice = None
                        correct = 0
                        n_calls = None
                    strategy_latency_ms = (time.perf_counter() - t0) * 1000.0

                strategy_features = extract_features(answer)
                cumulative_latency_ms = single_latency_ms + (0.0 if name == "single" else strategy_latency_ms)
                cumulative_calls = (single_calls or 0) + (0 if name == "single" else (n_calls or 0))

                rec = {
                    "dataset": args.dataset,
                    "split": args.split,
                    "seed": args.seed,
                    "index": idx,
                    "text": text,
                    "gold": gold,
                    "gold_normalized": gold_norm,
                    "strategy": name,
                    "choice": choice,
                    "correct": bool(correct),
                    "latency_ms": round(strategy_latency_ms, 1),
                    "calls": n_calls,
                    "error": error,
                    # Strategy's own diagnostic output.
                    "strategy_answer_confidence": strategy_features["answer_confidence"],
                    "strategy_entropy_confidence": strategy_features["entropy_confidence"],
                    "strategy_probabilities": strategy_features["probabilities"],
                    "strategy_top1_prob": strategy_features["top1_prob"],
                    "strategy_top2_prob": strategy_features["top2_prob"],
                    "strategy_margin": strategy_features["margin"],
                    "strategy_entropy": strategy_features["entropy"],
                    "strategy_act_probability": strategy_features["act_probability"],
                    # SINGLE-first controller-visible features (identical for all strategy rows).
                    "initial_choice": single_features["choice"],
                    "initial_answer_confidence": single_features["answer_confidence"],
                    "initial_entropy_confidence": single_features["entropy_confidence"],
                    "initial_probabilities": single_features["probabilities"],
                    "initial_top1_prob": single_features["top1_prob"],
                    "initial_top2_prob": single_features["top2_prob"],
                    "initial_margin": single_features["margin"],
                    "initial_entropy": single_features["entropy"],
                    "initial_act_probability": single_features["act_probability"],
                    "initial_latency_ms": round(single_latency_ms, 1),
                    "initial_calls": single_calls,
                    "initial_error": single_error,
                    # Cascade cost accounting.
                    "incremental_latency_ms": round(0.0 if name == "single" else strategy_latency_ms, 1),
                    "cumulative_latency_ms": round(cumulative_latency_ms, 1),
                    "cumulative_calls": cumulative_calls,
                }
                records.append(rec)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                peak_rss = max(peak_rss, proc.memory_info().rss)
                print(f"{args.dataset} {idx+1}/{len(rows)} {name} correct={correct} strategy_ms={strategy_latency_ms:.0f} cumulative_ms={cumulative_latency_ms:.0f}", flush=True)

    elapsed = time.perf_counter() - started_run

    # Per-strategy summary.
    by_strategy = {}
    for name in record_strategies:
        subset = [r for r in records if r["strategy"] == name]
        n = len(subset)
        lats = sorted(r["latency_ms"] for r in subset)
        cum_lats = sorted(r["cumulative_latency_ms"] for r in subset)
        def pct(values, q):
            if not values: return None
            pos = min(len(values) - 1, int(len(values) * q))
            return values[pos]
        by_strategy[name] = {
            "n": n,
            "accuracy": round(sum(r["correct"] for r in subset) / n, 4) if n else None,
            "mean_ms": round(sum(r["latency_ms"] for r in subset) / n, 1) if n else None,
            "p50_ms": pct(lats, 0.50),
            "p95_ms": pct(lats, 0.95),
            "max_ms": (max(lats) if lats else None),
            "min_ms": (min(lats) if lats else None),
            "mean_cumulative_ms": round(sum(r["cumulative_latency_ms"] for r in subset) / n, 1) if n else None,
            "p95_cumulative_ms": pct(cum_lats, 0.95),
            "calls_per_row": round(sum((r["calls"] or 0) for r in subset) / n, 3) if n else None,
            "cumulative_calls_per_row": round(sum(r["cumulative_calls"] for r in subset) / n, 3) if n else None,
            "errors": sum(r["error"] is not None for r in subset),
            "initial_errors": sum(r["initial_error"] is not None for r in subset),
        }

    # Oracle upper bound on cumulative-cost-aware choices.
    by_example = {}
    for r in records:
        by_example.setdefault(r["index"], []).append(r)
    oracle_correct = 0
    best_fixed_name = max(record_strategies, key=lambda s: (by_strategy[s]["accuracy"] or 0, -(by_strategy[s]["mean_cumulative_ms"] or 0)))
    best_fixed_correct = sum(r["correct"] for r in records if r["strategy"] == best_fixed_name)
    for idx, items in by_example.items():
        best = max(items, key=lambda r: (r["correct"], -(r["cumulative_calls"]), -r["cumulative_latency_ms"]))
        oracle_correct += int(best["correct"])
    n_examples = len(by_example)
    oracle_acc = oracle_correct / n_examples if n_examples else 0.0

    # All pairwise error overlaps.
    pairwise = []
    for i in range(len(record_strategies)):
        for j in range(i + 1, len(record_strategies)):
            a = {r["index"]: r["correct"] for r in records if r["strategy"] == record_strategies[i]}
            b = {r["index"]: r["correct"] for r in records if r["strategy"] == record_strategies[j]}
            if not a or not b:
                continue
            pairwise.append({
                "strategies": [record_strategies[i], record_strategies[j]],
                "both_wrong": sum((not a[k]) and (not b[k]) for k in a),
                "only_first_wrong": sum((not a[k]) and b[k] for k in a),
                "only_second_wrong": sum(a[k] and (not b[k]) for k in a),
                "both_correct": sum(a[k] and b[k] for k in a),
            })

    n_expected = n_examples * len(record_strategies)
    n_errors = sum(r["error"] is not None for r in records) + sum(r["initial_error"] is not None for r in records)
    incomplete = len(records) != n_expected or any(len([r for r in records if r["strategy"] == s]) != n_examples for s in record_strategies)
    if n_errors > 0 or incomplete:
        status = "partial_failure" if n_errors < n_expected else "failed"
    else:
        status = "passed"

    laya_commit = None
    project_git_sha = None
    try:
        import subprocess
        laya_commit = subprocess.run(["git", "-C", str(LAYA_ROOT), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
        project_git_sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    except Exception:
        pass

    dirty = False
    try:
        import subprocess
        dirty = bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"], check=True, capture_output=True, text=True).stdout.strip())
    except Exception:
        pass

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "windows_sleep_prevention_active": sleep_prevention_active,
        "dataset": args.dataset,
        "split": args.split,
        "seed": args.seed,
        "n_examples": n_examples,
        "strategies": record_strategies,
        "by_strategy": by_strategy,
        "best_fixed_strategy": best_fixed_name,
        "best_fixed_accuracy": round(best_fixed_correct / n_examples, 4) if n_examples else None,
        "oracle_accuracy": round(oracle_acc, 4),
        "oracle_gain_over_best_fixed": round(oracle_acc - (best_fixed_correct / n_examples), 4) if n_examples else None,
        "pairwise_error_overlap": pairwise,
        "elapsed_seconds": round(elapsed, 1),
        "peak_rss_mb": round(peak_rss / (1024 * 1024), 1),
        "cpu_threads": args.threads,
        "device": args.device,
        "model": args.model,
        "dataset_revision": bench.REVISIONS.get(DATASET_TO_REPO.get(args.dataset)),
        "laya_commit": laya_commit,
        "project_git_sha": project_git_sha,
        "working_tree_dirty": dirty,
        "status": status,
        "errors": n_errors,
        "records": len(records),
        "expected_records": n_expected,
    }

    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
PY