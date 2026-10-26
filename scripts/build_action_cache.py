#!/usr/bin/env python3
"""Build a per-example action cache for the NEAT controller.

Runs the frozen Laya model once per (example, strategy) and stores prediction,
confidence, correctness, latency and call count. NEAT later reads this cache
instead of calling Laya again.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAYA_ROOT = ROOT / "external" / "laya"
sys.path.insert(0, str(LAYA_ROOT))
sys.path.insert(0, str(LAYA_ROOT / "research" / "benchmarks" / "tournament"))

import psutil  # noqa: E402
import torch  # noqa: E402

import benchmark as bench  # noqa: E402
import laya  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(bench.DATASETS))
    ap.add_argument("--split", required=True)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--strategies", default="single,wide,tournament")
    ap.add_argument("--model", default="english")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_available():
        raise RuntimeError("CPU-only run refused: CUDA is available")

    load, _instructions = bench.DATASETS[args.dataset]
    names, rows = load(args.split)
    if args.limit and args.limit < len(rows):
        rows = random.Random(args.seed).sample(rows, args.limit)

    question = {"type": "choice", "instructions": bench.DATASETS[args.dataset][1],
                "criteria": {name.replace("_", " "): None for name in names}}
    agent = laya.load(args.model, device=args.device)
    few = dict(question, criteria=dict(list(question["criteria"].items())[:8]))
    for text, _gold in rows[:3]:
        agent.predict({"message": text}, {"q": few})

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    answer_fns = {name: bench.strategy(name, agent) for name in strategies}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    proc = psutil.Process(os.getpid())
    peak_rss = proc.memory_info().rss

    records = []
    started_run = time.perf_counter()
    with out_path.open("w", encoding="utf-8") as f:
        for idx, (text, gold) in enumerate(rows):
            gold_norm = gold.replace("_", " ")
            for name in strategies:
                t0 = time.perf_counter()
                error = None
                try:
                    answer, n_calls = answer_fns[name]({"message": text}, question)
                    choice = answer["choice"]
                    conf = answer["answer_confidence"]
                    confidence = answer.get("confidence")
                    probabilities = answer.get("probabilities", {})
                    act_probability = (answer.get("action") or {}).get("act_probability")
                    correct = int(choice == gold_norm)
                    # Derived, cache-friendly features for later controllers.
                    sorted_probs = sorted(probabilities.values(), reverse=True)
                    top1_prob = sorted_probs[0] if sorted_probs else None
                    top2_prob = sorted_probs[1] if len(sorted_probs) > 1 else None
                    margin = (top1_prob - top2_prob) if (top1_prob is not None and top2_prob is not None) else None
                    entropy = None
                    if probabilities:
                        import math
                        entropy = -sum(p * math.log(p) for p in probabilities.values() if p and p > 0)
                except Exception as exc:  # noqa: BLE001
                    error = f"{type(exc).__name__}: {exc}"
                    choice = None
                    conf = None
                    confidence = None
                    probabilities = {}
                    act_probability = None
                    correct = 0
                    n_calls = None
                    top1_prob = top2_prob = margin = entropy = None
                latency_ms = (time.perf_counter() - t0) * 1000.0
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
                    "confidence": conf,
                    "model_confidence": confidence,
                    "act_probability": act_probability,
                    "probabilities": probabilities,
                    "top1_prob": top1_prob,
                    "top2_prob": top2_prob,
                    "margin": margin,
                    "entropy": entropy,
                    "correct": bool(correct),
                    "latency_ms": round(latency_ms, 1),
                    "calls": n_calls,
                    "error": error,
                }
                records.append(rec)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                peak_rss = max(peak_rss, proc.memory_info().rss)
                print(f"{args.dataset} {idx+1}/{len(rows)} {name} correct={correct} {latency_ms:.0f}ms", flush=True)

    elapsed = time.perf_counter() - started_run

    # Per-strategy summary.
    by_strategy = {}
    for name in strategies:
        subset = [r for r in records if r["strategy"] == name]
        n = len(subset)
        lats = sorted(r["latency_ms"] for r in subset)
        def pct(q):
            if not lats: return None
            pos = min(len(lats) - 1, int(len(lats) * q))
            return lats[pos]
        by_strategy[name] = {
            "n": n,
            "accuracy": round(sum(r["correct"] for r in subset) / n, 4) if n else None,
            "mean_ms": round(sum(r["latency_ms"] for r in subset) / n, 1) if n else None,
            "p50_ms": pct(0.50),
            "p95_ms": pct(0.95),
            "max_ms": (max(lats) if lats else None),
            "min_ms": (min(lats) if lats else None),
            "calls_per_row": round(sum((r["calls"] or 0) for r in subset) / n, 3) if n else None,
            "errors": sum(r["error"] is not None for r in subset),
        }

    # Oracle upper bound: best action per example by correctness then lower latency.
    by_example = {}
    for r in records:
        by_example.setdefault(r["index"], []).append(r)
    oracle_correct = 0
    best_fixed_name = max(strategies, key=lambda s: (by_strategy[s]["accuracy"] or 0, - (by_strategy[s]["mean_ms"] or 0)))
    best_fixed_correct = sum(r["correct"] for r in records if r["strategy"] == best_fixed_name)
    for idx, items in by_example.items():
        best = max(items, key=lambda r: (r["correct"], -(r["calls"] or 99), -r["latency_ms"]))
        oracle_correct += int(best["correct"])
    n_examples = len(by_example)
    oracle_acc = oracle_correct / n_examples if n_examples else 0.0

    # All pairwise error overlaps.
    pairwise = []
    for i in range(len(strategies)):
        for j in range(i + 1, len(strategies)):
            a = {r["index"]: r["correct"] for r in records if r["strategy"] == strategies[i]}
            b = {r["index"]: r["correct"] for r in records if r["strategy"] == strategies[j]}
            if not a or not b:
                continue
            pairwise.append({
                "strategies": [strategies[i], strategies[j]],
                "both_wrong": sum((not a[k]) and (not b[k]) for k in a),
                "only_first_wrong": sum((not a[k]) and b[k] for k in a),
                "only_second_wrong": sum(a[k] and (not b[k]) for k in a),
                "both_correct": sum(a[k] and b[k] for k in a),
            })

    n_expected = n_examples * len(strategies)
    n_errors = sum(r["error"] is not None for r in records)
    incomplete = len(records) != n_expected or any(len([r for r in records if r["strategy"] == s]) != n_examples for s in strategies)
    if n_errors > 0 or incomplete:
        status = "partial_failure" if n_errors < n_expected else "failed"
    else:
        status = "passed"

    import datetime as _dt
    summary = {
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "dataset": args.dataset,
        "split": args.split,
        "seed": args.seed,
        "n_examples": n_examples,
        "strategies": strategies,
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
        "dataset_revision": bench.REVISIONS.get({"banking77": "mteb/banking77", "clinc150": "clinc/clinc_oos", "massive": "SetFit/amazon_massive_intent_en-US"}.get(args.dataset)),
        "laya_commit": None,
        "project_git_sha": None,
        "status": status,
        "errors": n_errors,
        "records": len(records),
        "expected_records": n_expected,
    }
    # Resolve commits if git is available.
    try:
        import subprocess
        summary["laya_commit"] = subprocess.run(
            ["git", "-C", str(LAYA_ROOT), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True).stdout.strip()
        summary["project_git_sha"] = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True).stdout.strip()
    except Exception:
        pass

    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
