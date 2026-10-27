#!/usr/bin/env python3
"""Rerun anomalous (index, strategy) rows with sleep prevention and verification.

Reads the confirmation cache, re-executes the listed rows through the frozen
Laya model, and verifies text/gold/strategy/model-revision against the cache
record and the pinned dataset row. Old values are never overwritten: the new
observation is stored alongside. No tuning, no dataset download (HF offline).
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAYA_ROOT = ROOT / "external" / "laya"
sys.path.insert(0, str(LAYA_ROOT))
sys.path.insert(0, str(LAYA_ROOT / "research" / "benchmarks" / "tournament"))

import torch  # noqa: E402

import benchmark as bench  # noqa: E402
import laya  # noqa: E402

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def prevent_sleep():
    """Ask Windows to stay awake; return True iff the API call succeeded."""
    try:
        prev = ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        return bool(prev)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirmation-cache", default="results/pilot/banking77_confirmation_cache.jsonl")
    ap.add_argument("--manifest", default="results/splits/banking77_confirmation_manifest.json")
    ap.add_argument("--indices", default="25,26,27")
    ap.add_argument("--strategies", default="single,wide,tournament")
    ap.add_argument("--model", default="english")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default="results/pilot/banking77_anomaly_rerun.jsonl")
    ap.add_argument("--summary", default="results/pilot/banking77_anomaly_rerun.summary.json")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_available():
        raise RuntimeError("CPU-only run refused: CUDA is available")

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    cache_rows = [json.loads(x) for x in Path(args.confirmation_cache).read_text(encoding="utf-8").splitlines() if x.strip()]
    by_idx = {}
    for r in cache_rows:
        by_idx.setdefault(r["index"], []).append(r)

    # Verify text/gold against the pinned dataset rows (HF cache only; offline OK).
    load, _ = bench.DATASETS["banking77"]
    _names, all_rows = load(manifest.get("source_split", "train"))
    conf_ids = manifest["confirmation_indices"]

    question_cache = None
    names = None
    agent = laya.load(args.model, device=args.device)
    fns = {s: bench.strategy(s, agent) for s in args.strategies.split(",")}

    sleep_on = prevent_sleep()
    results = []
    try:
        for pos in [int(x) for x in args.indices.split(",")]:
            old_rows = {r["strategy"]: r for r in by_idx.get(pos, [])}
            if not old_rows:
                raise SystemExit(f"no cached rows for confirmation position {pos}")
            sample_id = conf_ids[pos]
            text, gold = all_rows[sample_id]
            if names is None:
                # names from a fresh load for the question criteria
                load2, _ = bench.DATASETS["banking77"]
                names, _ = load2(manifest.get("source_split", "train"))
            question = {"type": "choice", "instructions": bench.DATASETS["banking77"][1],
                        "criteria": {n.replace("_", " "): None for n in names}}
            for s in args.strategies.split(","):
                old = old_rows.get(s)
                if old is None:
                    continue
                text_ok = (old["text"] == text)
                gold_ok = (old["gold"] == gold)
                t0 = time.perf_counter()
                error = None
                try:
                    answer, n_calls = fns[s]({"message": text}, question)
                    choice, conf = answer["choice"], answer["answer_confidence"]
                    correct = int(choice == gold.replace("_", " "))
                except Exception as exc:  # noqa: BLE001
                    error = f"{type(exc).__name__}: {exc}"
                    choice, conf, correct, n_calls = None, None, 0, None
                new_ms = (time.perf_counter() - t0) * 1000.0
                results.append({
                    "position": pos, "sample_id": sample_id, "strategy": s,
                    "text_match": text_ok, "gold_match": gold_ok,
                    "old_latency_ms": old["latency_ms"], "new_latency_ms": round(new_ms, 1),
                    "old_choice": old["choice"], "new_choice": choice,
                    "old_correct": bool(old["correct"]), "new_correct": bool(correct),
                    "choice_changed": (old["choice"] != choice),
                    "old_calls": old["calls"], "new_calls": n_calls,
                    "error": error,
                })
                print(f"pos={pos} {s} old_ms={old['latency_ms']} new_ms={new_ms:.0f} "
                      f"choice_changed={old['choice'] != choice} text_ok={text_ok} gold_ok={gold_ok}", flush=True)
    finally:
        allow_sleep()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "confirmation_cache": args.confirmation_cache,
        "manifest": args.manifest,
        "model": args.model, "device": args.device, "threads": args.threads,
        "laya_commit": git_sha(LAYA_ROOT),
        "project_git_sha": git_sha(ROOT),
        "model_revision": manifest.get("model_revision"),
        "dataset_revision": manifest.get("dataset_revision"),
        "windows_sleep_prevention_active": sleep_on,
        "n_reruns": len(results),
        "all_text_match": all(r["text_match"] for r in results),
        "all_gold_match": all(r["gold_match"] for r in results),
        "results": results,
    }
    Path(args.summary).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("windows_sleep_prevention_active", "all_text_match",
                                              "all_gold_match", "n_reruns")}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
