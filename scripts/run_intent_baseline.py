#!/usr/bin/env python3
"""A1 native baselines for intent tasks (Laya/Vela single Choice, same rows).

Samples search_train groups from v2 manifests, runs one default Choice call
per row per model (separate processes per model), and writes pilot-cache-schema
JSONL so existing analysis scripts work unchanged. No test rows.
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


def load_search_rows(dataset, limit, seed):
    manifest = json.loads((ROOT / "results/v4/data_manifest" / f"{dataset}_groups.json").read_text(encoding="utf-8"))
    if dataset == "clinc150":
        import pyarrow.parquet as pq
        all_rows = {}
        for split in ("train", "validation", "test"):
            p = ROOT / f"data/raw/clinc150/plus/{split}-00000-of-00001.parquet"
            names = json.loads(pq.read_schema(p).metadata[b"huggingface"])["info"]["features"]["intent"]["names"]
            for r in pq.read_table(p).to_pylist():
                all_rows[(split, r["text"])] = names[r["intent"]]
        # members store train row indices
        raw_train = pq.read_table(ROOT / "data/raw/clinc150/plus/train-00000-of-00001.parquet").to_pylist()
        names = json.loads(pq.read_schema(ROOT / "data/raw/clinc150/plus/train-00000-of-00001.parquet").metadata[b"huggingface"])["info"]["features"]["intent"]["names"]
        pool = []
        for g in manifest["splits"]["search_train"]:
            for i in manifest["members"].get(g, []):
                r = raw_train[i]
                pool.append({"key": f"train:{i}", "text": r["text"], "gold": names[r["intent"]],
                             "group_id": g, "gold_idx": names.index(names[r["intent"]])})
    else:
        import pyarrow.parquet as pq
        raw_train = pq.read_table(ROOT / "data/raw/banking77/data/train-00000-of-00001.parquet").to_pylist()
        pool = []
        for g in manifest["splits"]["search_train"]:
            for i in manifest["members"].get(g, []):
                r = raw_train[i]
                pool.append({"key": f"train:{i}", "text": r["text"], "gold": r["label_text"],
                             "group_id": g, "gold_idx": None})
    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool[:limit]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["clinc150", "banking77"])
    ap.add_argument("--model", required=True, choices=["laya", "vela"])
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_available():
        raise RuntimeError("CPU-only run refused: CUDA is available")

    rows = load_search_rows(args.dataset, args.limit, args.seed)
    labels = sorted({r["gold"] for r in rows})
    if args.model == "laya":
        sys.path.insert(0, str(LAYA_ROOT / "research" / "benchmarks" / "tournament"))
        import laya  # noqa: E402
        agent = laya.load("english", device="cpu")
        def call(text, question):
            return agent.predict({"message": text}, {"q": question})["answers"]["q"], 1
    else:
        from transformers import AutoModel
        agent = AutoModel.from_pretrained(str(ROOT / "models/vela"), trust_remote_code=True)
        def call(text, question):  # noqa: E306
            return agent.system_one(text, {"q": question})["answers"]["q"], 1

    proc = psutil.Process(os.getpid())
    peak = proc.memory_info().rss
    sleep_on = prevent_sleep()
    out = []
    try:
        for pos, row in enumerate(rows):
            cands = labels
            q = {"type": "choice", "instructions": "Which intent does this message express?",
                 "criteria": {c.replace("_", " "): None for c in cands}}
            t0 = time.perf_counter()
            err, ans, n_calls = None, None, None
            try:
                ans, n_calls = call(row["text"], q)
                choice = ans.get("choice")
                correct = int((choice or "").replace(" ", "_") == row["gold"].replace(" ", "_"))
            except Exception as e:  # noqa: BLE001
                err, choice, correct = f"{type(e).__name__}: {e}", None, 0
            ms = (time.perf_counter() - t0) * 1000.0
            probs = (ans.get("probabilities") or {}) if ans else {}
            sp = sorted(probs.values(), reverse=True)
            out.append({
                "dataset": args.dataset, "split": "pilot_search", "seed": args.seed,
                "position": pos, "row_id": row["key"], "gold_rendered": row["gold"],
                "gold_normalized": row["gold"].replace("_", " "), "n_candidates": len(cands),
                "strategy": "single", "choice": choice, "correct": bool(correct),
                "latency_ms": round(ms, 1), "calls": n_calls, "error": err,
                "strategy_features": {}, "tournament": {"rounds": 0, "n": len(cands)},
                "initial_choice": choice,
                "initial_answer_confidence": ans.get("answer_confidence") if ans else None,
                "initial_latency_ms": round(ms, 1), "initial_calls": n_calls, "initial_error": err,
                "initial_n_options": len(cands), "initial_text_len": len(row["text"]),
                "cumulative_latency_ms": round(ms, 1), "cumulative_calls": n_calls,
                "group_id": row["group_id"]})
            peak = max(peak, proc.memory_info().rss)
            print(f"{pos+1}/{len(rows)} correct={bool(correct)} ms={ms:.0f}", flush=True)
    finally:
        allow_sleep()

    summary = {"created_utc": datetime.now(timezone.utc).isoformat(), "dataset": args.dataset,
               "model": args.model, "split": "pilot_search", "seed": args.seed, "n": len(out),
               "accuracy": round(sum(r["correct"] for r in out) / len(out), 4) if out else None,
               "device": "cpu", "threads": args.threads,
               "laya_commit": git_sha(LAYA_ROOT), "project_git_sha": git_sha(ROOT),
               "windows_sleep_prevention_active": sleep_on, "peak_rss_mb": round(peak / 2**20, 1),
               "status": "passed" if all(r["error"] is None for r in out) else "partial_failure"}
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    path.with_suffix("").with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("accuracy", "status", "peak_rss_mb")}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
