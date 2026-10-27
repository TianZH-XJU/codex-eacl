#!/usr/bin/env python3
"""Targeted replay of three sleep-polluted BANKING77 confirmation actions.

Reads the frozen manifest/cache, verifies exact sample text and gold label,
then runs SINGLE followed by WIDE for cache position 25 and Tournament for
positions 26/27. Original records are immutable. On Windows the process asks
the OS to prevent system sleep and restores the previous state on exit.
CPU-only, local-cache/offline, 2 threads, no test split.
"""
from __future__ import annotations
import argparse, ctypes, hashlib, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path

for k, v in {
    "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
    "OPENBLAS_NUM_THREADS": "2", "TOKENIZERS_PARALLELISM": "false",
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
}.items():
    os.environ[k] = v

ROOT = Path(__file__).resolve().parents[1]
LAYA = ROOT / "external" / "laya"
sys.path.insert(0, str(LAYA))
sys.path.insert(0, str(LAYA / "research" / "benchmarks" / "tournament"))
TARGETS = ((25, "wide"), (26, "tournament"), (27, "tournament"))
MODEL_REV = "7b928d828b7b0e022f929d9bd2e44165aa270148"
DATA_REV = "18072d2685ea682290f7b8924d94c62acc19c0b2"

def norm(s):
    return str(s or "").replace("_", " ").strip()

def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]

def prevent_sleep():
    if os.name != "nt":
        return False, lambda: None
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    set_state = kernel.SetThreadExecutionState
    set_state.argtypes, set_state.restype = [ctypes.c_uint], ctypes.c_uint
    if set_state(ES_CONTINUOUS | ES_SYSTEM_REQUIRED) == 0:
        raise ctypes.WinError(ctypes.get_last_error())
    return True, lambda: set_state(ES_CONTINUOUS)

def resolve_targets(manifest, cache, dataset_rows):
    indices = manifest.get("confirmation_indices")
    if not isinstance(indices, list) or len(indices) != 77:
        raise ValueError("confirmation manifest must contain 77 confirmation_indices")
    grouped = {}
    for r in cache:
        grouped.setdefault(int(r["index"]), []).append(r)
    out = []
    for pos, strategy in TARGETS:
        if pos not in grouped:
            raise ValueError(f"cache position {pos} missing")
        single = next((r for r in grouped[pos] if r.get("strategy") == "single"), None)
        old = next((r for r in grouped[pos] if r.get("strategy") == strategy), None)
        if single is None or old is None:
            raise ValueError(f"position {pos} lacks single/{strategy}")
        source_idx = int(indices[pos])
        text, gold = dataset_rows[source_idx]
        if text != single.get("text"):
            raise ValueError(f"text mismatch at cache position {pos}; refusing wrong sample")
        if norm(gold) != norm(single.get("gold_normalized") or single.get("gold")):
            raise ValueError(f"gold mismatch at cache position {pos}; refusing wrong sample")
        out.append({
            "cache_position": pos, "source_dataset_index": source_idx,
            "strategy": strategy, "text": text, "gold_normalized": norm(gold),
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "old_single": single, "old_target": old,
        })
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="results/splits/banking77_confirmation_manifest.json")
    ap.add_argument("--cache", default="results/pilot/banking77_confirmation_cache.jsonl")
    ap.add_argument("--out", default="results/pilot/banking77_anomaly_rerun.jsonl")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    if args.threads != 2:
        raise SystemExit("Protocol requires exactly 2 CPU threads")
    manifest_path, cache_path, out_path = [(ROOT / p).resolve() for p in (args.manifest, args.cache, args.out)]
    summary_path = out_path.with_suffix(".summary.json")
    if out_path.exists() or summary_path.exists():
        raise SystemExit("Output exists; refusing to overwrite old rerun results")
    if not manifest_path.is_file() or not cache_path.is_file():
        raise SystemExit("Manifest/cache missing; run from project root")
    import torch, psutil, benchmark as bench, laya
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_available():
        raise RuntimeError("CPU-only replay refused because CUDA is available")
    manifest, cache = json.loads(manifest_path.read_text(encoding="utf-8")), load_jsonl(cache_path)
    load, instructions = bench.DATASETS["banking77"]
    labels, dataset_rows = load(manifest.get("source_split", "train"))
    targets = resolve_targets(manifest, cache, dataset_rows)
    awake, restore = prevent_sleep()
    results, t_run = [], time.perf_counter()
    peak_rss = 0
    try:
        agent = laya.load("english", device="cpu")
        q = {"type": "choice", "instructions": instructions,
             "criteria": {name.replace("_", " "): None for name in labels}}
        few = dict(q, criteria=dict(list(q["criteria"].items())[:8]))
        for text, _ in dataset_rows[:3]:
            agent.predict({"message": text}, {"q": few})  # standard, unmeasured warmup
        funcs = {s: bench.strategy(s, agent) for s in ("single", "wide", "tournament")}
        proc = psutil.Process()
        peak_rss = proc.memory_info().rss
        for item in targets:
            payload = {"message": item["text"]}
            t = time.perf_counter()
            single, calls_single = funcs["single"](payload, q)
            ms_single = (time.perf_counter() - t) * 1000
            t = time.perf_counter()
            answer, calls_target = funcs[item["strategy"]](payload, q)
            ms_target = (time.perf_counter() - t) * 1000
            row = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "cache_position": item["cache_position"],
                "source_dataset_index": item["source_dataset_index"],
                "strategy_rerun": item["strategy"], "text_sha256": item["text_sha256"],
                "gold_normalized": item["gold_normalized"],
                "old_single_choice": item["old_single"].get("choice"),
                "new_single_choice": single.get("choice"),
                "old_single_correct": bool(item["old_single"].get("correct")),
                "new_single_correct": norm(single.get("choice")) == item["gold_normalized"],
                "old_target_choice": item["old_target"].get("choice"),
                "new_target_choice": answer.get("choice"),
                "old_target_correct": bool(item["old_target"].get("correct")),
                "new_target_correct": norm(answer.get("choice")) == item["gold_normalized"],
                "old_target_latency_ms": item["old_target"].get("latency_ms"),
                "old_cumulative_latency_ms": item["old_target"].get("cumulative_latency_ms"),
                "rerun_single_latency_ms": round(ms_single, 1),
                "rerun_target_latency_ms": round(ms_target, 1),
                "rerun_cumulative_latency_ms": round(ms_single + ms_target, 1),
                "rerun_cumulative_calls": (calls_single or 0) + (calls_target or 0),
                "target_error": None,
            }
            results.append(row)
            peak_rss = max(peak_rss, proc.memory_info().rss)
            print(json.dumps({k: row[k] for k in ("cache_position", "strategy_rerun",
                "old_target_latency_ms", "rerun_target_latency_ms",
                "old_target_correct", "new_target_correct")}), flush=True)
    finally:
        restore()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("x", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "banking77", "dataset_revision": DATA_REV, "model_revision": MODEL_REV,
        "split": "confirmation", "manifest_seed": manifest.get("seed"),
        "n_rows": len(results), "targets": [{k: x[k] for k in ("cache_position", "source_dataset_index", "strategy")} for x in targets],
        "windows_sleep_prevention_active": awake, "cpu_threads": 2, "device": "cpu",
        "status": "passed" if len(results) == 3 else "failed",
        "elapsed_seconds_including_load_warmup": round(time.perf_counter() - t_run, 1),
        "peak_rss_mb": round(peak_rss / (1024 * 1024), 1),
        "latency_by_strategy": {
            strategy: {
                "n": sum(x["strategy_rerun"] == strategy for x in results),
                "old_target_latency_ms": round(sum(x["old_target_latency_ms"] for x in results if x["strategy_rerun"] == strategy) /
                                               max(1, sum(x["strategy_rerun"] == strategy for x in results)), 1),
                "rerun_target_latency_ms": round(sum(x["rerun_target_latency_ms"] for x in results if x["strategy_rerun"] == strategy) /
                                                 max(1, sum(x["strategy_rerun"] == strategy for x in results)), 1),
            } for strategy in sorted({x["strategy_rerun"] for x in results})
        },
        "accuracy_changed_on_rerun": sum(x["old_target_correct"] != x["new_target_correct"] for x in results),
        "choices_changed_on_rerun": sum(x["old_target_choice"] != x["new_target_choice"] for x in results),
        "original_values_overwritten": False,
        "note": "Rerun costs are separate diagnostics; they do not replace original confirmation costs.",
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["status"] == "passed" else 1

if __name__ == "__main__":
    raise SystemExit(main())
