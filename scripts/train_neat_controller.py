#!/usr/bin/env python3
"""Fit the already-locked NEAT configuration on the full 3-shot train cache.

Run only after reviewing results/pilot/banking77_neat_cv.json and choosing one
predeclared configuration without consulting the new confirmation set. The
pickle artifact is local/ignored; the JSON summary records its provenance.
No Laya calls, no dev/confirmation/test cache access.
"""
from __future__ import annotations
import argparse, atexit, ctypes, hashlib, json, os, pickle, sys, time
from datetime import datetime, timezone
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

def prevent_system_sleep():
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
from run_controller_baselines import features, single_row
from run_three_action import STRATEGIES, group, load_cache
from run_cv_stability import fit_choice_models, fit_numeric_models, fold_costs
from run_neat_cv import evolve

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-cache", default="results/pilot/banking77_train3shot_cache.jsonl")
    ap.add_argument("--mode", choices=["choice", "numeric"], default="choice")
    ap.add_argument("--topology", choices=["neat", "fixed"], default="neat")
    ap.add_argument("--lambda-per-ms", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--pop-size", type=int, default=30)
    ap.add_argument("--generations", type=int, default=20)
    ap.add_argument("--artifact", default="artifacts/checkpoints/banking77_neat_controller.pkl")
    ap.add_argument("--summary", default="results/pilot/banking77_neat_controller_train.json")
    args = ap.parse_args()
    train_path, artifact_path, summary_path = [(ROOT / p).resolve() for p in (args.train_cache, args.artifact, args.summary)]
    if not train_path.is_file():
        raise SystemExit(f"train cache not found: {train_path}")
    if artifact_path.exists() or summary_path.exists():
        raise SystemExit("Refusing to overwrite artifact or summary; choose a new output name")
    try:
        import numpy as np
        import neat  # noqa: F401
        from threadpoolctl import threadpool_limits
    except Exception as exc:
        raise SystemExit(f"Required offline dependency unavailable: {exc}") from exc
    rows = load_cache(train_path)
    by = group(rows)
    class_counts = {}
    for i in by:
        gold = str(single_row(by[i]).get("gold_normalized") or single_row(by[i]).get("gold") or "")
        class_counts[gold] = class_counts.get(gold, 0) + 1
    if len(class_counts) != 77 or len(set(class_counts.values())) != 1:
        raise SystemExit("Expected a balanced BANKING77 cache with 77 classes and equal examples/class")
    examples_per_class = next(iter(class_counts.values()))
    if examples_per_class not in (3, 5) or len(by) != 77 * examples_per_class:
        raise SystemExit("Expected either 3-shot (n=231) or 5-shot (n=385) cache")
    ids = sorted(by)
    Xraw = np.asarray([features(single_row(by[i])) for i in ids], dtype=float)
    if args.mode == "numeric":
        scaler, _ = fit_numeric_models(by, ids, class_weight=None)
        encoder = None
        X = scaler.transform(Xraw)
    else:
        scaler, encoder, _ = fit_choice_models(by, ids, class_weight=None)
        cats = np.asarray([[single_row(by[i]).get("initial_choice") or ""] for i in ids])
        X = np.hstack([scaler.transform(Xraw), encoder.transform(cats).toarray()])
    costs = fold_costs(by, ids)
    sleep_prevention_active = prevent_system_sleep()
    started = time.perf_counter()
    with threadpool_limits(limits=2):
        _net, genome_info, genome, config_str = evolve(
            X, ids, by, costs, args.lambda_per_ms, args.seed, args.pop_size, args.generations,
            fixed_topology=(args.topology == "fixed"))
    bundle = {
        "format_version": 1, "genome": genome, "config_text": config_str,
        "mode": args.mode, "topology": args.topology, "lambda_per_ms": args.lambda_per_ms,
        "n_train": len(ids), "examples_per_class": examples_per_class,
        "seed": args.seed, "pop_size": args.pop_size, "generations": args.generations,
        "scaler": scaler, "encoder": encoder, "strategies": STRATEGIES,
        "train_cache": str(train_path.relative_to(ROOT)) if train_path.is_relative_to(ROOT) else str(train_path),
        "training_costs": costs, "genome_summary": genome_info,
        "model_revision": "7b928d828b7b0e022f929d9bd2e44165aa270148",
        "dataset_revision": "18072d2685ea682290f7b8924d94c62acc19c0b2",
    }
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with artifact_path.open("xb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    try:
        sha = __import__("subprocess").run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                 capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        sha = None
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "project_git_sha": sha,
        "status": "passed", "laya_calls": 0,
        "mode": args.mode, "topology": args.topology, "lambda_per_ms": args.lambda_per_ms, "seed": args.seed,
        "windows_sleep_prevention_active": sleep_prevention_active,
        "pop_size": args.pop_size, "generations": args.generations,
        "n_train": len(ids), "n_classes": len({str(single_row(by[i]).get("gold_normalized") or single_row(by[i]).get("gold")) for i in ids}),
        "artifact": str(artifact_path), "artifact_sha256": digest,
        "training_costs": costs, "best_genome": genome_info,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "note": "Fit on train cache only; artifact is local and should not be committed.",
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
