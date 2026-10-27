#!/usr/bin/env python3
"""One-shot evaluation of a locked NEAT artifact on a fresh held-out cache.

Fits comparators on the 3-shot train cache only, evaluates every method on the
fresh NEAT confirmation cache, and reports paired uncertainty. Never tunes on
confirmation labels. The confirmation cache must have been generated from the
new manifest that excludes all previously used train/dev/confirmation rows.
"""
from __future__ import annotations
import argparse, hashlib, json, os, pickle, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_controller_baselines import features, single_row
from run_three_action import STRATEGIES, choose_action, group, load_cache
from run_cv_stability import (
    fit_choice_models, fit_numeric_models, fold_costs,
    predict_proba_choice, predict_proba_numeric,
)
from run_neat_cv import metrics, fit_threshold

def bootstrap_pair(by, candidate, baseline, reps=5000, seed=17):
    import numpy as np
    ids = sorted(candidate)
    diff = []
    both_correct = both_wrong = cand_only = base_only = 0
    for i in ids:
        cr = next(r for r in by[i] if r["strategy"] == candidate[i])
        br = next(r for r in by[i] if r["strategy"] == baseline[i])
        c, b = int(bool(cr["correct"])), int(bool(br["correct"]))
        diff.append(c - b)
        both_correct += int(c == 1 and b == 1)
        both_wrong += int(c == 0 and b == 0)
        cand_only += int(c == 1 and b == 0)
        base_only += int(c == 0 and b == 1)
    rng = np.random.RandomState(seed)
    draws = [float(np.mean(rng.choice(diff, size=len(diff), replace=True))) for _ in range(reps)]
    return {
        "candidate_minus_baseline_accuracy": round(float(np.mean(diff)), 4),
        "bootstrap95_lo": round(float(np.percentile(draws, 2.5)), 4),
        "bootstrap95_hi": round(float(np.percentile(draws, 97.5)), 4),
        "bootstrap": "paired, unstratified over samples because confirmation has one example/class",
        "reps": reps, "seed": seed,
        "mcnemar_counts": {"both_correct": both_correct, "both_wrong": both_wrong,
                           "candidate_only_correct": cand_only, "baseline_only_correct": base_only},
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-cache", default="results/pilot/banking77_train3shot_cache.jsonl")
    ap.add_argument("--confirmation-manifest", default="results/splits/banking77_neat_confirmation_manifest.json")
    ap.add_argument("--confirmation-cache", default="results/pilot/banking77_neat_confirmation_cache.jsonl")
    ap.add_argument("--artifact", default="artifacts/checkpoints/banking77_neat_controller.pkl")
    ap.add_argument("--out", default="results/pilot/banking77_neat_confirmation_result.json")
    args = ap.parse_args()
    train_path, manifest_path, cache_path, artifact_path, out_path = [
        (ROOT / p).resolve() for p in (
            args.train_cache, args.confirmation_manifest, args.confirmation_cache,
            args.artifact, args.out)]
    if out_path.exists():
        raise SystemExit(f"Refusing to overwrite one-shot evaluation output: {out_path}")
    for p in (train_path, manifest_path, cache_path, artifact_path):
        if not p.is_file():
            raise SystemExit(f"Required input missing: {p}")
    try:
        import neat, numpy as np
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:
        raise SystemExit(f"Required offline dependency unavailable: {exc}") from exc

    train_by = group(load_cache(train_path))
    conf_by = group(load_cache(cache_path))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cache_summary_path = cache_path.with_suffix(".summary.json")
    if not cache_summary_path.is_file():
        raise SystemExit(f"Cache summary missing: {cache_summary_path}")
    cache_summary = json.loads(cache_summary_path.read_text(encoding="utf-8"))
    if manifest.get("n_confirmation") != 77 or len(manifest.get("confirmation_indices", [])) != 77:
        raise SystemExit("Expected fresh one-example-per-class confirmation manifest (n=77)")
    if len(train_by) not in (231, 385) or len(conf_by) != 77:
        raise SystemExit(f"Wrong cache sizes: train={len(train_by)} (expected 231 or 385) confirmation={len(conf_by)}")
    if len({str(single_row(conf_by[i]).get("gold_normalized") or single_row(conf_by[i]).get("gold")) for i in conf_by}) != 77:
        raise SystemExit("Confirmation cache is not one sample per each of 77 classes")
    if cache_summary.get("dataset") != "banking77" or cache_summary.get("status") != "passed":
        raise SystemExit("Confirmation cache summary is not a passed BANKING77 run")
    if cache_summary.get("n_examples") != 77 or cache_summary.get("records") != 231:
        raise SystemExit("Confirmation cache summary must have 77 examples / 231 action records")
    if cache_summary.get("seed") != manifest.get("seed") or cache_summary.get("split") != "confirmation":
        raise SystemExit("Cache seed/split does not match the fresh confirmation manifest")
    excluded = manifest.get("excluded_index_counts") or {}
    if excluded.get("previous_confirmation") != 77 or not manifest.get("old_confirmation_manifest_sha256"):
        raise SystemExit("Fresh confirmation manifest provenance/exclusion record is missing")

    with artifact_path.open("rb") as f:
        bundle = pickle.load(f)
    if bundle.get("format_version") != 1:
        raise SystemExit("Unsupported NEAT artifact format")
    if int(bundle.get("n_train", -1)) != len(train_by):
        raise SystemExit("Training cache size differs from the controller artifact provenance")
    lam = float(bundle["lambda_per_ms"])
    mode = bundle["mode"]
    costs = fold_costs(train_by, sorted(train_by))
    # Train baselines using train-only labels/features and train-estimated costs.
    ids = sorted(train_by)
    num_scaler, num_models = fit_numeric_models(train_by, ids, class_weight=None)
    choice_scaler, enc, choice_models = fit_choice_models(train_by, ids, class_weight=None)
    train_acc = {s: sum(int(bool(next(r for r in train_by[i] if r["strategy"] == s)["correct"])) for i in ids) / len(ids)
                 for s in STRATEGIES}
    fixed_best = max(STRATEGIES, key=lambda s: train_acc[s])
    upgrade = max(("wide", "tournament"), key=lambda s: train_acc[s])
    threshold = fit_threshold(train_by, ids, upgrade)

    with tempfile.TemporaryDirectory(prefix="neat_laya_eval_cfg_") as tmp:
        cfg_path = Path(tmp) / "config.ini"
        cfg_path.write_text(bundle["config_text"], encoding="utf-8")
        cfg = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                          neat.DefaultSpeciesSet, neat.DefaultStagnation, str(cfg_path))
        net = neat.nn.FeedForwardNetwork.create(bundle["genome"], cfg)
        choices = {name: {} for name in (
            "neat", "numeric_lr", "choice_lr", "threshold_cascade", "fixed_train_best",
            "fixed_single", "fixed_wide", "fixed_tournament")}
        for i in sorted(conf_by):
            sr = single_row(conf_by[i])
            x = np.asarray([features(sr)], dtype=float)
            if mode == "numeric":
                z = bundle["scaler"].transform(x)[0]
            else:
                cat = np.asarray([[sr.get("initial_choice") or ""]])
                z = np.hstack([bundle["scaler"].transform(x), bundle["encoder"].transform(cat).toarray()])[0]
            choices["neat"][i] = STRATEGIES[int(np.argmax(net.activate(z)))]
            choices["numeric_lr"][i] = choose_action(predict_proba_numeric(num_scaler, num_models, sr), costs, lam)
            choices["choice_lr"][i] = choose_action(predict_proba_choice(choice_scaler, enc, choice_models, sr), costs, lam)
            confidence = float(sr.get("initial_answer_confidence") or 0.0)
            choices["threshold_cascade"][i] = "single" if confidence >= threshold else upgrade
            choices["fixed_train_best"][i] = fixed_best
            for s in STRATEGIES:
                choices[f"fixed_{s}"][i] = s

    result_metrics = {k: metrics(conf_by, v) for k, v in choices.items()}
    comparisons = {}
    for baseline in ("threshold_cascade", "fixed_train_best", "fixed_tournament", "choice_lr", "numeric_lr"):
        comparisons[f"neat_minus_{baseline}"] = bootstrap_pair(conf_by, choices["neat"], choices[baseline])

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "banking77", "split": "fresh_neat_confirmation",
        "manifest_seed": manifest.get("seed"),
        "manifest_excluded_counts": manifest.get("excluded_index_counts"),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "confirmation_cache_summary": str(cache_summary_path.relative_to(ROOT)) if cache_summary_path.is_relative_to(ROOT) else str(cache_summary_path),
        "confirmation_cache_records": cache_summary.get("records"),
        "model_artifact_sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
        "neat_config": {k: bundle.get(k) for k in ("mode", "lambda_per_ms", "seed", "pop_size", "generations", "genome_summary")},
        "training_costs": costs,
        "threshold_cascade": {"upgrade_target": upgrade, "threshold": threshold},
        "fixed_train_best": fixed_best,
        "metrics": result_metrics,
        "paired_comparisons": comparisons,
        "evaluated_once": True, "laya_calls_in_evaluator": 0,
        "confirmation_labels_used_for_training_or_selection": False,
        "status": "passed",
        "note": "One-shot held-out evaluation; do not tune using this result. The separate cache-build step measured real Laya action costs.",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "neat": result_metrics["neat"],
                      "baselines": {k: v for k, v in result_metrics.items() if k != "neat"},
                      "paired_comparisons": comparisons}, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
