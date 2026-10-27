#!/usr/bin/env python3
"""Three-fold offline stability check on the fixed 3-shot train cache.

Each true class has 3 train samples; fold k holds out the k-th sample of
every class (77 OOF samples) and fits on the other 154. Every sample is OOF
exactly once (231 OOF predictions total ~ 2-shot training scale).

Model A: locked 6 numeric features. Model B: 6 numeric + one-hot
initial_choice (C=0.2, fixed, no re-tuning). Each model is run with
class_weight=None (primary probability estimator) and class_weight="balanced"
(control). Scaler, encoder, models and per-action costs are fit on the
fold-train subset only. Fixed lambda grid, no dev data, no Laya calls.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from run_controller_baselines import features, single_row
from run_three_action import LAMBDAS, STRATEGIES, choose_action

C_NUMERIC = 1.0
C_CHOICE = 0.2
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 17


def load_cache(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def group(rows):
    by = {}
    for r in rows:
        by.setdefault(r["index"], []).append(r)
    return by


def build_folds(by):
    """Stratify by gold label; fold k holds out the k-th sample of each class."""
    by_label = defaultdict(list)
    for idx, items in by.items():
        gold = single_row(items).get("gold_normalized") or single_row(items).get("gold", "")
        by_label[gold].append(idx)
    for label in by_label:
        by_label[label].sort()
    assert all(len(v) == 3 for v in by_label.values()), "expected 3 samples per class"
    folds = []
    for k in range(3):
        test = sorted(v[k] for v in by_label.values())
        test_set = set(test)
        train = sorted(i for i in by.keys() if i not in test_set)
        folds.append({"train": train, "test": test})
    return folds


def fit_numeric_models(by, train_idx, class_weight):
    X = np.array([features(single_row(by[i])) for i in train_idx], dtype=float)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    models = {}
    for s in STRATEGIES:
        y = np.array([int(next(x for x in by[i] if x["strategy"] == s)["correct"]) for i in train_idx])
        clf = LogisticRegression(max_iter=1000, random_state=17, class_weight=class_weight, C=C_NUMERIC)
        clf.fit(Xs, y)
        models[s] = clf
    return scaler, models


def fit_choice_models(by, train_idx, class_weight):
    Xn = np.array([features(single_row(by[i])) for i in train_idx], dtype=float)
    cats = np.array([[single_row(by[i]).get("initial_choice") or ""] for i in train_idx])
    scaler = StandardScaler().fit(Xn)
    enc = OneHotEncoder(handle_unknown="ignore").fit(cats)
    Xtr = np.hstack([scaler.transform(Xn), enc.transform(cats).toarray()])
    models = {}
    for s in STRATEGIES:
        y = np.array([int(next(x for x in by[i] if x["strategy"] == s)["correct"]) for i in train_idx])
        clf = LogisticRegression(max_iter=1000, random_state=17, class_weight=class_weight, C=C_CHOICE)
        clf.fit(Xtr, y)
        models[s] = clf
    return scaler, enc, models


def predict_proba_numeric(scaler, models, row):
    X = scaler.transform(np.array([features(row)], dtype=float))
    return {s: float(models[s].predict_proba(X)[0][list(models[s].classes_).index(1)]
                        if 1 in models[s].classes_ else 0.0) for s in STRATEGIES}


def predict_proba_choice(scaler, enc, models, row):
    Xn = scaler.transform(np.array([features(row)], dtype=float))
    C = enc.transform(np.array([[row.get("initial_choice") or ""]])).toarray()
    X = np.hstack([Xn, C])
    out = {}
    for s in STRATEGIES:
        cls = list(models[s].classes_)
        p = models[s].predict_proba(X)[0]
        out[s] = float(p[cls.index(1)]) if 1 in cls else 0.0
    return out


def fold_costs(by, train_idx):
    costs = {}
    for s in STRATEGIES:
        lats = [next(x for x in by[i] if x["strategy"] == s)["cumulative_latency_ms"] for i in train_idx]
        calls = [next(x for x in by[i] if x["strategy"] == s)["cumulative_calls"] for i in train_idx]
        costs[s] = {"mean_cumulative_ms": float(sum(lats) / len(lats)),
                    "mean_cumulative_calls": float(sum(calls) / len(calls))}
    return costs


def stratified_bootstrap_ci(golds, diff, reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED):
    """Paired bootstrap CI for mean(diff), resampling within each gold class."""
    rng = np.random.RandomState(seed)
    by_label = defaultdict(list)
    for g, d in zip(golds, diff):
        by_label[g].append(d)
    labels = sorted(by_label)
    arr = {lab: np.array(by_label[lab], dtype=float) for lab in labels}
    means = []
    for _ in range(reps):
        vals = [rng.choice(arr[lab], size=len(arr[lab]), replace=True).mean() for lab in labels]
        means.append(float(sum(v * len(arr[lab]) for v, lab in zip(vals, labels)) / len(diff)))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean": round(float(np.mean(means)), 4), "ci95_lo": round(float(lo), 4), "ci95_hi": round(float(hi), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-cache", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    by = group(load_cache(args.train_cache))
    folds = build_folds(by)
    golds = [(single_row(by[i]).get("gold_normalized") or single_row(by[i]).get("gold", "")) for i in by]

    # Reference: fixed-strategy accuracy on all 231 train samples.
    fixed_ref = {}
    for s in STRATEGIES:
        fixed_ref[s] = round(sum(next(x for x in items if x["strategy"] == s)["correct"]
                                 for items in by.values()) / len(by), 4)

    oof = {}  # (model, weight, lam) -> {idx: choice}
    fold_detail = []
    for fi, fold in enumerate(folds):
        tr, te = fold["train"], fold["test"]
        costs = fold_costs(by, tr)
        for model_name in ("numeric", "choice"):
            for weight in ("none", "balanced"):
                cw = None if weight == "none" else "balanced"
                if model_name == "numeric":
                    scaler, models = fit_numeric_models(by, tr, cw)
                    proba_fn = lambda row, _s=scaler, _m=models: predict_proba_numeric(_s, _m, row)
                else:
                    scaler, enc, models = fit_choice_models(by, tr, cw)
                    proba_fn = lambda row, _s=scaler, _e=enc, _m=models: predict_proba_choice(_s, _e, _m, row)
                for lam in LAMBDAS:
                    key = (model_name, weight, lam)
                    for i in te:
                        probs = proba_fn(single_row(by[i]))
                        oof.setdefault(key, {})[i] = choose_action(probs, costs, lam)
        fold_detail.append({"fold": fi, "n_train": len(tr), "n_test": len(te),
                            "train_costs": costs})

    # Aggregate OOF metrics.
    results = {}
    oof_golds = [single_row(by[i]).get("gold_normalized") or single_row(by[i]).get("gold", "") for i in by]
    for (model_name, weight, lam), choices in sorted(oof.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        correct, lats, calls, mix = 0, [], [], {s: 0 for s in STRATEGIES}
        per_idx_correct = {}
        for i in by:
            c = choices[i]
            row = next(x for x in by[i] if x["strategy"] == c)
            per_idx_correct[i] = int(row["correct"])
            correct += per_idx_correct[i]
            lats.append(row["cumulative_latency_ms"])
            calls.append(row["cumulative_calls"])
            mix[c] += 1
        n = len(by)
        results[f"{model_name}_{weight}_lam_{lam:g}"] = {
            "model": model_name, "class_weight": weight, "lambda_per_ms": lam,
            "oof_accuracy": round(correct / n, 4),
            "oof_mean_cumulative_ms": round(float(sum(lats)) / n, 1),
            "oof_p95_cumulative_ms": float(sorted(lats)[min(n - 1, int(n * 0.95))]),
            "oof_mean_cumulative_calls": round(float(sum(calls)) / n, 3),
            "action_mix": mix,
            "_per_idx_correct": per_idx_correct,
        }

    # Paired comparison: choice vs numeric, per (weight, lam), with stratified bootstrap CI.
    comparisons = {}
    for weight in ("none", "balanced"):
        for lam in LAMBDAS:
            a = results[f"numeric_{weight}_lam_{lam:g}"]["_per_idx_correct"]
            b = results[f"choice_{weight}_lam_{lam:g}"]["_per_idx_correct"]
            idx = sorted(by)
            diff = [b[i] - a[i] for i in idx]
            acc_a = sum(a[i] for i in idx) / len(idx)
            acc_b = sum(b[i] for i in idx) / len(idx)
            comparisons[f"choice_vs_numeric_{weight}_lam_{lam:g}"] = {
                "weight": weight, "lambda_per_ms": lam,
                "oof_acc_numeric": round(acc_a, 4), "oof_acc_choice": round(acc_b, 4),
                "oof_acc_diff": round(acc_b - acc_a, 4),
                "bootstrap": stratified_bootstrap_ci([single_row(by[i]).get("gold_normalized") or single_row(by[i]).get("gold", "") for i in idx], diff),
            }

    for v in results.values():
        v.pop("_per_idx_correct", None)

    out = {
        "train_cache": args.train_cache,
        "n_examples": len(by),
        "n_classes": len({single_row(items).get("gold_normalized") for items in by.values()}),
        "lambdas": LAMBDAS,
        "fixed_reference_train_accuracy": fixed_ref,
        "folds": fold_detail,
        "results": results,
        "paired_comparisons": comparisons,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k not in ("folds",)}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
