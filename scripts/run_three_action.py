#!/usr/bin/env python3
"""Three-action offline policy evaluation on the SINGLE-first action cache.

For each action (single/wide/tournament) train one lightweight correctness
model from the SAME locked 6 pre-decision features. Action choice maximizes
predicted correctness minus a preset cost weight times the TRAIN-estimated
mean cumulative cost of each action. Per-sample dev latency is NEVER used
for the decision; only train-estimated per-action mean costs.

Fully offline: reads existing JSONL caches, no Laya calls.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from run_controller_baselines import FEATURE_NAMES, features, single_row

STRATEGIES = ["single", "wide", "tournament"]

# Pre-registered cost-weight grid (per ms of train-estimated mean cumulative latency).
LAMBDAS = [0.0, 2e-05, 5e-05, 1e-04]


def load_cache(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def group(rows):
    by = {}
    for r in rows:
        by.setdefault(r["index"], []).append(r)
    return by


def estimate_train_costs(train_by):
    """Mean cumulative latency/calls per strategy, estimated on TRAIN only."""
    costs = {}
    for s in STRATEGIES:
        lats = [next(x for x in items if x["strategy"] == s)["cumulative_latency_ms"]
                for items in train_by.values()]
        calls = [next(x for x in items if x["strategy"] == s)["cumulative_calls"]
                 for items in train_by.values()]
        costs[s] = {
            "mean_cumulative_ms": float(sum(lats) / len(lats)),
            "mean_cumulative_calls": float(sum(calls) / len(calls)),
        }
    return costs


def train_correctness_models(train_by):
    """One binary LR per action: P(action correct | initial features)."""
    models = {}
    X = np.array([features(single_row(train_by[i])) for i in train_by], dtype=float)
    for s in STRATEGIES:
        y = np.array([int(next(x for x in train_by[i] if x["strategy"] == s)["correct"])
                      for i in train_by])
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=17, class_weight="balanced")),
        ])
        clf.fit(X, y)
        models[s] = clf
    return models


def choose_action(probs, costs, lam):
    """Argmax over actions of p_correct - lam * train-mean-cost. Pure function of
    predicted probabilities and train costs (no per-sample dev latency)."""
    best, best_score = None, None
    for s in STRATEGIES:
        score = probs[s] - lam * costs[s]["mean_cumulative_ms"]
        if best_score is None or score > best_score:
            best, best_score = s, score
        elif score == best_score and costs[s]["mean_cumulative_ms"] < costs[best]["mean_cumulative_ms"]:
            best = s
    return best


def evaluate_choices(dev_by, choices):
    correct, lats, calls = 0, [], []
    for i, pred in zip(dev_by.keys(), choices):
        row = next(r for r in dev_by[i] if r["strategy"] == pred)
        correct += int(row["correct"])
        lats.append(row["cumulative_latency_ms"])
        calls.append(row["cumulative_calls"])
    n = len(dev_by)
    return {
        "accuracy": round(correct / n, 4) if n else None,
        "mean_cumulative_ms": round(float(sum(lats)) / n, 1) if lats else None,
        "p95_cumulative_ms": float(sorted(lats)[min(n - 1, int(n * 0.95))]) if lats else None,
        "mean_cumulative_calls": round(float(sum(calls)) / n, 3) if calls else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-cache", required=True)
    ap.add_argument("--dev-cache", required=True)
    ap.add_argument("--binary-baseline-json", default=None,
                    help="run_controller_baselines.py output to merge into Pareto table")
    ap.add_argument("--with-choice-ablation", action="store_true",
                    help="one controlled ablation adding one-hot initial_choice (strong L2, C=0.2)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    train = load_cache(args.train_cache)
    dev = load_cache(args.dev_cache)
    train_by, dev_by = group(train), group(dev)
    dev_indices = list(dev_by.keys())

    costs = estimate_train_costs(train_by)
    models = train_correctness_models(train_by)

    # Predicted correctness per dev example per action.
    X_dev = np.array([features(single_row(dev_by[i])) for i in dev_indices], dtype=float)
    proba = {}
    for s in STRATEGIES:
        p1 = models[s].predict_proba(X_dev)
        cls = list(models[s].named_steps["clf"].classes_)
        idx1 = cls.index(1) if 1 in cls else int(np.argmax(p1[0]))
        proba[s] = np.array([row[idx1] if 1 in cls else float(row[idx1]) for row in p1])

    results = {}
    for lam in LAMBDAS:
        choices = []
        for k in range(len(dev_indices)):
            probs = {s: float(proba[s][k]) for s in STRATEGIES}
            choices.append(choose_action(probs, costs, lam))
        m = evaluate_choices(dev_by, choices)
        # action mix for diagnosis
        mix = {s: int(sum(1 for c in choices if c == s)) for s in STRATEGIES}
        results[f"tri_action_lam_{lam:g}"] = {"lambda_per_ms": lam, "action_mix": mix, **m}

    out = {
        "train_cache": args.train_cache,
        "dev_cache": args.dev_cache,
        "n_train": len(train_by),
        "n_dev": len(dev_by),
        "feature_names": FEATURE_NAMES,
        "train_mean_costs": costs,
        "lambdas": LAMBDAS,
        "results": results,
    }

    # Controlled single ablation: add one-hot predicted category, strong L2.
    if args.with_choice_ablation:
        cats_train = np.array([[single_row(train_by[i]).get("initial_choice") or ""] for i in train_by])
        cats_dev = np.array([[single_row(dev_by[i]).get("initial_choice") or ""] for i in dev_indices])
        Xtr_num = np.array([features(single_row(train_by[i])) for i in train_by], dtype=float)
        Xde_num = X_dev
        try:
            enc = OneHotEncoder(handle_unknown="ignore")
            Ctr = enc.fit_transform(cats_train).toarray()
            Cde = enc.transform(cats_dev).toarray()
            import numpy as _np
            Xtr = _np.hstack([StandardScaler().fit_transform(Xtr_num), Ctr])
            Xde = _np.hstack([StandardScaler().fit(Xtr_num).transform(Xde_num), Cde])
            ab_models = {}
            for s in STRATEGIES:
                y = _np.array([int(next(x for x in train_by[i] if x["strategy"] == s)["correct"])
                               for i in train_by])
                clf = LogisticRegression(max_iter=1000, random_state=17, class_weight="balanced", C=0.2)
                clf.fit(Xtr, y)
                ab_models[s] = clf
            ab_results = {}
            for lam in LAMBDAS:
                choices = []
                for k in range(len(dev_indices)):
                    probs = {}
                    for s in STRATEGIES:
                        cls = list(ab_models[s].classes_)
                        row = ab_models[s].predict_proba(Xde[k : k + 1])[0]
                        probs[s] = float(row[cls.index(1)]) if 1 in cls else float(row[0])
                    choices.append(choose_action(probs, costs, lam))
                m = evaluate_choices(dev_by, choices)
                mix = {s: int(sum(1 for c in choices if c == s)) for s in STRATEGIES}
                ab_results[f"tri_action_choice_lam_{lam:g}"] = {
                    "lambda_per_ms": lam, "action_mix": mix, "C": 0.2, **m}
            out["choice_ablation"] = {
                "note": "one controlled ablation: 6 numeric + one-hot initial_choice, C=0.2",
                "n_categories": int(Ctr.shape[1]),
                "results": ab_results,
            }
        except Exception as e:  # noqa: BLE001
            out["choice_ablation"] = {"error": f"{type(e).__name__}: {e}"}

    # Merge binary-baseline dev results into one Pareto table if provided.
    if args.binary_baseline_json:
        base = json.loads(Path(args.binary_baseline_json).read_text(encoding="utf-8"))
        pareto = {}
        for name, r in base.get("results", {}).items():
            if isinstance(r, dict) and "accuracy" in r:
                pareto[name] = {k: r.get(k) for k in
                                ("accuracy", "mean_cumulative_ms", "p95_cumulative_ms", "mean_cumulative_calls")}
        for name, r in results.items():
            pareto[name] = {k: r.get(k) for k in
                            ("accuracy", "mean_cumulative_ms", "p95_cumulative_ms", "mean_cumulative_calls")}
        if "choice_ablation" in out and "results" in out["choice_ablation"]:
            for name, r in out["choice_ablation"]["results"].items():
                pareto[name] = {k: r.get(k) for k in
                                ("accuracy", "mean_cumulative_ms", "p95_cumulative_ms", "mean_cumulative_calls")}
        out["pareto_dev"] = pareto

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
