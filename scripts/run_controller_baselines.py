#!/usr/bin/env python3
"""Run simple controller baselines on the SINGLE-first action cache."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]


def load_cache(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def features(r):
    text = r["text"]
    return [
        r.get("initial_answer_confidence") or 0.0,
        r.get("initial_entropy_confidence") or 0.0,
        r.get("initial_margin") or 0.0,
        r.get("initial_entropy") or 0.0,
        r.get("initial_top1_prob") or 0.0,
        r.get("initial_top2_prob") or 0.0,
        r.get("initial_act_probability") if r.get("initial_act_probability") is not None else 0.0,
        len(text),
        len(r.get("initial_probabilities") or {}),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-cache", required=True)
    ap.add_argument("--dev-cache", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    train = load_cache(args.train_cache)
    dev = load_cache(args.dev_cache)
    strategies = sorted({r["strategy"] for r in train})

    # Group rows by example index.
    def group(rows):
        by = {}
        for r in rows:
            by.setdefault(r["index"], []).append(r)
        return by

    train_by = group(train)
    dev_by = group(dev)

    # Label each example with its best strategy.
    def label_example(items):
        best = max(items, key=lambda r: (r["correct"], -(r["cumulative_calls"] or 0), -r["cumulative_latency_ms"]))
        return best["strategy"]

    X_train = np.array([features(next(r for r in train_by[i] if r["strategy"] == "single")) for i in train_by], dtype=float)
    # Use single row features for every example.
    def single_row(items):
        for r in items:
            if r["strategy"] == "single":
                return r
        return items[0]

    X_train = np.array([features(single_row(train_by[i])) for i in train_by], dtype=float)

    def upgrade_needed(items):
        sr = single_row(items)
        others_correct = any(r["correct"] for r in items if r["strategy"] != "single")
        return int((not sr["correct"]) and others_correct)

    y_train = np.array([upgrade_needed(train_by[i]) for i in train_by])
    X_dev = np.array([features(single_row(dev_by[i])) for i in dev_by], dtype=float)
    dev_indices = list(dev_by.keys())

    # Best fixed by train accuracy.
    train_acc = {s: sum(label_example(train_by[i]) == s for i in train_by) / len(train_by) for s in strategies}
    # Actually best fixed accuracy should be strategy accuracy on train.
    train_strategy_acc = {}
    for s in strategies:
        correct = sum(r["correct"] for r in train if r["strategy"] == s)
        n = len(train_by)
        train_strategy_acc[s] = correct / n if n else 0.0
    best_fixed = max(strategies, key=lambda s: train_strategy_acc[s])

    upgrade_candidates = [s for s in strategies if s != "single"]
    upgrade_target = max(upgrade_candidates, key=lambda s: train_strategy_acc[s]) if upgrade_candidates else "single"

    def evaluate(preds):
        correct = 0
        latencies = []
        calls = []
        for i, pred in zip(dev_indices, preds):
            chosen = pred
            row = next(r for r in dev_by[i] if r["strategy"] == chosen)
            correct += int(row["correct"])
            latencies.append(row["cumulative_latency_ms"])
            calls.append(row["cumulative_calls"])
        n = len(dev_indices)
        return {
            "accuracy": round(float(correct) / n, 4) if n else None,
            "mean_cumulative_ms": round(float(sum(latencies)) / n, 1) if n else None,
            "p95_cumulative_ms": float(sorted(latencies)[min(n - 1, int(n * 0.95))]) if latencies else None,
            "mean_cumulative_calls": round(float(sum(calls)) / n, 3) if n else None,
        }

    results = {}
    # Best fixed.
    preds = [best_fixed] * len(dev_indices)
    results["best_fixed"] = {"strategy": best_fixed, **evaluate(preds)}

    # Confidence threshold cascade.
    best_t = None
    best_acc = -1
    confs = sorted({r_single["initial_answer_confidence"] for r_single in [single_row(train_by[i]) for i in train_by]})
    for t in confs:
        preds_train = []
        for i in train_by:
            r = single_row(train_by[i])
            if (r.get("initial_answer_confidence") or 0.0) >= t:
                preds_train.append("single")
            else:
                preds_train.append(upgrade_target)
        acc = sum(1 for i, p in zip(train_by, preds_train) if (next(x for x in train_by[i] if x["strategy"] == p)["correct"])) / len(train_by)
        if acc > best_acc:
            best_acc = acc
            best_t = t
    preds = []
    for i in dev_by:
        r = single_row(dev_by[i])
        if (r.get("initial_answer_confidence") or 0.0) >= best_t:
            preds.append("single")
        else:
            preds.append(upgrade_target)
    results["confidence_threshold"] = {"threshold": (float(best_t) if best_t is not None else None), "upgrade_target": upgrade_target, **evaluate(preds)}

    # Margin threshold.
    best_t = None
    best_acc = -1
    margins = sorted({r_single.get("initial_margin") or 0.0 for r_single in [single_row(train_by[i]) for i in train_by]})
    for t in margins:
        preds_train = []
        for i in train_by:
            r = single_row(train_by[i])
            if (r.get("initial_margin") or 0.0) >= t:
                preds_train.append("single")
            else:
                preds_train.append(upgrade_target)
        acc = sum(1 for i, p in zip(train_by, preds_train) if (next(x for x in train_by[i] if x["strategy"] == p)["correct"])) / len(train_by)
        if acc > best_acc:
            best_acc = acc
            best_t = t
    preds = []
    for i in dev_by:
        r = single_row(dev_by[i])
        if (r.get("initial_margin") or 0.0) >= best_t:
            preds.append("single")
        else:
            preds.append(upgrade_target)
    results["margin_threshold"] = {"threshold": (float(best_t) if best_t is not None else None), "upgrade_target": upgrade_target, **evaluate(preds)}

    # Entropy threshold (lower entropy -> more confident).
    best_t = None
    best_acc = -1
    ents = sorted({r_single.get("initial_entropy") or 0.0 for r_single in [single_row(train_by[i]) for i in train_by]})
    for t in ents:
        preds_train = []
        for i in train_by:
            r = single_row(train_by[i])
            if (r.get("initial_entropy") or 0.0) <= t:
                preds_train.append("single")
            else:
                preds_train.append(upgrade_target)
        acc = sum(1 for i, p in zip(train_by, preds_train) if (next(x for x in train_by[i] if x["strategy"] == p)["correct"])) / len(train_by)
        if acc > best_acc:
            best_acc = acc
            best_t = t
    preds = []
    for i in dev_by:
        r = single_row(dev_by[i])
        if (r.get("initial_entropy") or 0.0) <= best_t:
            preds.append("single")
        else:
            preds.append(upgrade_target)
    results["entropy_threshold"] = {"threshold": (float(best_t) if best_t is not None else None), "upgrade_target": upgrade_target, **evaluate(preds)}

    # Logistic regression (multiclass best strategy).
    try:
        lr = LogisticRegression(max_iter=1000, random_state=17, class_weight="balanced")
        lr.fit(X_train, y_train)
        raw = list(lr.predict(X_dev))
        preds = ["single" if p == 0 else upgrade_target for p in raw]
        results["logistic_regression"] = {"classes": [str(x) for x in lr.classes_], "positive_rate_train": float(y_train.mean()), **evaluate(preds)}
    except Exception as e:
        results["logistic_regression"] = {"error": f"{type(e).__name__}: {e}"}

    # Shallow decision tree.
    try:
        dt = DecisionTreeClassifier(max_depth=3, random_state=17, class_weight="balanced")
        dt.fit(X_train, y_train)
        raw = list(dt.predict(X_dev))
        preds = ["single" if p == 0 else upgrade_target for p in raw]
        results["decision_tree"] = {"max_depth": 3, **evaluate(preds)}
    except Exception as e:
        results["decision_tree"] = {"error": f"{type(e).__name__}: {e}"}

    # Small MLP.
    try:
        mlp = MLPClassifier(hidden_layer_sizes=(8,), max_iter=500, random_state=17)
        mlp.fit(X_train, y_train)
        raw = list(mlp.predict(X_dev))
        preds = ["single" if p == 0 else upgrade_target for p in raw]
        results["mlp"] = {"hidden_layer_sizes": [8], **evaluate(preds)}
    except Exception as e:
        results["mlp"] = {"error": f"{type(e).__name__}: {e}"}

    # Random routing baseline.
    rng = __import__("random").Random(17)
    preds = [rng.choice(strategies) for _ in dev_indices]
    results["random_routing"] = evaluate(preds)

    # Oracle upper bound from cache.
    oracle_preds = []
    for i in dev_by:
        items = dev_by[i]
        best = max(items, key=lambda r: (r["correct"], -(r["cumulative_calls"] or 0), -r["cumulative_latency_ms"]))
        oracle_preds.append(best["strategy"])
    results["oracle"] = evaluate(oracle_preds)

    out = {
        "train_cache": args.train_cache,
        "dev_cache": args.dev_cache,
        "n_train": len(train_by),
        "n_dev": len(dev_by),
        "strategies": strategies,
        "train_strategy_accuracy": train_strategy_acc,
        "results": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
PY