#!/usr/bin/env python3
"""Run simple controller baselines on the SINGLE-first action cache.

Learning target: binary upgrade decision tied to a concrete upgrade_target
chosen on TRAIN only. 0 = stay with SINGLE, 1 = upgrade to upgrade_target.
Features use only initial SINGLE outputs plus cheap input metadata.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]

# Locked concise feature set (pre-registered; no ad-hoc search).
FEATURE_NAMES = [
    "answer_confidence",
    "margin",
    "entropy_confidence",
    "act_probability",
    "text_len",
    "choice_name_len",
]


def load_cache(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def single_row(items):
    for r in items:
        if r["strategy"] == "single":
            return r
    return items[0]


def features(r):
    """Only SINGLE-visible features + cheap input metadata.

    Excludes: fixed option count (constant 77 on BANKING77),
    duplicate top1/top2 derivations, and raw entropy (redundant
    with entropy_confidence given fixed class count).
    """
    text = r.get("text") or ""
    choice = r.get("initial_choice") or ""
    act = r.get("initial_act_probability")
    return [
        float(r.get("initial_answer_confidence") or 0.0),
        float(r.get("initial_margin") or 0.0),
        float(r.get("initial_entropy_confidence") or 0.0),
        float(act if act is not None else 0.0),
        float(len(text)),
        float(len(choice)),
    ]


def upgrade_needed(items, upgrade_target):
    """1 iff SINGLE is wrong and the concrete upgrade target is right."""
    single = next(r for r in items if r["strategy"] == "single")
    target = next(r for r in items if r["strategy"] == upgrade_target)
    return int(bool(target["correct"]) and not bool(single["correct"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-cache", required=True)
    ap.add_argument("--dev-cache", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    train = load_cache(args.train_cache)
    dev = load_cache(args.dev_cache)
    strategies = sorted({r["strategy"] for r in train})

    def group(rows):
        by = {}
        for r in rows:
            by.setdefault(r["index"], []).append(r)
        return by

    train_by = group(train)
    dev_by = group(dev)
    dev_indices = list(dev_by.keys())

    # 1. Choose best_fixed and upgrade_target on TRAIN only.
    train_strategy_acc = {}
    for s in strategies:
        correct = sum(r["correct"] for r in train if r["strategy"] == s)
        n = len(train_by)
        train_strategy_acc[s] = correct / n if n else 0.0
    best_fixed = max(strategies, key=lambda s: train_strategy_acc[s])
    upgrade_candidates = [s for s in strategies if s != "single"]
    upgrade_target = max(upgrade_candidates, key=lambda s: train_strategy_acc[s]) if upgrade_candidates else "single"

    # 2. Build binary labels tied to the concrete upgrade target.
    X_train = np.array([features(single_row(train_by[i])) for i in train_by], dtype=float)
    y_train = np.array([upgrade_needed(train_by[i], upgrade_target) for i in train_by])
    X_dev = np.array([features(single_row(dev_by[i])) for i in dev_by], dtype=float)

    def evaluate(preds):
        correct = 0
        latencies = []
        calls = []
        for i, pred in zip(dev_indices, preds):
            row = next(r for r in dev_by[i] if r["strategy"] == pred)
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
    preds = [best_fixed] * len(dev_indices)
    results["best_fixed"] = {"strategy": best_fixed, **evaluate(preds)}

    # Threshold cascades: threshold selected on TRAIN, applied to DEV.
    def fit_threshold(key, higher_is_confident):
        vals = sorted({float(single_row(train_by[i]).get(key) or 0.0) for i in train_by})
        best_t, best_acc = None, -1.0
        for t in vals:
            n_ok, n_hit = 0, 0
            for i in train_by:
                r = single_row(train_by[i])
                v = float(r.get(key) or 0.0)
                pred = "single" if (v >= t if higher_is_confident else v <= t) else upgrade_target
                row = next(x for x in train_by[i] if x["strategy"] == pred)
                n_ok += int(row["correct"])
                n_hit += 1
            acc = n_ok / n_hit if n_hit else 0.0
            if acc > best_acc:
                best_acc, best_t = acc, t
        return best_t, best_acc

    for name, key, hic in [
        ("confidence_threshold", "initial_answer_confidence", True),
        ("margin_threshold", "initial_margin", True),
        ("entropy_confidence_threshold", "initial_entropy_confidence", True),
    ]:
        best_t, train_acc = fit_threshold(key, hic)
        preds = []
        for i in dev_by:
            v = float(single_row(dev_by[i]).get(key) or 0.0)
            preds.append("single" if (v >= best_t if hic else v <= best_t) else upgrade_target)
        results[name] = {
            "threshold": float(best_t) if best_t is not None else None,
            "threshold_key": key,
            "train_accuracy": round(train_acc, 4),
            "upgrade_target": upgrade_target,
            **evaluate(preds),
        }

    # Learned binary controllers. Scaler fitted on TRAIN only.
    try:
        lr = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=17, class_weight="balanced")),
        ])
        lr.fit(X_train, y_train)
        raw = list(lr.predict(X_dev))
        preds = ["single" if p == 0 else upgrade_target for p in raw]
        results["logistic_regression"] = {
            "feature_names": FEATURE_NAMES,
            "standardized": True,
            "positive_rate_train": float(y_train.mean()),
            **evaluate(preds),
        }
    except Exception as e:  # noqa: BLE001
        results["logistic_regression"] = {"error": f"{type(e).__name__}: {e}"}

    try:
        dt = DecisionTreeClassifier(max_depth=3, random_state=17, class_weight="balanced")
        dt.fit(X_train, y_train)
        raw = list(dt.predict(X_dev))
        preds = ["single" if p == 0 else upgrade_target for p in raw]
        results["decision_tree"] = {
            "max_depth": 3,
            "feature_names": FEATURE_NAMES,
            "standardized": False,
            "positive_rate_train": float(y_train.mean()),
            **evaluate(preds),
        }
    except Exception as e:  # noqa: BLE001
        results["decision_tree"] = {"error": f"{type(e).__name__}: {e}"}

    try:
        mlp = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(8,), max_iter=500, random_state=17)),
        ])
        mlp.fit(X_train, y_train)
        raw = list(mlp.predict(X_dev))
        preds = ["single" if p == 0 else upgrade_target for p in raw]
        results["mlp"] = {
            "hidden_layer_sizes": [8],
            "feature_names": FEATURE_NAMES,
            "standardized": True,
            "positive_rate_train": float(y_train.mean()),
            **evaluate(preds),
        }
    except Exception as e:  # noqa: BLE001
        results["mlp"] = {"error": f"{type(e).__name__}: {e}"}

    rng = __import__("random").Random(17)
    preds = [rng.choice(strategies) for _ in dev_indices]
    results["random_routing"] = evaluate(preds)

    oracle_preds = []
    for i in dev_by:
        best = max(dev_by[i], key=lambda r: (r["correct"], -(r["cumulative_calls"] or 0), -r["cumulative_latency_ms"]))
        oracle_preds.append(best["strategy"])
    results["oracle"] = evaluate(oracle_preds)

    out = {
        "train_cache": args.train_cache,
        "dev_cache": args.dev_cache,
        "n_train": len(train_by),
        "n_dev": len(dev_by),
        "strategies": strategies,
        "feature_names": FEATURE_NAMES,
        "train_strategy_accuracy": train_strategy_acc,
        "best_fixed_strategy": best_fixed,
        "upgrade_target": upgrade_target,
        "results": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
