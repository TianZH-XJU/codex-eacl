#!/usr/bin/env python3
"""Locked single-shot confirmation evaluation (no tuning on this set).

Fit on the 3-shot train cache only; evaluate ONCE on the frozen confirmation
cache. Locked config: unweighted LR, numeric C=1.0 vs numeric+choice C=0.2,
lambda=2e-05. Reports accuracy, mean/p95 cumulative cost, action mix, paired
difference with bootstrap CI, and the McNemar discordant table.

Limitation: 1 sample per class precludes stratified resampling, so the CI is
a plain paired bootstrap over the 77 samples; interpret with care.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_controller_baselines import features, single_row  # noqa: E402
from run_three_action import STRATEGIES, choose_action, estimate_train_costs, group, load_cache  # noqa: E402

LOCKED = {"class_weight": "none", "C_numeric": 1.0, "C_choice": 0.2, "lambda_per_ms": 2e-05,
          "upgrade_reference": "tournament"}
BOOTSTRAP_REPS = 5000
BOOTSTRAP_SEED = 17


def git_info():
    def run(*a):
        try:
            return subprocess.run(["git", "-C", str(ROOT), *a],
                                  check=True, capture_output=True, text=True).stdout.strip()
        except Exception:
            return None
    sha = run("rev-parse", "HEAD")
    try:
        dirty = bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"],
                                    check=True, capture_output=True, text=True).stdout.strip())
    except Exception:
        dirty = None
    return sha, dirty


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-cache", required=True)
    ap.add_argument("--confirmation-cache", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    train_by = group(load_cache(args.train_cache))
    conf_by = group(load_cache(args.confirmation_cache))
    conf_idx = sorted(conf_by.keys())
    lam = LOCKED["lambda_per_ms"]
    costs = estimate_train_costs(train_by)

    Xtr = np.array([features(single_row(train_by[i])) for i in train_by], dtype=float)
    Xcf = np.array([features(single_row(conf_by[i])) for i in conf_idx], dtype=float)

    # Numeric-only model.
    scaler = StandardScaler().fit(Xtr)
    num_models = {}
    for s in STRATEGIES:
        y = np.array([int(next(x for x in train_by[i] if x["strategy"] == s)["correct"]) for i in train_by])
        clf = LogisticRegression(max_iter=1000, random_state=17, class_weight=None, C=LOCKED["C_numeric"])
        clf.fit(scaler.transform(Xtr), y)
        num_models[s] = clf

    # Numeric + choice model (single controlled ablation config).
    cats_tr = np.array([[single_row(train_by[i]).get("initial_choice") or ""] for i in train_by])
    cats_cf = np.array([[single_row(conf_by[i]).get("initial_choice") or ""] for i in conf_idx])
    enc = OneHotEncoder(handle_unknown="ignore").fit(cats_tr)
    Xtr2 = np.hstack([scaler.transform(Xtr), enc.transform(cats_tr).toarray()])
    Xcf2 = np.hstack([scaler.transform(Xcf), enc.transform(cats_cf).toarray()])
    ch_models = {}
    for s in STRATEGIES:
        y = np.array([int(next(x for x in train_by[i] if x["strategy"] == s)["correct"]) for i in train_by])
        clf = LogisticRegression(max_iter=1000, random_state=17, class_weight=None, C=LOCKED["C_choice"])
        clf.fit(Xtr2, y)
        ch_models[s] = clf

    def predict_num(row):
        X = scaler.transform(np.array([features(row)], dtype=float))
        return {s: float(num_models[s].predict_proba(X)[0][list(num_models[s].classes_).index(1)]) for s in STRATEGIES}

    def predict_choice(row):
        Xn = scaler.transform(np.array([features(row)], dtype=float))
        C = enc.transform(np.array([[row.get("initial_choice") or ""]])).toarray()
        X = np.hstack([Xn, C])
        return {s: float(ch_models[s].predict_proba(X)[0][list(ch_models[s].classes_).index(1)]) for s in STRATEGIES}

    per_method = {}
    for name, fn in (("numeric_only", predict_num), ("numeric_choice", predict_choice)):
        choices = [choose_action(fn(single_row(conf_by[i])), costs, lam) for i in conf_idx]
        correct, lats, calls, mix = 0, [], [], {s: 0 for s in STRATEGIES}
        per_idx = {}
        for i, c in zip(conf_idx, choices):
            row = next(x for x in conf_by[i] if x["strategy"] == c)
            per_idx[i] = int(row["correct"])
            correct += per_idx[i]
            lats.append(row["cumulative_latency_ms"])
            calls.append(row["cumulative_calls"])
            mix[c] += 1
        n = len(conf_idx)
        per_method[name] = {
            "accuracy": round(correct / n, 4),
            "mean_cumulative_ms": round(float(sum(lats)) / n, 1),
            "p95_cumulative_ms": float(sorted(lats)[min(n - 1, int(n * 0.95))]),
            "mean_cumulative_calls": round(float(sum(calls)) / n, 3),
            "action_mix": mix,
            "_per_idx": per_idx,
        }

    a = per_method["numeric_only"]["_per_idx"]
    b = per_method["numeric_choice"]["_per_idx"]
    diff = [b[i] - a[i] for i in conf_idx]
    acc_diff = sum(diff) / len(diff)
    rng = np.random.RandomState(BOOTSTRAP_SEED)
    means = [float(np.mean(rng.choice(diff, size=len(diff), replace=True))) for _ in range(BOOTSTRAP_REPS)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    both_r = sum(1 for i in conf_idx if a[i] and b[i])
    both_w = sum(1 for i in conf_idx if not a[i] and not b[i])
    choice_only = sum(1 for i in conf_idx if b[i] and not a[i])
    numeric_only = sum(1 for i in conf_idx if a[i] and not b[i])

    # Reference fixed strategies on the confirmation set.
    fixed = {}
    for s in STRATEGIES:
        c = sum(next(x for x in conf_by[i] if x["strategy"] == s)["correct"] for i in conf_idx)
        lats = [next(x for x in conf_by[i] if x["strategy"] == s)["cumulative_latency_ms"] for i in conf_idx]
        fixed[s] = {"accuracy": round(c / len(conf_idx), 4),
                    "mean_cumulative_ms": round(float(sum(lats)) / len(lats), 1)}

    sha, dirty = git_info()
    out = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "locked_config": LOCKED,
        "train_cache": args.train_cache,
        "confirmation_cache": args.confirmation_cache,
        "n_confirmation": len(conf_idx),
        "project_git_sha": sha,
        "working_tree_dirty": dirty,
        "results": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in per_method.items()},
        "paired_choice_minus_numeric": {
            "accuracy_diff": round(float(acc_diff), 4),
            "bootstrap95_lo": round(float(lo), 4),
            "bootstrap95_hi": round(float(hi), 4),
            "bootstrap_note": "plain paired bootstrap over 77 samples (5000 reps, seed 17); "
                              "1 sample/class precludes stratified resampling — wide uncertainty expected",
            "mcnemar_table": {"both_correct": both_r, "both_wrong": both_w,
                              "choice_only_correct": choice_only, "numeric_only_correct": numeric_only},
        },
        "fixed_reference_on_confirmation": fixed,
        "evaluated_once": True,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
