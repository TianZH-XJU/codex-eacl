#!/usr/bin/env python3
"""Gate 1 strategy-complementarity analysis (offline, reads pilot caches only).

Reports per strategy accuracy, pairwise error overlap, only-A/only-B fixes,
all-correct/all-wrong rates, delta-correctness/delta-cost distributions, and
held-out predictability of upgrade gain (DeltaCorrect in {-1,0,+1}) with a
small regularized LR under problem-grouped 3-fold CV. No Laya calls.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

STRATEGIES = ["single", "wide", "tournament"]
FEAT_KEYS = ["initial_answer_confidence", "initial_entropy_confidence", "initial_margin",
             "initial_entropy", "initial_top1_prob", "initial_text_len", "initial_n_options"]


def load_cache(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def group(rows):
    by = {}
    for r in rows:
        by.setdefault((r["split"], r["position"]), []).append(r)
    return by


def correctness(by, idx, strategy):
    return next(x for x in by[idx] if x["strategy"] == strategy)["correct"]


def single_row(by, idx):
    return next(x for x in by[idx] if x["strategy"] == "single")


def features(row):
    return [float(row.get(k) or 0.0) for k in FEAT_KEYS]


def delta_correct(by, idx, action, baseline="single"):
    return int(correctness(by, idx, action)) - int(correctness(by, idx, baseline))


def predictability(by, seed=17):
    """Problem-grouped 3-fold CV: can initial SINGLE features predict
    upgrade gain (DeltaCorrect in {-1,0,+1}) for wide/tournament?"""
    idxs = sorted(by.keys())
    rng = random.Random(seed)
    rng.shuffle(idxs)
    folds = [idxs[i::3] for i in range(3)]
    out = {}
    for action in ("wide", "tournament"):
        accs, majs = [], []
        for k in range(3):
            te, tr = set(folds[k]), [i for j, f in enumerate(folds) if j != k for i in f]
            Xtr = np.array([features(single_row(by, i)) for i in tr], dtype=float)
            ytr = np.array([delta_correct(by, i, action) + 1 for i in tr])  # shift to 0,1,2
            sc = StandardScaler().fit(Xtr)
            clf = LogisticRegression(max_iter=1000, random_state=17)
            clf.fit(sc.transform(Xtr), ytr)
            Xte = sc.transform(np.array([features(single_row(by, i)) for i in te], dtype=float))
            yte = np.array([delta_correct(by, i, action) + 1 for i in te])
            pred = clf.predict(Xte)
            accs.append(float((pred == yte).mean()))
            maj = max(np.bincount(yte, minlength=3) / len(yte))
            majs.append(float(maj))
        out[action] = {"oof_accuracy": [round(a, 4) for a in accs],
                       "majority_baseline": [round(m, 4) for m in majs],
                       "mean_oof": round(float(sum(accs) / len(accs)), 4),
                       "mean_majority": round(float(sum(majs) / len(majs)), 4)}
    return out


def split_report(by, strat_costs=None):
    idxs = list(by.keys())
    correct = {s: sum(correctness(by, i, s) for i in idxs) for s in STRATEGIES}
    acc = {s: round(correct[s] / len(idxs), 4) for s in STRATEGIES}
    overlap = {}
    for x in STRATEGIES:
        for y in STRATEGIES:
            if x < y:
                overlap[f"{x}_vs_{y}"] = {
                    "both_wrong": sum((not correctness(by, i, x)) and (not correctness(by, i, y)) for i in idxs),
                    "only_first_wrong": sum((not correctness(by, i, x)) and correctness(by, i, y) for i in idxs),
                    "only_second_wrong": sum(correctness(by, i, x) and (not correctness(by, i, y)) for i in idxs),
                    "both_correct": sum(correctness(by, i, x) and correctness(by, i, y) for i in idxs),
                }
    only_fix, new_err = {}, {}
    for a in ("wide", "tournament"):
        only_fix[a] = sum((not correctness(by, i, "single")) and correctness(by, i, a)
                          and not any(correctness(by, i, b) for b in ("wide", "tournament") if b != a) for i in idxs)
        new_err[a] = sum(correctness(by, i, "single") and (not correctness(by, i, a)) for i in idxs)
    all_correct = sum(all(correctness(by, i, s) for s in STRATEGIES) for i in idxs)
    all_wrong = sum(not any(correctness(by, i, s) for s in STRATEGIES) for i in idxs)
    oracle = sum(any(correctness(by, i, s) for s in STRATEGIES) for i in idxs)
    lat = {}
    for s in STRATEGIES:
        vals = sorted(next(x for x in by[i] if x["strategy"] == s)["cumulative_latency_ms"] for i in idxs)
        lat[s] = {"mean_ms": round(sum(vals) / len(vals), 1), "p50_ms": vals[len(vals) // 2],
                  "p95_ms": vals[min(len(vals) - 1, int(len(vals) * 0.95))], "max_ms": vals[-1]}
    # prediction distribution of SINGLE (position-bias probe)
    from collections import Counter
    pred_dist = Counter(next(x for x in by[i] if x["strategy"] == "single")["choice"] for i in idxs)
    return {"n": len(idxs), "accuracy": acc, "correct_counts": correct,
            "pairwise_overlap": overlap, "only_fix": only_fix, "new_errors": new_err,
            "all_correct": all_correct, "all_wrong": all_wrong,
            "oracle_accuracy": round(oracle / len(idxs), 4),
            "oracle_gain_over_best_fixed": round(oracle / len(idxs) - max(acc.values()), 4),
            "cumulative_latency_ms": lat,
            "single_prediction_top": pred_dist.most_common(8)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-cache", default="results/processbench/pilot_train_cache.jsonl")
    ap.add_argument("--dev-cache", default="results/processbench/pilot_dev_cache.jsonl")
    ap.add_argument("--out", default="results/analysis/processbench_gate1.json")
    args = ap.parse_args()

    by = group(load_cache(args.train_cache) + load_cache(args.dev_cache))
    train_by = {k: v for k, v in by.items() if k[0] == "pilot_train"}
    dev_by = {k: v for k, v in by.items() if k[0] == "pilot_dev"}
    out = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gate": "Gate 1 strategy complementarity (pilot)",
        "train": split_report(train_by),
        "dev": split_report(dev_by),
        "predictability_train_3fold": predictability(train_by),
        "note": "predictability uses initial SINGLE features only; DeltaCorrect in {-1,0,+1}",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: {kk: out[k][kk] for kk in ("n", "accuracy", "oracle_accuracy",
                   "oracle_gain_over_best_fixed", "only_fix", "new_errors",
                   "all_correct", "all_wrong")} for k in ("train", "dev")}, indent=2))
    print(json.dumps(out["predictability_train_3fold"], indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
