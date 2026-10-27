#!/usr/bin/env python3
"""Offline NEAT/gate diagnostics from frozen caches (no Laya, no test).

1. Fold stability for the primary config (choice, lambda=2e-05) and all seeds:
   per-fold train fitness, genome size, action mix, OOF accuracy.
2. Held-out predictability of per-action correctness (numeric vs +choice),
   including how often an upgrade-fixable sample is routed to a correct upgrade.

Reads the 3-shot train cache + committed NEAT OOF JSON + local progress log.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_controller_baselines import features, single_row  # noqa: E402
from run_cv_stability import build_folds  # noqa: E402
from run_three_action import STRATEGIES, group, load_cache  # noqa: E402

LAM = 2e-05


def fold_stability(progress_records):
    """Per-fold rows for the primary config and all seeds, both modes."""
    rows = []
    for r in progress_records:
        if abs(float(r.get("lambda_per_ms", -1)) - LAM) > 1e-12:
            continue
        if r.get("topology_type") == "fixed_full_direct":
            kind = "fixed_ea_choice"
        else:
            kind = f"neat_{r.get('mode')}"
        g = r.get("best_genome", {})
        m = r.get("fold_oof_metrics", {})
        rows.append({
            "fold": r.get("fold"), "kind": kind, "seed": r.get("seed"),
            "train_fitness": round(float(g.get("fitness", float("nan"))), 4),
            "nodes": g.get("nodes"), "connections": g.get("connections_enabled"),
            "oof_accuracy": m.get("accuracy"),
            "oof_mean_ms": m.get("mean_cumulative_ms"),
            "oof_action_mix": m.get("action_mix"),
        })
    return sorted(rows, key=lambda r: (r["kind"], r["seed"], r["fold"]))


def correctness_predictability(by, folds):
    """AUC of P(action correct | initial features) on fold-held-out samples."""
    auc = {m: {s: [] for s in STRATEGIES} for m in ("numeric", "choice")}
    upgrade_id = {m: [] for m in ("numeric", "choice")}  # routed-to-correct-upgrade rate
    acc_first = {m: [] for m in ("numeric", "choice")}
    for f in folds:
        tr, te = f["train"], f["test"]
        Xtr = np.array([features(single_row(by[i])) for i in tr], dtype=float)
        sc = StandardScaler().fit(Xtr)
        cats = np.array([[single_row(by[i]).get("initial_choice") or ""] for i in tr])
        enc = OneHotEncoder(handle_unknown="ignore").fit(cats)
        Xtr2 = np.hstack([sc.transform(Xtr), enc.transform(cats).toarray()])
        for m, Xt in (("numeric", sc.transform(
                    np.array([features(single_row(by[i])) for i in te], dtype=float))),
                      ("choice", None)):
            if m == "choice":
                Xte_num = sc.transform(np.array([features(single_row(by[i])) for i in te], dtype=float))
                Cte = enc.transform(np.array([[single_row(by[i]).get("initial_choice") or ""] for i in te])).toarray()
                Xte = np.hstack([Xte_num, Cte])
                Xf = Xtr2
            else:
                Xte = Xt
                Xf = sc.transform(Xtr)
            proba = {}
            for s in STRATEGIES:
                ytr = np.array([int(next(x for x in by[i] if x["strategy"] == s)["correct"]) for i in tr])
                clf = LogisticRegression(max_iter=1000, random_state=17, class_weight=None, C=1.0)
                clf.fit(Xf, ytr)
                proba[s] = clf.predict_proba(Xte)[:, list(clf.classes_).index(1)]
            yte = {s: np.array([int(next(x for x in by[i] if x["strategy"] == s)["correct"]) for i in te]) for s in STRATEGIES}
            for s in STRATEGIES:
                try:
                    auc[m][s].append(round(float(roc_auc_score(yte[s], proba[s])), 4))
                except ValueError:
                    auc[m][s].append(None)
            # accuracy-first routing on held-out
            hit, tot, up_hit, up_tot = 0, 0, 0, 0
            for k, i in enumerate(te):
                best = max(STRATEGIES, key=lambda s: proba[s][k])
                row = next(x for x in by[i] if x["strategy"] == best)
                hit += int(row["correct"])
                tot += 1
                single_ok = next(x for x in by[i] if x["strategy"] == "single")["correct"]
                any_up = any(next(x for x in by[i] if x["strategy"] == s)["correct"] for s in ("wide", "tournament"))
                if (not single_ok) and any_up:
                    up_tot += 1
                    up_hit += int(row["correct"])
            acc_first[m].append(round(hit / tot, 4) if tot else None)
            upgrade_id[m].append(round(up_hit / up_tot, 4) if up_tot else None)
    return {"auc_per_action": auc, "accuracy_first_routing": acc_first, "upgrade_fix_rate": upgrade_id}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-cache", default="results/pilot/banking77_train3shot_cache.jsonl")
    ap.add_argument("--neat-json", default="results/pilot/banking77_neat_cv.json")
    ap.add_argument("--progress", default="results/pilot/banking77_neat_cv.progress.jsonl")
    ap.add_argument("--out", default="results/analysis/banking77_neat_diagnostics.json")
    args = ap.parse_args()

    by = group(load_cache(args.train_cache))
    folds = build_folds(by)
    progress = [json.loads(x) for x in Path(args.progress).read_text(encoding="utf-8").splitlines() if x.strip()]
    stab = fold_stability(progress)
    pred = correctness_predictability(by, folds)

    out = {
        "locked_config": {"features": "choice primary (numeric control)", "lambda_per_ms": LAM,
                          "note": "unweighted LR; fold-train-only preprocessing"},
        "fold_stability_primary": stab,
        "correctness_predictability": pred,
        "interpretation_notes": [
            "fold_stability_primary: per-fold train fitness / genome size / action mix / OOF accuracy",
            "auc_per_action: can initial features predict each action's held-out correctness?",
            "upgrade_fix_rate: among held-out samples where single is wrong but an upgrade is right, "
            "share routed (accuracy-first) to a correct action",
        ],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    for r in stab:
        print(r)
    print(json.dumps(pred, indent=1))


if __name__ == "__main__":
    raise SystemExit(main())
