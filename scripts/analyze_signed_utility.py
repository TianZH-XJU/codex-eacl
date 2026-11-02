#!/usr/bin/env python3
"""Signed incremental-utility analysis for the E2 revision (offline).

The original E2 predictor was trained on a binary repair label.  The ACL
review correctly notes that repair prediction is not the same quantity as the
signed incremental utility of switching:

    Delta_i = 1[Choice_i correct] - 1[Noul_i correct]  in {-1, 0, +1}.

This script re-trains the same deployment-visible feature set on Delta:
  * three-class logistic regression and shallow tree, with expected Delta
    computed as P(+1) - P(-1);
  * ridge regression on Delta;
  * group-aware 3-fold OOF on search_train;
  * threshold selected on selection_dev by net utility and frozen;
  * one-shot evaluation on confirm with group-bootstrap CIs.

Zero new model calls: all inputs are the committed per-row caches.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
LAM = 0.05
RNG = np.random.RandomState(17)

TASKS = {
    "code_laya": (
        "results/v4/evolution/cache_train_code_laya.json",
        "results/v4/evolution/cache_code_laya.json",
        "results/v4/evolution/cache_confirm_code_laya.json",
    ),
    "code_vela": (
        "results/v4/evolution/cache_train_code_vela.json",
        "results/v4/evolution/cache_code_vela.json",
        "results/v4/evolution/cache_confirm_code_vela.json",
    ),
    "findver_laya": (
        "results/v4/evolution/cache_train_findver_laya.json",
        "results/v4/evolution/cache_findver_laya.json",
        "results/v4/evolution/cache_confirm_findver_laya.json",
    ),
    "findver_vela": (
        "results/v4/evolution/cache_train_findver_vela.json",
        "results/v4/evolution/cache_findver_vela.json",
        "results/v4/evolution/cache_confirm_findver_vela.json",
    ),
}


def load(path):
    d = json.loads((ROOT / path).read_text(encoding="utf-8"))
    return d["rows"], d.get("split", "unknown")


def delta(rows):
    out = []
    for r in rows:
        noul_ok = (float(r["noul_p"]) >= 0.5) == bool(r["gold"])
        choice_ok = bool(r.get("switch_correct", False))
        out.append(int(choice_ok) - int(noul_ok))
    return np.array(out, dtype=int)


def features(rows):
    X = []
    for r in rows:
        p = float(r["noul_p"])
        X.append([p, abs(p - 0.5), -(p * np.log(p + 1e-12) + (1 - p) * np.log((1 - p) + 1e-12))])
    return np.array(X, dtype=float)


def expected_from_proba(model, X):
    proba = model.predict_proba(X)
    out = np.zeros(len(X), dtype=float)
    for j, cls in enumerate(model.classes_):
        out += float(cls) * proba[:, j]
    return out


def oof_scores(X, y, groups, kind):
    gkf = GroupKFold(n_splits=3)
    scores = np.zeros(len(y), dtype=float)
    for tr, va in gkf.split(X, y, groups):
        if kind == "logreg3":
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
            if len(set(y[tr])) < 2:
                scores[va] = 0.0
                continue
            m.fit(X[tr], y[tr])
            scores[va] = expected_from_proba(m, X[va])
        elif kind == "tree3":
            m = DecisionTreeClassifier(max_depth=3, random_state=17)
            if len(set(y[tr])) < 2:
                scores[va] = 0.0
                continue
            m.fit(X[tr], y[tr])
            scores[va] = expected_from_proba(m, X[va])
        elif kind == "ridge":
            m = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            m.fit(X[tr], y[tr])
            scores[va] = m.predict(X[va])
        else:
            raise ValueError(kind)
    return scores


def safe_auroc(y, s):
    return round(float(roc_auc_score(y, s)), 4) if len(set(y)) > 1 else 0.5


def net_utility(rows, mask):
    d = delta(rows)
    n = len(rows)
    rep = int(np.sum((d > 0) & mask))
    reg = int(np.sum((d < 0) & mask))
    switched = int(np.sum(mask))
    raw = float(np.sum(d * mask)) / n
    cost = raw - LAM * switched / n
    return {
        "raw": round(raw, 4),
        "raw_pp": round(100.0 * raw, 2),
        "cost": round(cost, 4),
        "cost_pp": round(100.0 * cost, 2),
        "repair": rep,
        "regression": reg,
        "n_switched": switched,
    }


def select_threshold(rows, scores):
    """Maximise dev cost-adjusted utility; ties prefer fewer switches."""
    best_t, best = float("inf"), net_utility(rows, np.zeros(len(rows), dtype=bool))
    for t in sorted(set(np.round(scores, 6).tolist())):
        mask = scores >= t
        cur = net_utility(rows, mask)
        if cur["cost"] > best["cost"] or (
            cur["cost"] == best["cost"] and cur["n_switched"] < best["n_switched"]
        ):
            best_t, best = float(t), cur
    return best_t, best


def bootstrap_net_cis(rows, mask, n_boot=5000):
    by_group = {}
    for r, m, d in zip(rows, mask, delta(rows)):
        by_group.setdefault(r["group_id"], []).append((int(m), int(d)))
    groups = sorted(by_group)
    if not groups:
        return {"raw_ci95_pp": [0.0, 0.0], "cost_ci95_pp": [0.0, 0.0]}
    raw_b, cost_b = [], []
    for _ in range(n_boot):
        sample = RNG.choice(groups, size=len(groups), replace=True)
        vals = [v for g in sample for v in by_group[g]]
        n = len(vals)
        raw = sum(d for m, d in vals if m) / n
        raw_b.append(raw)
        cost_b.append(raw - LAM * sum(m for m, d in vals) / n)
    rlo, rhi = np.percentile(raw_b, [2.5, 97.5])
    clo, chi = np.percentile(cost_b, [2.5, 97.5])
    return {"raw_ci95_pp": [round(100.0 * float(rlo), 2), round(100.0 * float(rhi), 2)],
            "cost_ci95_pp": [round(100.0 * float(clo), 2), round(100.0 * float(chi), 2)]}


def bootstrap_spearman(y, scores, groups, n_boot=2000):
    by_group = {}
    for i, g in enumerate(groups):
        by_group.setdefault(g, []).append(i)
    keys = sorted(by_group)
    vals = []
    for _ in range(n_boot):
        sample = RNG.choice(keys, size=len(keys), replace=True)
        idx = [i for g in sample for i in by_group[g]]
        if len(set(y[idx].tolist())) < 2:
            continue
        rho = spearmanr(scores[idx], y[idx]).statistic
        if np.isfinite(rho):
            vals.append(float(rho))
    if not vals:
        return [0.0, 0.0]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--boots", type=int, default=5000)
    args = ap.parse_args()
    try:
        sha = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except Exception:
        sha = None
    cells = {}
    for key, (tr_p, dev_p, conf_p) in TASKS.items():
        train, tr_split = load(tr_p)
        dev, dev_split = load(dev_p)
        conf, conf_split = load(conf_p)
        Xtr, Xdev, Xconf = features(train), features(dev), features(conf)
        ytr, ydev, yconf = delta(train), delta(dev), delta(conf)
        groups = np.array([r["group_id"] for r in train])
        models = {}
        for kind in ("logreg3", "tree3", "ridge"):
            sc = oof_scores(Xtr, ytr, groups, kind)
            rho = spearmanr(sc, ytr).statistic
            models[kind] = {
                "oof_spearman": round(float(rho), 4) if np.isfinite(rho) else 0.0,
                "oof_spearman_ci95": bootstrap_spearman(ytr, sc, groups, n_boot=2000),
                "oof_auroc_repair": safe_auroc((ytr == 1).astype(int), sc),
                "oof_auroc_regression": safe_auroc((ytr == -1).astype(int), -sc),
                "oof_auroc_positive": safe_auroc((ytr > 0).astype(int), sc),
                "oof_scores": [round(float(v), 4) for v in sc],
            }
        # Fit on full train, select threshold on dev, freeze for confirm.
        candidates = {}
        fitted = {}
        for kind in ("logreg3", "tree3", "ridge"):
            if kind == "logreg3":
                m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
                m.fit(Xtr, ytr)
                fn = lambda X, _m=m: expected_from_proba(_m, X)
            elif kind == "tree3":
                m = DecisionTreeClassifier(max_depth=3, random_state=17)
                m.fit(Xtr, ytr)
                fn = lambda X, _m=m: expected_from_proba(_m, X)
            else:
                m = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
                m.fit(Xtr, ytr)
                fn = lambda X, _m=m: _m.predict(X)
            fitted[kind] = fn
            dev_sc = fn(Xdev)
            t, dev_net = select_threshold(dev, dev_sc)
            conf_sc = fn(Xconf)
            mask = conf_sc >= t
            conf_net = net_utility(conf, mask)
            conf_net.update(bootstrap_net_cis(conf, mask, n_boot=args.boots))
            candidates[kind] = {
                "dev_t": round(float(t), 4),
                "dev_net": dev_net,
                "confirm_net": conf_net,
            }
        champ = max(candidates, key=lambda k: candidates[k]["dev_net"]["cost"])
        # Baselines on confirm.
        never = net_utility(conf, np.zeros(len(conf), dtype=bool))
        always = net_utility(conf, np.ones(len(conf), dtype=bool))
        cells[key] = {
            "n_train": len(train),
            "n_dev": len(dev),
            "n_confirm": len(conf),
            "n_groups": {"train": len(set(groups.tolist())), "dev": len({r["group_id"] for r in dev}), "confirm": len({r["group_id"] for r in conf})},
            "splits": {"train": tr_split, "dev": dev_split, "confirm": conf_split},
            "delta_counts": {
                "train": {"repair": int(np.sum(ytr == 1)), "unchanged": int(np.sum(ytr == 0)), "regression": int(np.sum(ytr == -1))},
                "dev": {"repair": int(np.sum(ydev == 1)), "unchanged": int(np.sum(ydev == 0)), "regression": int(np.sum(ydev == -1))},
                "confirm": {"repair": int(np.sum(yconf == 1)), "unchanged": int(np.sum(yconf == 0)), "regression": int(np.sum(yconf == -1))},
            },
            "models": models,
            "candidates": candidates,
            "champion": champ,
            "baselines_confirm": {"never_switch": never, "always_switch": always},
        }
    report = {
        "experiment_id": "EXP-02B",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repo_commit": sha,
        "protocol_version": "v4.1",
        "analysis_script": "analyze_signed_utility.py",
        "analysis_type": "exploratory",
        "target": "signed incremental utility Delta in {-1,0,+1}",
        "features": ["p", "abs(p-0.5)", "binary entropy"],
        "pools": "train OOF -> dev threshold -> confirm frozen",
        "lam_cost": LAM,
        "cells": cells,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for k, c in cells.items():
        m = c["models"]
        print(k, "OOF rho", {x: m[x]["oof_spearman"] for x in m},
              "| champ", c["champion"],
              "dev raw/cost", c["candidates"][c["champion"]]["dev_net"]["raw_pp"],
              c["candidates"][c["champion"]]["dev_net"]["cost_pp"],
              "confirm raw/cost", c["candidates"][c["champion"]]["confirm_net"]["raw_pp"],
              c["candidates"][c["champion"]]["confirm_net"]["cost_pp"],
              c["candidates"][c["champion"]]["confirm_net"]["cost_ci95_pp"])


if __name__ == "__main__":
    raise SystemExit(main())
