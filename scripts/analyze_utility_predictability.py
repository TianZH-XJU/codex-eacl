#!/usr/bin/env python3
"""S-02 / EXP-02: incremental-utility predictability (offline).

Question: can deployment-visible features predict whether the switch action
repairs a noul error (Y=1) or not (Y=0)? Labels need gold (train/dev/confirm
pools only); features use NOTHING that requires running the switch action
(confidence |p-0.5|, p, entropy from the cached noul_p).

Protocol: group-aware OOF on train (GroupKFold over group_id) -> select model
class + operating threshold on dev (max net utility) -> frozen evaluation on
confirm. Comparators: majority, single confidence threshold, LogReg, shallow
tree. Reports AUROC/AUPRC (OOF), repair precision/recall, net utility and
cost-adjusted net utility (lam=0.05 per extra call, matching fitness).
Zero new model calls.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

LAM = 0.05

TASKS = {
    "code_laya": ("results/v4/evolution/cache_train_code_laya.json",
                  "results/v4/evolution/cache_code_laya.json",
                  "results/v4/evolution/cache_confirm_code_laya.json"),
    "code_vela": ("results/v4/evolution/cache_train_code_vela.json",
                  "results/v4/evolution/cache_code_vela.json",
                  "results/v4/evolution/cache_confirm_code_vela.json"),
    "findver_laya": ("results/v4/evolution/cache_train_findver_laya.json",
                     "results/v4/evolution/cache_findver_laya.json",
                     "results/v4/evolution/cache_confirm_findver_laya.json"),
    "findver_vela": ("results/v4/evolution/cache_train_findver_vela.json",
                     "results/v4/evolution/cache_findver_vela.json",
                     "results/v4/evolution/cache_confirm_findver_vela.json"),
}


def load(path):
    d = json.loads((ROOT / path).read_text(encoding="utf-8"))
    return d["rows"], d.get("split", "unknown")


def features(rows):
    """Deployment-visible features only (no switch output, no gold)."""
    X = []
    for r in rows:
        p = float(r["noul_p"])
        c = abs(p - 0.5)
        ent = -(p * np.log(p + 1e-12) + (1 - p) * np.log((1 - p) + 1e-12))
        X.append([c, p, ent])
    return np.array(X)


def labels(rows):
    """Y=1 iff switch repairs a noul error (needs gold: labels only)."""
    y = []
    for r in rows:
        noul_ok = (float(r["noul_p"]) >= 0.5) == bool(r["gold"])
        sw_ok = bool(r.get("switch_correct", False))
        y.append(int((not noul_ok) and sw_ok))
    return np.array(y)


def net_utility(rows, switch_mask):
    """(repaired - regressed)/n vs never-switch + cost-adjusted version."""
    rep = reg = extra = 0
    for r, s in zip(rows, switch_mask):
        noul_ok = (float(r["noul_p"]) >= 0.5) == bool(r["gold"])
        if not s:
            continue
        extra += 1
        sw_ok = bool(r.get("switch_correct", False))
        if not noul_ok and sw_ok:
            rep += 1
        elif noul_ok and not sw_ok:
            reg += 1
    n = len(rows)
    net = (rep - reg) / n
    return {"net": round(net, 4), "repair": rep, "regression": reg,
            "n_switched": int(np.sum(switch_mask)),
            "cost_adjusted": round(net - LAM * (extra / n), 4)}


def oof_scores(X, y, groups, kind):
    """Group-aware out-of-fold positive-class scores on train."""
    gkf = GroupKFold(n_splits=3)
    scores = np.zeros(len(y))
    for tr, va in gkf.split(X, y, groups):
        if kind == "logreg":
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        else:
            m = DecisionTreeClassifier(max_depth=3, random_state=17)
        if len(set(y[tr])) < 2:
            scores[va] = 0.0
            continue
        m.fit(X[tr], y[tr])
        scores[va] = m.predict_proba(X[va])[:, 1]
    return scores


def pr_auc(y, scores):
    if len(set(y)) < 2:
        return 0.0
    prec, rec, _ = precision_recall_curve(y, scores)
    return round(float(auc(rec, prec)), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    try:
        sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                             check=True, capture_output=True, text=True).stdout.strip()
    except Exception:
        sha = None
    cells = {}
    for key, (tr_p, dev_p, conf_p) in TASKS.items():
        train, _ = load(tr_p)
        dev, dev_split = load(dev_p)
        conf, conf_split = load(conf_p)
        Xtr, ytr = features(train), labels(train)
        groups = np.array([r["group_id"] for r in train])
        Xdev, ydev = features(dev), labels(dev)
        Xconf = features(conf)
        ms = {}
        # OOF scores for probabilistic models
        for kind in ("logreg", "tree"):
            sc = oof_scores(Xtr, ytr, groups, kind)
            auroc = round(float(roc_auc_score(ytr, sc)), 4) if len(set(ytr)) > 1 else 0.5
            ms[kind] = {"oof_auroc": auroc, "oof_auprc": pr_auc(ytr, sc), "oof_scores": [round(float(v), 4) for v in sc]}
        # threshold rule on raw confidence (switch if conf < t)
        conf_tr = Xtr[:, 0]
        best_t, best_net = 0.1, -1e9
        for t in (0.05, 0.1, 0.15, 0.2, 0.25):
            n = net_utility(train, conf_tr < t)["net"]
            if n > best_net:
                best_t, best_net = t, n
        ms["threshold"] = {"oof_rule": "switch if conf < t", "oof_t": best_t,
                           "oof_net": round(best_net, 4)}
        ms["majority"] = {"oof_net": 0.0, "note": "never-switch baseline"}
        # fit on full train, select operating point on dev, freeze, eval confirm
        cands = {}
        lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        lr.fit(Xtr, ytr) if len(set(ytr)) > 1 else None
        tr = DecisionTreeClassifier(max_depth=3, random_state=17)
        tr.fit(Xtr, ytr) if len(set(ytr)) > 1 else None
        models = {"logreg": (lambda X: lr.predict_proba(X)[:, 1]) if len(set(ytr)) > 1 else (lambda X: np.zeros(len(X))),
                  "tree": (lambda X: tr.predict_proba(X)[:, 1]) if len(set(ytr)) > 1 else (lambda X: np.zeros(len(X))),
                  "threshold": lambda X, _t=best_t: (X[:, 0] < _t).astype(float)}
        for name, fn in models.items():
            dev_sc = fn(Xdev)
            # operating threshold tuned on dev net utility (frozen afterwards)
            ths = sorted(set(dev_sc.tolist())) + [1.1]
            bt, bn = 0.5, net_utility(dev, dev_sc >= 0.5)["net"]
            for th in ths:
                v = net_utility(dev, dev_sc >= th)["net"]
                if v > bn:
                    bt, bn = th, v
            conf_sc = fn(Xconf)
            cands[name] = {"dev_t": round(float(bt), 4),
                           "dev_net": net_utility(dev, dev_sc >= bt),
                           "confirm_net": net_utility(conf, conf_sc >= bt),
                           "dev_auroc": round(float(roc_auc_score(ydev, dev_sc)), 4) if len(set(ydev)) > 1 else 0.5,
                           "dev_auprc": pr_auc(ydev, dev_sc)}
        # champion = argmax dev net (pre-registered selection, report all).
        # Tie-break is explicit: fixed candidate order logreg > tree > threshold,
        # first-max wins (dict insertion order). Recorded, not silent.
        champ = max(cands, key=lambda k: cands[k]["dev_net"]["net"])
        cells[key] = {"n_train": len(train), "n_dev": len(dev), "n_confirm": len(conf),
                      "n_pos": {"train": int(np.sum(ytr)), "dev": int(np.sum(ydev)),
                                "confirm": int(np.sum(labels(conf)))},
                      "n_groups_train": len(set(groups.tolist())),
                      "splits": {"train": "search_train", "dev": dev_split, "confirm": conf_split},
                      "models": ms, "candidates": cands, "champion": champ,
                      "champion_tie_break": "fixed candidate order logreg>tree>threshold, first-max wins"}
    report = {"experiment_id": "EXP-02", "created_utc": datetime.now(timezone.utc).isoformat(),
              "repo_commit": sha, "protocol_version": "v4.1",
              "analysis_script": "analyze_utility_predictability.py",
              "analysis_type": "exploratory", "selection_used": True, "confirm_used": True,
              "pools": "train OOF -> dev select (model+threshold) -> confirm frozen eval",
              "selection_rule": ("champion = argmax dev net utility over 3 candidates "
                                 "(logreg/tree/threshold; majority excluded as baseline-only); "
                                 "ties broken by fixed candidate order logreg>tree>threshold, first-max wins"),
              "lam_cost": LAM, "run_manifest": {"argv": sys.argv[1:]}, "cells": cells}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for k, c in cells.items():
        oof = {m: c["models"][m].get("oof_auroc", "-") for m in ("logreg", "tree")}
        print(k, "OOF-auroc:", oof, "| champ:", c["champion"],
              "dev-net:", c["candidates"][c["champion"]]["dev_net"]["net"],
              "conf-net:", c["candidates"][c["champion"]]["confirm_net"]["net"])


if __name__ == "__main__":
    raise SystemExit(main())
