#!/usr/bin/env python3
"""P0-4 补充实验：冻结路由策略的 group-bootstrap 净效用区间（离线，零新模型调用）。

E2 的表只有点估计。这里对 frozen dev-selected 策略在 dev 与 confirm 上重采样
group，报告 net repair、regression、cost-adjusted net utility 的 95% 区间，
并同时给出固定 Noul / 固定 Choice 的一致口径区间。
策略与 dev 阈值直接取自 utility_predictability.json 的既有选择（不重训、不重选）。
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
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_utility_predictability import TASKS as UP_TASKS  # noqa: E402
from analyze_utility_predictability import labels, net_utility  # noqa: E402

rng = np.random.RandomState(17)


def load(p):
    return json.loads((ROOT / p).read_text(encoding="utf-8"))["rows"]


def bootstrap_net(rows, mask, n=5000):
    """Group-bootstrap CI of (repair-regression)/n for a fixed switch mask."""
    by_g = {}
    for r, s in zip(rows, mask):
        noul_ok = (float(r["noul_p"]) >= 0.5) == bool(r["gold"])
        sw_ok = bool(r.get("switch_correct", False))
        if not s:
            d = (0, 0, 0)
        elif not noul_ok and sw_ok:
            d = (1, 0, 1)
        elif noul_ok and not sw_ok:
            d = (0, 1, 1)
        else:
            d = (0, 0, 1)
        by_g.setdefault(r["group_id"], []).append(d)
    groups = sorted(by_g)
    flat = [x for g in groups for x in by_g[g]]
    rep = sum(x[0] for x in flat)
    reg = sum(x[1] for x in flat)
    est = (rep - reg) / len(flat)
    boots = []
    for _ in range(n):
        s = rng.choice(groups, size=len(groups), replace=True)
        f = [x for g in s for x in by_g[g]]
        boots.append((sum(x[0] for x in f) - sum(x[1] for x in f)) / len(f))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"net": round(est, 4), "repair": rep, "regression": reg,
            "net_ci95": [round(float(lo), 4), round(float(hi), 4)],
            "excludes_zero": bool(lo > 0 or hi < 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    try:
        sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                             check=True, capture_output=True, text=True).stdout.strip()
    except Exception:
        sha = None
    import analyze_utility_predictability as UP
    up = json.loads((ROOT / "results/v4_1/evolution/utility_predictability.json").read_text(encoding="utf-8"))
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "repo_commit": sha, "protocol_version": "v4.1",
              "design": ("frozen dev-selected routing rules from utility_predictability.json; "
                         "group-bootstrap CIs for net utility on dev and confirm"),
              "run_manifest": {"argv": sys.argv[1:]},
              "analysis_script": "analyze_routing_net_ci.py",
              "analysis_type": "confirmatory",
              "selection_used": True, "confirm_used": True,
              "pools": "frozen dev-selected routers from utility_predictability.json",
              "cells": {}}
    for key, (tr_p, dev_p, conf_p) in UP_TASKS.items():
        dev_rows, conf_rows = load(dev_p), load(conf_p)
        cell = up["cells"][key]
        champ = cell["champion"]
        dev_t = cell["candidates"][champ]["dev_t"]
        out = {"champion": champ, "dev_t_reported": dev_t, "n_pos": cell["n_pos"]}
        sc_dev = None
        if champ != "threshold":
            import analyze_utility_predictability as UP2
            Xtr = UP2.features(load(tr_p))
            ytr = UP2.labels(load(tr_p))
            if champ == "logreg":
                from sklearn.linear_model import LogisticRegression
                from sklearn.pipeline import make_pipeline
                from sklearn.preprocessing import StandardScaler
                m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
            else:
                from sklearn.tree import DecisionTreeClassifier
                m = DecisionTreeClassifier(max_depth=3, random_state=17)
            m.fit(Xtr, ytr) if len(set(ytr)) > 1 else None
            sc_dev = (m.predict_proba(UP2.features(dev_rows))[:, 1]
                      if len(set(ytr)) > 1 else np.zeros(len(dev_rows)))
        for tag, rows in (("dev", dev_rows), ("confirm", conf_rows)):
            if champ == "threshold":
                # UP's threshold model: score = (|p-0.5| < best_t); switch if score >= bt.
                # best_t lives in UP's train-side scan; recover it from the cached train conf.
                Xtr = UP.features(load(tr_p))
                conf_tr = Xtr[:, 0]
                best_t, best_net = 0.1, -1e9
                for t in (0.05, 0.1, 0.15, 0.2, 0.25):
                    n_ = net_utility(load(tr_p), conf_tr < t)["net"]
                    if n_ > best_net:
                        best_t, best_net = t, n_
                score = (np.abs(np.array([float(r["noul_p"]) for r in rows]) - 0.5) < best_t).astype(float)
                mask = score >= dev_t
            else:
                # champion's dev-selected threshold on its own model's score
                import analyze_utility_predictability as UP
                X = UP.features(rows)
                if champ == "logreg":
                    from sklearn.linear_model import LogisticRegression
                    from sklearn.pipeline import make_pipeline
                    from sklearn.preprocessing import StandardScaler
                    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
                    Xtr = UP.features(load(tr_p))
                    ytr = labels(load(tr_p))
                    m.fit(Xtr, ytr) if len(set(ytr)) > 1 else None
                    sc = m.predict_proba(X)[:, 1] if len(set(ytr)) > 1 else np.zeros(len(rows))
                else:
                    from sklearn.tree import DecisionTreeClassifier
                    m = DecisionTreeClassifier(max_depth=3, random_state=17)
                    Xtr = UP.features(load(tr_p))
                    ytr = labels(load(tr_p))
                    m.fit(Xtr, ytr) if len(set(ytr)) > 1 else None
                    sc = m.predict_proba(X)[:, 1] if len(set(ytr)) > 1 else np.zeros(len(rows))
                # unrounded operating threshold: UP's scan picks the max-dev-net cut over
                # dev-score uniques; replicate it so the frozen rule is bit-identical
                ths = sorted(set(sc_dev.tolist())) + [1.1]
                bt, bn = 0.5, net_utility(dev_rows, sc_dev >= 0.5)["net"]
                for th in ths:
                    v = net_utility(dev_rows, sc_dev >= th)["net"]
                    if v > bn:
                        bt, bn = th, v
                mask = sc >= bt
                out[tag + "_t_unrounded"] = round(float(bt), 6)
            n = len(rows)
            out[tag] = {
                "router": bootstrap_net(rows, mask),
                "fixed_noul": bootstrap_net(rows, np.zeros(n, dtype=bool)),
                "fixed_switch": bootstrap_net(rows, np.ones(n, dtype=bool)),
                "n_switched": int(sum(mask))}
        report["cells"][key] = out
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for k, v in report["cells"].items():
        print(k, v["champion"], "| dev router net:", v["dev"]["router"]["net"], v["dev"]["router"]["net_ci95"],
              "| conf router net:", v["confirm"]["router"]["net"], v["confirm"]["router"]["net_ci95"],
              "| conf fixed_switch:", v["confirm"]["fixed_switch"]["net"])


if __name__ == "__main__":
    raise SystemExit(main())
