#!/usr/bin/env python3
"""Fixed-action macro-F1 on dev/confirm (offline E1 completion).

The paper's primary metric is macro-F1, but Table 1 reports accuracy-level
repair counts.  This script computes the fixed Noul and fixed Choice macro-F1
on selection_dev and confirm for the four binary cells, plus the paired
group-bootstrap difference (Choice - Noul).  No model calls.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RNG = np.random.RandomState(17)

TASKS = {
    "code_laya": (
        "results/v4/evolution/cache_code_laya.json",
        "results/v4/evolution/cache_confirm_code_laya.json",
    ),
    "code_vela": (
        "results/v4/evolution/cache_code_vela.json",
        "results/v4/evolution/cache_confirm_code_vela.json",
    ),
    "findver_laya": (
        "results/v4/evolution/cache_findver_laya.json",
        "results/v4/evolution/cache_confirm_findver_laya.json",
    ),
    "findver_vela": (
        "results/v4/evolution/cache_findver_vela.json",
        "results/v4/evolution/cache_confirm_findver_vela.json",
    ),
}

POSITIVE = {
    "code": {"vulnerable"},
    "findver": {"supported"},
}


def task_kind(key: str) -> str:
    return "code" if key.startswith("code") else "findver"


def load(path):
    d = json.loads((ROOT / path).read_text(encoding="utf-8"))
    return d["rows"], d.get("split", "unknown")


def preds(rows, kind):
    gold = np.array([bool(r["gold"]) for r in rows], dtype=bool)
    noul = np.array([(float(r["noul_p"]) >= 0.5) for r in rows], dtype=bool)
    pos = POSITIVE[kind]
    choice = np.array([str(r["switch_choice"]) in pos for r in rows], dtype=bool)
    return gold, noul, choice


def macro_f1_np(gold, pred):
    """Fast binary macro-F1 without per-call sklearn overhead."""
    f1s = []
    for cls in (False, True):
        tp = float(np.sum((gold == cls) & (pred == cls)))
        fp = float(np.sum((gold != cls) & (pred == cls)))
        fn = float(np.sum((gold == cls) & (pred != cls)))
        denom = 2.0 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else 2.0 * tp / denom)
    return float(np.mean(f1s))


def metrics(gold, pred):
    return {
        "macro_f1": round(macro_f1_np(gold, pred), 4),
        "accuracy": round(float(np.mean(gold == pred)), 4),
    }


def group_ci(rows, kind, metric: str, n_boot=5000):  # noqa: PLR0913
    gold, noul, choice = preds(rows, kind)
    by_group = {}
    for i, r in enumerate(rows):
        by_group.setdefault(r["group_id"], []).append(i)
    groups = sorted(by_group)
    diffs = []
    for _ in range(n_boot):
        sample = RNG.choice(groups, size=len(groups), replace=True)
        idx = [i for g in sample for i in by_group[g]]
        g = gold[idx]
        if len(set(g.tolist())) < 2:
            continue
        a = metrics(g, noul[idx])[metric]
        b = metrics(g, choice[idx])[metric]
        diffs.append(b - a)
    if not diffs:
        return {"est": 0.0, "ci95": [0.0, 0.0], "n_boot": 0}
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    est = metrics(gold, choice)[metric] - metrics(gold, noul)[metric]
    return {
        "est": round(float(est), 4),
        "ci95": [round(float(lo), 4), round(float(hi), 4)],
        "n_boot": len(diffs),
    }


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
    for key, (dev_p, conf_p) in TASKS.items():
        kind = task_kind(key)
        dev, dev_split = load(dev_p)
        conf, conf_split = load(conf_p)
        row = {"n_dev": len(dev), "n_confirm": len(conf),
               "n_groups": {"dev": len({r["group_id"] for r in dev}),
                            "confirm": len({r["group_id"] for r in conf})},
               "splits": {"dev": dev_split, "confirm": conf_split},
               "dev": {}, "confirm": {}}
        for split, rows in (("dev", dev), ("confirm", conf)):
            g, n, c = preds(rows, kind)
            row[split] = {
                "noul": metrics(g, n),
                "choice": metrics(g, c),
                "delta_macro_f1": group_ci(rows, kind, "macro_f1", n_boot=args.boots),
                "delta_accuracy": group_ci(rows, kind, "accuracy", n_boot=args.boots),
            }
        cells[key] = row
    report = {
        "experiment_id": "EXP-01B",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repo_commit": sha,
        "protocol_version": "v4.1",
        "analysis_script": "analyze_fixed_macro_f1.py",
        "analysis_type": "exploratory",
        "target": "fixed Noul vs fixed Choice macro-F1 on held-out pools",
        "cells": cells,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for k, c in cells.items():
        print(k, "dev", c["dev"]["delta_macro_f1"], "confirm", c["confirm"]["delta_macro_f1"])


if __name__ == "__main__":
    raise SystemExit(main())
