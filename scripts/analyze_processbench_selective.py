#!/usr/bin/env python3
"""ProcessBench selective-prediction table (offline).

The paper treats ProcessBench as an abstention task, not as a Noul-Choice
complementarity pair.  This script reports a standard top-1 confidence
threshold baseline on selection_dev and confirm: for target coverages, pick the
threshold on dev, freeze it, and report confirm coverage / selective accuracy /
risk, plus the area under the risk-coverage curve (AURC).  No model calls.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASKS = {
    "math_laya": ("results/v4/evolution/cache_math_laya.json",
                  "results/v4/evolution/cache_confirm_math_laya.json"),
    "math_vela": ("results/v4/evolution/cache_math_vela.json",
                  "results/v4/evolution/cache_confirm_math_vela.json"),
}
TARGETS = (1.0, 0.75, 0.5, 0.25)


def load(path):
    d = json.loads((ROOT / path).read_text(encoding="utf-8"))
    return d["rows"], d.get("split", "unknown")


def curve(rows, threshold):
    sel = [r for r in rows if float(r["top1"]) >= threshold]
    if not sel:
        return {"coverage": 0.0, "selective_accuracy": None, "risk": None, "n": 0}
    acc = sum(bool(r["correct"]) for r in sel) / len(sel)
    return {"coverage": round(len(sel) / len(rows), 4),
            "selective_accuracy": round(acc, 4),
            "risk": round(1.0 - acc, 4), "n": len(sel)}


def pick_threshold(rows, target):
    scores = sorted({float(r["top1"]) for r in rows})
    best = None
    for t in scores:
        c = curve(rows, t)
        key = (abs(c["coverage"] - target), -c["coverage"])
        if best is None or key < best[0]:
            best = (key, t, c)
    return best[1], best[2]


def aurc(rows):
    ordered = sorted(rows, key=lambda r: -float(r["top1"]))
    risks, covs = [], []
    correct = 0
    for i, r in enumerate(ordered, 1):
        correct += int(bool(r["correct"]))
        covs.append(i / len(ordered))
        risks.append(1.0 - correct / i)
    # trapezoidal area under risk-coverage; lower is better
    area = float(np.trapezoid(risks, covs))
    return round(area, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
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
        dev, dev_split = load(dev_p)
        conf, conf_split = load(conf_p)
        out = {"n_dev": len(dev), "n_confirm": len(conf),
               "n_groups": {"dev": len({r["group_id"] for r in dev}),
                            "confirm": len({r["group_id"] for r in conf})},
               "splits": {"dev": dev_split, "confirm": conf_split},
               "full_coverage": curve(conf, -1.0),
               "aurc_confirm": aurc(conf), "points": []}
        for target in TARGETS:
            t, dev_pt = pick_threshold(dev, target)
            conf_pt = curve(conf, t)
            out["points"].append({"target_coverage": target, "dev_threshold": round(float(t), 4),
                                  "dev": dev_pt, "confirm": conf_pt})
        cells[key] = out
    report = {"experiment_id": "EXP-05B", "created_utc": datetime.now(timezone.utc).isoformat(),
              "repo_commit": sha, "protocol_version": "v4.1",
              "analysis_script": "analyze_processbench_selective.py",
              "analysis_type": "exploratory",
              "policy": "answer if top1 >= threshold; threshold selected on selection_dev",
              "cells": cells}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for k, c in cells.items():
        print(k, "full", c["full_coverage"], "AURC", c["aurc_confirm"])
        for p in c["points"]:
            print(" ", p["target_coverage"], "dev", p["dev"]["coverage"], "conf",
                  p["confirm"]["coverage"], p["confirm"]["selective_accuracy"])


if __name__ == "__main__":
    raise SystemExit(main())
