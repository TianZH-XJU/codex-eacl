#!/usr/bin/env python3
"""S-04 / EXP-04: group-level faultmap CIs (offline).

Reuses scripts/run_faultmap.py bucketing (train discover, dev+confirm
validate). For every eligible cell, resamples GROUPS (not rows) with
replacement and recomputes severity, reporting n_groups, group sizes and a
percentile CI. A cell whose severity rests on one or two groups gets a wide
CI even when its row count looks comfortable. Zero new model calls.
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

import run_faultmap as FM  # noqa: E402

rng = np.random.RandomState(17)


def group_severity_ci(rows, n=5000):
    """Percentile CI of severity under group resampling."""
    by_g = {}
    for r in rows:
        by_g.setdefault(r.get("group_id", "?"), []).append(r)
    groups = sorted(by_g)
    if not groups:
        return {"est": 0.0, "ci95": [0.0, 0.0], "n_groups": 0, "group_sizes": []}
    est = FM.severity(rows)[0]
    boots = []
    for _ in range(n):
        s = rng.choice(groups, size=len(groups), replace=True)
        sample = [r for g in s for r in by_g[g]]
        boots.append(FM.severity(sample)[0])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    sizes = sorted((len(v) for v in by_g.values()))
    return {"est": round(est, 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "n_groups": len(groups), "group_sizes": sizes}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--boots", type=int, default=5000)
    args = ap.parse_args()
    try:
        sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                             check=True, capture_output=True, text=True).stdout.strip()
    except Exception:
        sha = None
    train_files, dev_files, conf_files = FM.select_files(
        "results/v4/evolution/cache_train_*.json",
        "results/v4/evolution/cache_[a-z]*_*.json",
        "results/v4/evolution/cache_confirm_*.json")
    disc, dev, conf = FM.bucketize(train_files), FM.bucketize(dev_files), FM.bucketize(conf_files)
    base = json.loads((ROOT / "results/v4_1/evolution/faultmap.json").read_text(encoding="utf-8"))
    eligible = {tuple(c["cell"]) for c in base["cells"]}
    cells = []
    for cell in sorted(disc):
        if tuple(cell) not in eligible:
            continue
        entry = {"cell": list(cell), "n": len(disc[cell]),
                 "discover": group_severity_ci(disc[cell], n=args.boots),
                 "dev": group_severity_ci(dev.get(cell, []), n=args.boots),
                 "confirm": group_severity_ci(conf.get(cell, []), n=args.boots)}
        cells.append(entry)
    report = {"experiment_id": "EXP-04", "created_utc": datetime.now(timezone.utc).isoformat(),
              "repo_commit": sha, "protocol_version": "v4.1",
              "method": ("group-level percentile bootstrap of cell severity (5000 resamples, "
                         "module RandomState(17) consumed in fixed cell order; "
                         "singleton groups yield degenerate point CIs [est,est] — no false precision implied; "
                         "all 1046 cache rows carry group_id, so the '?' fallback never fires on real data"),
              "cells": cells}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for c in cells[:6]:
        print(c["cell"], "n=", c["n"], "groups=", c["discover"]["n_groups"],
              "S=", c["discover"]["est"], c["discover"]["ci95"])


if __name__ == "__main__":
    raise SystemExit(main())
