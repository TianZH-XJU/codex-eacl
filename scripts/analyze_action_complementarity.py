#!/usr/bin/env python3
"""S-01 / EXP-01: action complementarity + oracle headroom (offline).

For each task x model train pool with a noul/switch action pair:
best-fixed accuracy, per-example oracle upper bound, headroom, and the
repairable / harmful / neutral split relative to the best fixed action,
with group-level bootstrap CIs. Math has a single representation:
best fixed == oracle, headroom is 0 by construction (recorded, not searched).

Oracle is an in-action-space upper bound (perfect per-example routing), not a
model capability bound. Zero new model calls.
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

rng = np.random.RandomState(17)

TASKS = {
    "code_laya": "results/v4/evolution/cache_train_code_laya.json",
    "code_vela": "results/v4/evolution/cache_train_code_vela.json",
    "findver_laya": "results/v4/evolution/cache_train_findver_laya.json",
    "findver_vela": "results/v4/evolution/cache_train_findver_vela.json",
    "math_laya": "results/v4/evolution/cache_train_math_laya.json",
    "math_vela": "results/v4/evolution/cache_train_math_vela.json",
}


def load(path):
    d = json.loads((ROOT / path).read_text(encoding="utf-8"))
    return d["rows"], d.get("split", "unknown")


def group_bootstrap_ci(values_by_group, stat, n=5000):
    """Percentile CI of `stat(flat_values)` under group resampling."""
    groups = sorted(values_by_group)
    if not groups:
        return {"est": 0.0, "ci95": [0.0, 0.0]}
    est = stat([v for g in groups for v in values_by_group[g]])
    boots = []
    for _ in range(n):
        s = rng.choice(groups, size=len(groups), replace=True)
        boots.append(stat([v for g in s for v in values_by_group[g]]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"est": round(float(est), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)]}


def cell_stats(rows):
    """Pure function: complementarity decomposition of one pool."""
    if not rows:
        raise ValueError("cell_stats: empty rows (fail-closed, no silent scalar)")
    r = [int((float(x["noul_p"]) >= 0.5) == bool(x["gold"])) for x in rows]
    s = [int(bool(x.get("switch_correct", False))) for x in rows]
    n = len(rows)
    acc_r = sum(r) / n
    acc_s = sum(s) / n
    base, alt = (r, s) if acc_r >= acc_s else (s, r)
    base_name = "noul" if acc_r >= acc_s else "switch"
    oracle = sum(1 for a, b in zip(r, s) if a or b) / n
    repairable = sum(1 for a, b in zip(base, alt) if not a and b)
    harmful = sum(1 for a, b in zip(base, alt) if a and not b)
    return {"n": n, "base_action": base_name,
            "base_acc": round(sum(base) / n, 4),
            "alt_acc": round(sum(alt) / n, 4),
            "oracle_acc": round(oracle, 4),
            "headroom": round(oracle - sum(base) / n, 4),
            "repairable": repairable, "harmful": harmful,
            "neutral": n - repairable - harmful,
            "disagreement": round(sum(1 for a, b in zip(r, s) if a != b) / n, 4)}


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
    cells = {}
    for key, path in TASKS.items():
        rows, split = load(path)
        if not rows:
            raise SystemExit(f"empty cache pool {path}")
        has_p = ["noul_p" in r for r in rows]
        if any(has_p) and not all(has_p):
            raise SystemExit(f"mixed schema in {path}: some rows lack noul_p")
        if "noul_p" not in rows[0]:
            acc = sum(1 for r in rows if r.get("correct")) / len(rows)
            cells[key] = {"n": len(rows), "split": split,
                          "n_groups": len({r["group_id"] for r in rows}),
                          "note": "single representation: best fixed == oracle",
                          "base_acc": round(acc, 4), "oracle_acc": round(acc, 4),
                          "headroom": 0.0}
            continue
        st = cell_stats(rows)
        by_group_o, by_group_rep, by_group_net = {}, {}, {}
        for x in rows:
            a = int((float(x["noul_p"]) >= 0.5) == bool(x["gold"]))
            b = int(bool(x.get("switch_correct", False)))
            by_group_o.setdefault(x["group_id"], []).append(int(a or b))
            base_is_r = st["base_action"] == "noul"
            ba, al = (a, b) if base_is_r else (b, a)
            # headroom == repair/n exactly (oracle - base = P(alt fixes base));
            # net additionally subtracts regression.
            by_group_rep.setdefault(x["group_id"], []).append(int((not ba) and al))
            by_group_net.setdefault(x["group_id"], []).append(int((not ba) and al) - int(ba and not al))
        st["oracle_acc_ci"] = group_bootstrap_ci(by_group_o, lambda v: sum(v) / len(v), n=args.boots)
        st["headroom_ci"] = group_bootstrap_ci(by_group_rep, lambda v: sum(v) / len(v), n=args.boots)
        st["net_repair"] = st["repairable"] - st["harmful"]
        st["net_repair_ci"] = group_bootstrap_ci(by_group_net, lambda v: sum(v) / len(v), n=args.boots)
        groups = sorted({x["group_id"] for x in rows})
        st["n_groups"] = len(groups)
        st["split"] = split
        cells[key] = st
    report = {"experiment_id": "EXP-01", "created_utc": datetime.now(timezone.utc).isoformat(),
              "repo_commit": sha, "protocol_version": "v4.1",
              "pools": "search_train (in-sample screen; headroom is an upper bound, not a method)",
              "ci_method": ("group-level percentile bootstrap (5000 resamples, module RandomState(17), "
                            "fixed TASKS call order); singleton groups (code: 60/60) reduce to row "
                            "bootstrap with on-grid CIs, multi-row groups (findver: 41 pools) give off-grid CIs"),
              "cells": cells}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for k, c in cells.items():
        print(k, {kk: c.get(kk) for kk in ("n", "base_acc", "oracle_acc", "headroom", "repairable", "harmful") if kk in c})


if __name__ == "__main__":
    raise SystemExit(main())
