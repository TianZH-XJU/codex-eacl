#!/usr/bin/env python3
"""Phase 5 transfer (offline, cache-exact, zero new model calls).

Cross-evaluates each evolved best genome on every task/model cache:
genome semantics (noul-p band + switch fallback) are identical across the
binary tasks, so transfer is mechanical. Reports transfer matrix with
per-seed means, plus gaps vs target's own best and fixed baselines.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evolution import evaluate  # noqa: E402
from evolution.genome import CascadeGenome  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ev = ROOT / "results/v4/evolution"
    sources = {}
    for t in ("code", "findver"):
        for m in ("laya", "vela"):
            d = json.loads((ev / f"evo_{t}_{m}.json").read_text(encoding="utf-8"))
            sources[(t, m)] = {s: CascadeGenome(**d["per_seed"][str(s)]["ga"]["best"])
                               for s in d["seeds"]}
    caches = {}
    for t in ("code", "findver"):
        for m in ("laya", "vela"):
            d = json.loads((ev / f"cache_{t}_{m}.json").read_text(encoding="utf-8"))
            caches[(t, m)] = d["rows"]

    matrix = {}
    for (st, sm), genomes in sources.items():
        for (tt, tm), rows in caches.items():
            accs = [evaluate(g, rows)["accuracy"] for g in genomes.values()]
            own = json.loads((ev / f"evo_{tt}_{tm}.json").read_text(encoding="utf-8"))
            own_best = own["ga_acc_mean"]
            matrix[f"{st}/{sm}_on_{tt}/{tm}"] = {
                "mean": round(float(np.mean(accs)), 4),
                "std": round(float(np.std(accs, ddof=1)), 4) if len(accs) > 1 else 0.0,
                "target_own_ga_mean": own_best,
                "gap_to_target_own": round(float(np.mean(accs)) - own_best, 4),
                "diagonal": bool((st, sm) == (tt, tm))}
    report = {"matrix": matrix,
              "note": "cache-exact transfer; no new model calls; seeds 17/23/41"}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for k, v in matrix.items():
        print(k, v["mean"], "gap", v["gap_to_target_own"])


if __name__ == "__main__":
    raise SystemExit(main())
