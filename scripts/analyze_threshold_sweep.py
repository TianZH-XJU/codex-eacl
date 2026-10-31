#!/usr/bin/env python3
"""Offline threshold sweep on saved A2 P(true) scores (no model calls).

For each (model, task, semantic_noul): sweep thresholds, report balanced
accuracy / macro-F1 operating points + AUROC. Tests whether any threshold
rescues the collapsed Noul outputs.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="results/v2/schema_ablation/*.json")
    ap.add_argument("--out", default="results/v2/schema_ablation/threshold_sweep.json")
    args = ap.parse_args()
    out = {}
    for path in sorted(glob.glob(args.glob)):
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        key = Path(path).stem
        rows = [r for r in d["results"] if r["rep"] == "semantic_noul" and r["error"] is None]
        y = np.array([int(r["gold"]) for r in rows])
        s = np.array([r["score"] for r in rows])
        thr = sorted(set([0.0] + [float(x) for x in np.quantile(s, [0.1, 0.25, 0.5, 0.75, 0.9])] + [1.0]))
        ops = []
        for t in thr:
            p = (s >= t).astype(int)
            ops.append({"threshold": round(t, 4),
                        "balanced_accuracy": round(float(balanced_accuracy_score(y, p)), 4),
                        "macro_f1": round(float(f1_score(y, p, average="macro", zero_division=0)), 4)})
        best = max(ops, key=lambda o: o["balanced_accuracy"])
        out[key] = {"n": len(rows), "operating_points": ops, "best_balanced_acc": best}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    for k, v in out.items():
        print(k, "n=", v["n"], "best:", v["best_balanced_acc"])


if __name__ == "__main__":
    raise SystemExit(main())
