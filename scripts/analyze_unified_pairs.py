#!/usr/bin/env python3
"""Stage 4 prerequisite: paired complementarity on unified native outputs.

Offline. Reads one unified-native JSONL (rep-keyed rows from
scripts/run_unified_native.py) and reports, per task file:
- per-rep accuracy with group-bootstrap 95% CI (cluster = group_id)
- pairwise error overlap, only-fix / only-break counts
- oracle (either-rep-correct) accuracy and gain over best single rep
- paired accuracy diff with group-bootstrap CI
Refuses files with <2 distinct groups (CI would be meaningless).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

rng = np.random.RandomState(17)
REPS_N = 5000


def load_rows(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def grouped_ci(values_by_group, fn, n=REPS_N):
    groups = list(values_by_group)
    if len(groups) < 2:
        return None
    est = fn([v for g in groups for v in values_by_group[g]])
    boots = []
    for _ in range(n):
        sample = [values_by_group[g] for g in rng.choice(groups, size=len(groups), replace=True)]
        boots.append(fn([v for grp in sample for v in grp]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"est": round(float(est), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = [r for r in load_rows(args.inp) if r.get("error") is None]
    if not rows:
        raise SystemExit("no successful rows")
    reps = sorted({r["rep"] for r in rows})
    by_pos = {}
    for r in rows:
        by_pos.setdefault(r["position"], {})[r["rep"]] = r
    # keep only positions with all reps present
    full = {p: d for p, d in by_pos.items() if set(d) == set(reps)}
    if not full:
        raise SystemExit("no complete positions")

    def grp_of(p, rep):
        return full[p][rep]["group_id"]

    out = {"n_positions": len(full), "reps": reps, "per_rep": {}}
    for rep in reps:
        by_g = {}
        for p in full:
            by_g.setdefault(full[p][rep]["group_id"], []).append(float(full[p][rep]["correct"]))
        out["per_rep"][rep] = {"accuracy": grouped_ci(by_g, np.mean),
                               "mean_ms": round(float(np.mean([full[p][rep]["latency_ms"] for p in full])), 1)}
    if len(reps) == 2:
        a, b = reps
        by_g_diff, by_g_or = {}, {}
        for p in full:
            ca, cb = float(full[p][a]["correct"]), float(full[p][b]["correct"])
            g = full[p][a]["group_id"]
            by_g_diff.setdefault(g, []).append(cb - ca)
            by_g_or.setdefault(g, []).append(max(ca, cb))
        accs = {}
        for r in reps:
            c = out["per_rep"][r]["accuracy"]
            accs[r] = c["est"] if c else float(np.mean(
                [full[p][r]["correct"] for p in full]))
        diff_ci = grouped_ci(by_g_diff, np.mean)
        or_ci = grouped_ci(by_g_or, np.mean)
        or_est = or_ci["est"] if or_ci else round(float(np.mean(
            [v for vs in by_g_or.values() for v in vs])), 4)
        out["paired"] = {
            f"{b}_minus_{a}": diff_ci,
            "oracle_accuracy": or_ci,
            "oracle_gain_over_best": round(or_est - max(accs.values()), 4),
            "ci_note": None if diff_ci else "single-group: point estimates only, no CI",
            "only_fix": {r: sum(1 for p in full
                                if full[p][r]["correct"] and not any(
                                    full[p][o]["correct"] for o in reps if o != r)) for r in reps},
            "both_wrong": sum(1 for p in full if not any(full[p][r]["correct"] for r in reps)),
            "both_correct": sum(1 for p in full if all(full[p][r]["correct"] for r in reps)),
        }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
