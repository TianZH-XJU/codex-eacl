#!/usr/bin/env python3
"""Paired-bootstrap significance for GA-best vs Random-best (offline).

For each evo result file and each seed: re-evaluate the GA-best and
Random-best genomes row-by-row on the shared cache, then group-bootstrap
(cluster = group_id, 5000 reps) the paired accuracy diff. Reports per-seed
diff + CI and whether any CI excludes zero. No model calls.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evolution import decide  # noqa: E402
from evolution.genome import CascadeGenome  # noqa: E402

rng = np.random.RandomState(17)


def rowwise(g, rows):
    out = []
    for r in rows:
        if r.get("noul_p") is None:
            out.append((r["group_id"], 0))
            continue
        ok, _ = decide(g, r["noul_p"], bool(r.get("switch_correct", False)), r["gold"])
        out.append((r["group_id"], int(ok)))
    return out


def paired_ci(a, b, n=5000):
    groups = sorted({g for g, _ in a})
    da = {g: [v for gg, v in a if gg == g] for g in groups}
    db = {g: [v for gg, v in b if gg == g] for g in groups}
    diffs = [db[g][i] - da[g][i] for g in groups for i in range(len(da[g]))]
    est = float(np.mean(diffs))
    boots = []
    for _ in range(n):
        s = rng.choice(groups, size=len(groups), replace=True)
        sample = [db[g][i] - da[g][i] for g in s for i in range(len(da[g]))]
        boots.append(float(np.mean(sample)))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"est": round(est, 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "excludes_zero": bool(lo > 0 or hi < 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evo", nargs="+", required=True)
    ap.add_argument("--fixed", default=None, choices=[None, "noul-only", "switch-only"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    report = {}
    for ef in args.evo:
        d = json.loads(Path(ef).read_text(encoding="utf-8"))
        cache = json.loads((ROOT / d["cache"]).read_text(encoding="utf-8"))
        rows = cache["rows"]
        per_seed = {}
        for s in d["seeds"]:
            gb = CascadeGenome(**d["per_seed"][str(s)][d.get("algo", "ga")]["best"])
            if args.fixed == "noul-only":
                other, tag = CascadeGenome(lo=0.0, hi=1.0, use_second=0), "ga_minus_noulonly"
            elif args.fixed == "switch-only":
                other, tag = CascadeGenome(lo=0.0, hi=1.0, use_second=1), "ga_minus_switchonly"
            else:
                other = CascadeGenome(**d["per_seed"][str(s)]["random"]["best"])
                tag = "ga_minus_random"
            per_seed[str(s)] = paired_ci(rowwise(other, rows), rowwise(gb, rows))
            per_seed[str(s)]["comparison"] = tag
        report[Path(ef).stem] = per_seed
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for k, v in report.items():
        print(k, {s: (d["est"], d["ci95"], "SIG" if d["excludes_zero"] else "ns")
                  for s, d in v.items()})


if __name__ == "__main__":
    raise SystemExit(main())
