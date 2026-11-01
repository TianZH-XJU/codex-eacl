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
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evolution import AbstainGenome, CascadeGenome, decide, evaluate_math  # noqa: E402
from evolution import evaluate as cascade_evaluate  # noqa: E402

rng = np.random.RandomState(17)


def rowwise(g, rows):
    """Per-row correctness of a cascade genome on binary-task cache rows."""
    out = []
    for r in rows:
        if r.get("noul_p") is None:
            out.append((r["group_id"], 0))
            continue
        ok, _ = decide(g, r["noul_p"], bool(r.get("switch_correct", False)), r["gold"])
        out.append((r["group_id"], int(ok)))
    return out


def rowwise_math(g, rows):
    """Per-row correctness of an abstain genome (1 = answered correctly)."""
    out = []
    for r in rows:
        if r.get("top1") is None:
            out.append((r["group_id"], 0))
            continue
        abstain = r["top1"] < g.t or (g.use_margin and (r.get("margin") or 0.0) < g.m)
        out.append((r["group_id"], 0 if abstain else int(bool(r.get("correct", False)))))
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
    ap.add_argument("--evo", nargs="*", default=[], required=False)
    ap.add_argument("--fixed", default=None, choices=[None, "noul-only", "switch-only"])
    ap.add_argument("--nsga2", nargs="*", default=None,
                    help="nsga2 result files: deploy max-accuracy front member vs "
                         "reconstructed random-best (same sampling RNG), paired CI")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    report = {}
    for ef in args.evo:
        d = json.loads(Path(ef).read_text(encoding="utf-8"))
        cache_path = d.get("cache") or (d.get("train_cache") or {}).get("path")
        if not cache_path:
            raise SystemExit(f"evo file {ef} has neither cache nor train_cache.path")
        cache = json.loads((ROOT / cache_path).read_text(encoding="utf-8"))
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
    for nf in args.nsga2 or []:
        d = json.loads(Path(nf).read_text(encoding="utf-8"))
        cache = json.loads((ROOT / d["cache"]).read_text(encoding="utf-8"))
        rows = cache["rows"]
        is_math = "top1" in rows[0]
        per_seed = {}
        for s in d["seeds"]:
            # deploy rule (pre-registered): saved front member with max accuracy,
            # tie-break fewer calls (cascade) / higher coverage (abstain), then genome order
            front = d["per_seed"][str(s)]["pareto"]
            if is_math:
                cands = []
                for entry in front:
                    gg = AbstainGenome(t=entry["genome"][0], use_margin=int(round(entry["genome"][1])),
                                       m=entry["genome"][2])
                    f = evaluate_math(gg, rows)
                    cands.append((f["sel_accuracy"], f["coverage"], tuple(entry["genome"]), gg))
                cands.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
                deploy = cands[0][3]
                rw = rowwise_math
            else:
                cands = []
                for entry in front:
                    gg = CascadeGenome(lo=entry["genome"][0], hi=entry["genome"][1],
                                       use_second=int(round(entry["genome"][2])))
                    f = cascade_evaluate(gg, rows)
                    cands.append((f["accuracy"], -f["calls_per_row"], tuple(entry["genome"]), gg))
                cands.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
                deploy = cands[0][3]
                rw = rowwise
            # reconstruct random-best with the identical sampling RNG (seed+5000)
            from run_nsga2 import sample_abstain as _sa, sample_cascade as _sc  # noqa: E402
            r = random.Random(s + 5000)
            smp = _sa if is_math else _sc
            seen, acc_list = set(), []
            while len(seen) < d["budget_unique"]:
                gg = smp(r)
                key = gg.to_vector()
                if key in seen:
                    continue
                seen.add(key)
                if is_math:
                    f = evaluate_math(gg, rows)
                    acc_list.append((f["sel_accuracy"], f["coverage"], key, gg))
                else:
                    f = cascade_evaluate(gg, rows)
                    acc_list.append((f["accuracy"], -f["calls_per_row"], key, gg))
            acc_list.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
            rbest_g = acc_list[0][3]
            per_seed[str(s)] = paired_ci(rw(rbest_g, rows), rw(deploy, rows))
            per_seed[str(s)]["comparison"] = "nsga2deploy_minus_randombest"
        report[Path(nf).stem + "_deploy_sig"] = per_seed
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for k, v in report.items():
        print(k, {s: (d["est"], d["ci95"], "SIG" if d["excludes_zero"] else "ns")
                  for s, d in v.items()})


if __name__ == "__main__":
    raise SystemExit(main())
