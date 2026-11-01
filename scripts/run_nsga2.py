#!/usr/bin/env python3
"""NSGA-II vs random multi-objective sampling on committed caches (offline).

Genomes:
- cascade (code/findver): genes (lo, hi, use_second); objectives =
  (accuracy, -calls_per_row), both maximized.
- abstain (processbench): genes (t, use_margin, m); objectives =
  (selective_accuracy, coverage), both maximized.
Matched unique-eval budget, multiple seeds. Compares hypervolume + front
sizes. No model calls.
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evolution import (AbstainGenome, BudgetTracker, CascadeGenome, crossover,  # noqa: E402
                       decide, evaluate_math, hypervolume_2d, mutate, nsga2_run)
from evolution import evaluate as cascade_evaluate  # noqa: E402


def sample_cascade(rng):
    return CascadeGenome(lo=rng.random(), hi=rng.random(),
                         use_second=rng.randint(0, 1)).repaired()


def sample_abstain(rng):
    return AbstainGenome(t=rng.random(), use_margin=rng.randint(0, 1),
                         m=rng.random()).repaired()


def cx_abstain(rng, a, b, rate):
    if rng.random() > rate:
        return a
    va, vb = a.to_vector(), b.to_vector()
    return AbstainGenome.from_vector(
        tuple(x if rng.random() < 0.5 else y for x, y in zip(va, vb)))


def mu_abstain(rng, g, rate, sigma=0.1):
    t, um, m = g.t, g.use_margin, g.m
    if rng.random() < rate:
        t += rng.gauss(0.0, sigma)
    if rng.random() < rate:
        m += rng.gauss(0.0, sigma)
    if rng.random() < rate:
        um = 1 - um
    return AbstainGenome(t=t, use_margin=um, m=m).repaired()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--genome", required=True, choices=["cascade", "abstain"])
    ap.add_argument("--budget", type=int, default=24)
    ap.add_argument("--seeds", default="17,23,41")
    ap.add_argument("--pop", type=int, default=8)
    ap.add_argument("--gens", type=int, default=3)
    ap.add_argument("--mu", type=int, default=4)
    ap.add_argument("--lams", type=int, default=8)
    ap.add_argument("--sigma", type=float, default=0.15,
                    help="gaussian sigma for abstain-genome mutation (cascade mutate uses fixed sigma)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cache = json.loads(Path(args.cache).read_text(encoding="utf-8"))
    rows = cache["rows"]
    seeds = [int(x) for x in args.seeds.split(",")]
    import subprocess as _sp
    try:
        _sha = _sp.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                       check=True, capture_output=True, text=True).stdout.strip()
    except Exception:
        _sha = None
    res = {"created_utc": datetime.now(timezone.utc).isoformat(), "cache": args.cache,
           "genome": args.genome, "budget_unique": args.budget, "seeds": seeds,
           "params": {"pop": args.pop, "gens": args.gens,
                      "mu": getattr(args, "mu", None), "lams": getattr(args, "lams", None),
                      "sigma": getattr(args, "sigma", None)},
           "run_manifest": {"argv": sys.argv[1:], "cpu_threads": os.getenv("OMP_NUM_THREADS"),
                            "project_git_sha": _sha, "protocol": "v4.1"},
           "per_seed": {}}
    for seed in seeds:
        ref = (0.0, -2.0) if args.genome == "cascade" else (0.0, 0.0)
        if args.genome == "cascade":
            def ev(g, _rows=rows):
                f = cascade_evaluate(g, _rows)
                return ((f["accuracy"], -f["calls_per_row"]), g.to_vector(), f)
            smp, cx, mu = sample_cascade, crossover, mutate
        else:
            def ev(g, _rows=rows):
                f = evaluate_math(g, _rows)
                return ((f["sel_accuracy"], f["coverage"]), g.to_vector(), f)
            smp, cx = sample_abstain, cx_abstain
            mu = functools.partial(mu_abstain, sigma=args.sigma)
        t1 = BudgetTracker(max_unique_evals=args.budget, max_model_calls=0)
        n = nsga2_run(ev, smp, cx, mu, budget=args.budget, seed=seed,
                      population=args.pop, generations=args.gens, tracker=t1, ref=ref,
                      mu=args.mu, lams=args.lams)
        # random multi-objective sampling, same unique budget
        rng = random.Random(seed + 5000)
        seen, pts = set(), []
        while len(seen) < args.budget:
            g = smp(rng)
            key = g.to_vector()
            if key in seen:
                continue
            seen.add(key)
            pts.append(ev(g)[0])
        res["per_seed"][seed] = {
            "nsga2_hv": n["hypervolume"], "nsga2_front": len(n["pareto"]),
            "nsga2_unique": n["n_unique_evals"], "nsga2_stopped_early": n["stopped_early"],
            "pareto": n["pareto"],
            "random_hv": hypervolume_2d(pts, ref),
            "random_unique": len(seen)}
    for algo in ("nsga2_hv", "random_hv"):
        vs = [res["per_seed"][s][algo] for s in seeds]
        res[f"{algo}_mean"] = round(float(np.mean(vs)), 4)
        res[f"{algo}_std"] = round(float(np.std(vs, ddof=1)), 4) if len(vs) > 1 else 0.0
    res["hv_diff_nsga2_minus_random"] = round(res["nsga2_hv_mean"] - res["random_hv_mean"], 4)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("nsga2_hv_mean", "nsga2_hv_std", "random_hv_mean",
                                          "random_hv_std", "hv_diff_nsga2_minus_random")}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
