#!/usr/bin/env python3
"""Phase 4 evolution runner: GA vs Random Search on a policy cache (offline).

Matched unique-genome budget, multiple seeds. Fitness is cache-exact (see
src/evolution/fitness.py). Reports per-seed best, mean/std, and whether GA
beats Random and the fixed single-rep baselines. No model calls.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evolution import BudgetTracker, CascadeGenome, evaluate, ga_run, random_run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--budget", type=int, default=24)
    ap.add_argument("--seeds", default="17,23,41")
    ap.add_argument("--pop", type=int, default=8)
    ap.add_argument("--gens", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cache = json.loads(Path(args.cache).read_text(encoding="utf-8"))
    rows = cache["rows"]
    seeds = [int(x) for x in args.seeds.split(",")]
    res = {"created_utc": datetime.now(timezone.utc).isoformat(), "cache": args.cache,
           "task": cache["task"], "model": cache["model"], "n_rows": len(rows),
           "budget_unique": args.budget, "seeds": seeds, "per_seed": {}}
    for seed in seeds:
        t_ga = BudgetTracker(max_unique_evals=args.budget, max_model_calls=0)
        g = ga_run(rows, population=args.pop, generations=args.gens,
                   crossover_rate=0.8, mutation_rate=0.2, elitism=1,
                   seed=seed, tracker=t_ga)
        t_rs = BudgetTracker(max_unique_evals=args.budget, max_model_calls=0)
        r = random_run(rows, budget=args.budget, seed=seed + 1000, tracker=t_rs)
        res["per_seed"][seed] = {"ga": {"best": g["best"], "fit": g["best_fit"],
                                        "unique": g["n_unique_evals"], "log": g["generations"]},
                                 "random": {"best": r["best"], "fit": r["best_fit"],
                                            "unique": r["n_evals"]}}
    for algo in ("ga", "random"):
        accs = [res["per_seed"][s][algo]["fit"]["accuracy"] for s in seeds]
        res[f"{algo}_acc_mean"] = round(float(np.mean(accs)), 4)
        res[f"{algo}_acc_std"] = round(float(np.std(accs, ddof=1)), 4) if len(accs) > 1 else 0.0
    base_noul = evaluate(CascadeGenome(lo=0.0, hi=1.0, use_second=0), rows)
    base_switch = {"accuracy": round(sum(r["switch_correct"] for r in rows) / len(rows), 4)}
    res["fixed_noul_only"] = base_noul
    res["fixed_switch_only"] = base_switch
    res["ga_minus_random_mean"] = round(res["ga_acc_mean"] - res["random_acc_mean"], 4)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("ga_acc_mean", "ga_acc_std", "random_acc_mean",
                                          "random_acc_std", "fixed_noul_only", "fixed_switch_only",
                                          "ga_minus_random_mean")}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
