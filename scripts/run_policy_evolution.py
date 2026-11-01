#!/usr/bin/env python3
"""v4.1 evolution runner: search on TRAIN cache, report on DEV/CONFIRM (offline).

Matched unique-genome budget, multiple seeds. Fitness is cache-exact.
v4.1 rules enforced here:
- selection metric comes from --metric (task primary; code/findver: macro_f1).
- generations are derived from the unique-eval budget: gens_eff =
  min(gens, (budget - pop) // max(pop - elit, 1)); requested vs effective
  recorded, never silently overflowing.
- per-seed best is evaluated on dev/confirm caches WITHOUT selection there.
- run_manifest records CLI, env, code SHA, protocol version.
No model calls.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evolution import BudgetTracker, CascadeGenome, evaluate, es_run, ga_run, random_run  # noqa: E402
from eval_trainselected_on_dev import assert_disjoint_pool  # noqa: E402


def git_sha(path):
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                              check=True, capture_output=True, text=True).stdout.strip()
    except Exception:
        return None


def load_rows(cache_path):
    d = json.loads(Path(cache_path).read_text(encoding="utf-8"))
    return d["rows"], {"task": d["task"], "model": d["model"], "split": d.get("split", "unknown"),
                       "n": len(d["rows"]), "path": str(cache_path)}


def acc_of(genome, rows):
    f = evaluate(genome, rows)
    return f["accuracy"], f["macro_f1"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-cache", default=None)
    ap.add_argument("--cache", default=None, help="legacy alias for --train-cache")
    ap.add_argument("--dev-cache", default=None)
    ap.add_argument("--confirm-cache", default=None)
    ap.add_argument("--budget", type=int, default=24)
    ap.add_argument("--seeds", default="17,23,41")
    ap.add_argument("--pop", type=int, default=8)
    ap.add_argument("--gens", type=int, default=3)
    ap.add_argument("--mut", type=float, default=0.2)
    ap.add_argument("--cross", type=float, default=0.8)
    ap.add_argument("--lam", type=float, default=0.05)
    ap.add_argument("--metric", default="macro_f1",
                    choices=["accuracy", "macro_f1", "balanced_accuracy"])
    ap.add_argument("--elit", type=int, default=1)
    ap.add_argument("--algo", default="ga", choices=["ga", "es"])
    ap.add_argument("--mu", type=int, default=4)
    ap.add_argument("--lams", type=int, default=8)
    ap.add_argument("--sigma", type=float, default=0.15)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    train_path = args.train_cache or args.cache
    if not train_path:
        raise SystemExit("need --train-cache (or legacy --cache)")
    rows, train_meta = load_rows(train_path)
    dev_rows, dev_meta = (load_rows(args.dev_cache) if args.dev_cache else (None, None))
    conf_rows, conf_meta = (load_rows(args.confirm_cache) if args.confirm_cache else (None, None))
    # Fail-closed pool hygiene: report pools must share no row keys with train.
    if dev_rows is not None:
        assert_disjoint_pool(rows, dev_rows, f"dev-cache {args.dev_cache}")
    if conf_rows is not None:
        assert_disjoint_pool(rows, conf_rows, f"confirm-cache {args.confirm_cache}")

    # B ~= P + G*(P-E): cap generations to the budget, record both values.
    denom = max(args.pop - args.elit, 1)
    gens_eff = min(args.gens, max(0, (args.budget - args.pop) // denom))
    seeds = [int(x) for x in args.seeds.split(",")]
    res = {"created_utc": datetime.now(timezone.utc).isoformat(),
           "protocol": "v4.1", "fitness_version": "v2-task-primary",
           "train_cache": train_meta, "dev_cache": dev_meta, "confirm_cache": conf_meta,
           "budget_unique": args.budget, "seeds": seeds, "per_seed": {},
           "gens_requested": args.gens, "gens_effective": gens_eff,
           "run_manifest": {"argv": sys.argv[1:], "cpu_threads": os.getenv("OMP_NUM_THREADS"),
                            "project_git_sha": git_sha(ROOT),
                            "metric": args.metric, "lam": args.lam, "algo": args.algo}}
    for seed in seeds:
        t_ga = BudgetTracker(max_unique_evals=args.budget, max_model_calls=0)
        if args.algo == "es":
            g = es_run(rows, mu=args.mu, lam=args.lams, sigma=args.sigma,
                       budget=args.budget, seed=seed, tracker=t_ga, lam_w=args.lam,
                       metric=args.metric)
        else:
            g = ga_run(rows, population=args.pop, generations=gens_eff,
                       crossover_rate=args.cross, mutation_rate=args.mut, elitism=args.elit,
                       seed=seed, tracker=t_ga, lam=args.lam, metric=args.metric)
        t_rs = BudgetTracker(max_unique_evals=args.budget, max_model_calls=0)
        r = random_run(rows, budget=args.budget, seed=seed + 1000, tracker=t_rs,
                       lam=args.lam, metric=args.metric)
        gb = CascadeGenome(**g["best"])
        rb = CascadeGenome(**r["best"])
        entry = {args.algo: {"best": g["best"], "fit_train": g["best_fit"],
                             "unique": g["n_unique_evals"], "log": g["generations"],
                             "budget_state": t_ga.state()},
                 "random": {"best": r["best"], "fit_train": r["best_fit"],
                            "unique": r["n_evals"]}}
        for tag, eval_rows in (("dev", dev_rows), ("confirm", conf_rows)):
            if eval_rows is None:
                continue
            entry[tag] = {
                "ga": {"accuracy": acc_of(gb, eval_rows)[0], "macro_f1": acc_of(gb, eval_rows)[1]},
                "random": {"accuracy": acc_of(rb, eval_rows)[0], "macro_f1": acc_of(rb, eval_rows)[1]},
                "fixed_noul": acc_of(CascadeGenome(lo=0.0, hi=1.0, use_second=0), eval_rows),
                "fixed_switch": acc_of(CascadeGenome(lo=0.0, hi=1.0, use_second=1), eval_rows)}
        res["per_seed"][seed] = entry
    res["algo"] = args.algo
    for algo in (args.algo, "random"):
        accs = [res["per_seed"][s][algo]["fit_train"]["scalar"] for s in seeds]
        res[f"{algo}_train_scalar_mean"] = round(float(np.mean(accs)), 4)
        res[f"{algo}_train_scalar_std"] = round(float(np.std(accs, ddof=1)), 4) if len(accs) > 1 else 0.0
    res[f"{args.algo}_minus_random_mean"] = round(
        res[f"{args.algo}_train_scalar_mean"] - res["random_train_scalar_mean"], 4)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: res[k] for k in (f"{args.algo}_train_scalar_mean", f"{args.algo}_train_scalar_std",
                                          "random_train_scalar_mean", "random_train_scalar_std",
                                          f"{args.algo}_minus_random_mean", "gens_effective")}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
