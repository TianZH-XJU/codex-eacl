#!/usr/bin/env python3
"""P0-2 isolation: search on TRAIN caches, report on DEV caches (offline).

For code/findver x laya/vela: GA + Random on train (matched budget 24,
seeds 17/23/41); per-seed best evaluated on dev rows; paired group-bootstrap
CIs vs fixed noul-only / switch-only on the SAME dev rows. No selection on dev.
For math x laya/vela: NSGA-II on train; deploy max-accuracy front member on dev;
report selective accuracy/coverage + fixed never-abstain + fixed thresholds.
No model calls. Writes train_search_dev_eval.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evolution import (AbstainGenome, BudgetTracker, CascadeGenome, evaluate,  # noqa: E402
                       evaluate_math, ga_run, nsga2_run, random_run)
from test_evo_significance import paired_ci, rowwise  # noqa: E402

EV = ROOT / "results" / "v4" / "evolution"
rng = np.random.RandomState(17)


def load(cache):
    d = json.loads((EV / cache).read_text(encoding="utf-8"))
    return d["rows"]


def assert_disjoint_pool(train_rows, report_rows, report_name):
    """Leakage guard: report rows must share no row keys with the search pool."""
    tr = {r["key"] for r in train_rows}
    rp = {r["key"] for r in report_rows}
    overlap = tr & rp
    if overlap:
        raise SystemExit(f"pool leak: {len(overlap)} keys shared with {report_name}: "
                         f"{sorted(overlap)[:5]}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=24)
    ap.add_argument("--seeds", default="17,23,41")
    ap.add_argument("--report-split", default="search_dev", choices=["search_dev", "confirm"],
                    help="split to REPORT on (never select on). search_train is always the search pool.")
    ap.add_argument("--evo-dir", default="results/v4_1/evolution",
                    help="directory with evo_train_*.json holding the train-selected best genomes")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]
    report = {"budget_unique": args.budget, "seeds": seeds,
              "search_pool": "search_train", "report_split": args.report_split, "tasks": {},
              "selection_source": "train-selected best genomes from evo files (no re-search here)"}
    dev_tag = "" if args.report_split == "search_dev" else "_confirm"
    dev_file = lambda t, m: f"cache{dev_tag}_{t}_{m}.json"

    def load_evo(task, model):
        p = ROOT / args.evo_dir / f"evo_train_{task}_{model}.json"
        return json.loads(Path(p).read_text(encoding="utf-8"))

    for t in ("code", "findver"):
        for m in ("laya", "vela"):
            train = load(f"cache_train_{t}_{m}.json")
            dev = load(dev_file(t, m))
            assert_disjoint_pool(train, dev, dev_file(t, m))
            evo = load_evo(t, m)
            algo = evo.get("algo", "ga")
            per_seed = {}
            for s in seeds:
                gb = CascadeGenome(**evo["per_seed"][str(s)][algo]["best"])
                rb = CascadeGenome(**evo["per_seed"][str(s)]["random"]["best"])
                base_n = CascadeGenome(lo=0.0, hi=1.0, use_second=0)
                base_s = CascadeGenome(lo=0.0, hi=1.0, use_second=1)
                gb_d, rb_d = rowwise(gb, dev), rowwise(rb, dev)
                n_d = rowwise(base_n, dev)
                s_d = rowwise(base_s, dev)
                per_seed[str(s)] = {
                    "ga_train_best": evo["per_seed"][str(s)][algo]["best"],
                    "ga_train_fit": evo["per_seed"][str(s)][algo]["fit_train"],
                    "ga_report_acc": round(float(np.mean([v for _, v in gb_d])), 4),
                    "random_train_best": evo["per_seed"][str(s)]["random"]["best"],
                    "random_report_acc": round(float(np.mean([v for _, v in rb_d])), 4),
                    "ga_minus_noulonly_report": paired_ci(n_d, gb_d),
                    "ga_minus_switchonly_report": paired_ci(s_d, gb_d),
                    "ga_minus_random_report": paired_ci(rb_d, gb_d)}
            accs = [per_seed[str(s)]["ga_report_acc"] for s in seeds]
            report["tasks"][f"{t}_{m}"] = {
                "evo_source": f"evo_train_{t}_{m}.json",
                "selection_metric": evo.get("run_manifest", {}).get("metric", "accuracy"),
                "per_seed": per_seed,
                "ga_report_mean": round(float(np.mean(accs)), 4),
                "ga_report_std": round(float(np.std(accs, ddof=1)), 4) if len(accs) > 1 else 0.0}
    for m in ("laya", "vela"):
        train = load(f"cache_train_math_{m}.json")
        dev = load(dev_file("math", m))
        assert_disjoint_pool(train, dev, dev_file("math", m))

        def ev(gg, _rows=train):
            f = evaluate_math(gg, _rows)
            return ((f["sel_accuracy"], f["coverage"]), gg.to_vector(), f)

        from run_nsga2 import cx_abstain as cx, mu_abstain as mu, sample_abstain as smp  # noqa: E402

        per_seed = {}
        for s in seeds:
            t = BudgetTracker(max_unique_evals=args.budget, max_model_calls=0)
            n = nsga2_run(ev, smp, cx, mu, budget=args.budget, seed=s,
                          population=8, generations=3, tracker=t,
                          ref=(0.0, 0.0))
            front = n["pareto"]
            cands = []
            for entry in front:
                gg = AbstainGenome(t=entry["genome"][0], use_margin=int(round(entry["genome"][1])),
                                   m=entry["genome"][2])
                f = evaluate_math(gg, train)
                cands.append((f["sel_accuracy"], f["coverage"], tuple(entry["genome"]), gg))
            cands.sort(key=lambda t_: (t_[0], t_[1], t_[2]), reverse=True)
            deploy = cands[0][3]
            fd = evaluate_math(deploy, dev)
            f0 = evaluate_math(AbstainGenome(t=0.0, use_margin=0, m=0.0), dev)
            th = {}
            for tt in (0.3, 0.5, 0.7):
                th[str(tt)] = evaluate_math(AbstainGenome(t=tt, use_margin=0, m=0.0), dev)
            per_seed[str(s)] = {"deploy": {"t": deploy.t, "use_margin": deploy.use_margin, "m": deploy.m},
                                "deploy_report": fd, "never_abstain_report": f0, "thresholds_report": th,
                                "hypervolume_train": n["hypervolume"]}
        report["tasks"][f"math_{m}"] = {"per_seed": per_seed}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for k, v in report["tasks"].items():
        if "ga_report_mean" in v:
            print(k, "GA-report:", v["ga_report_mean"], v["ga_report_std"],
                  {s: (d["ga_minus_switchonly_report"]["est"], d["ga_minus_switchonly_report"]["ci95"])
                   for s, d in v["per_seed"].items()})
        else:
            print(k, {s: d["deploy_report"] for s, d in v["per_seed"].items()})


if __name__ == "__main__":
    raise SystemExit(main())
