#!/usr/bin/env python3
"""C-14: NSGA-II deployment-point selection on selection_dev (offline).

The old deploy result selected the front member on the TRAIN pool (the same
pool the Pareto front was searched on) — an in-sample deployment claim.
This script implements the protocol-fixed flow:

1. Load the FROZEN train Pareto candidate sets from nsga2_train_*.json
   (no re-search; genomes + train objectives recorded as candidates).
2. Select exactly one deployment point per task/model/seed on SELECTION_DEV
   with a pre-registered rule (binary: max dev accuracy, tie-break fewer
   calls, then genome order; math: max dev sel_accuracy, tie-break coverage,
   then genome order).
3. Evaluate the frozen point on DEV and on CONFIRM (paired CIs vs the
   strongest fixed baseline on the same pool).
4. Confirm is report-only: nothing is re-selected on confirm.

Search/deploys use accuracy-HV fronts (cascade) as run; macro-F1 is reported
alongside (task primary) but does not change the pre-registered rule.
Zero model calls.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evolution import AbstainGenome, CascadeGenome, evaluate_math  # noqa: E402
from evolution import evaluate as cascade_evaluate  # noqa: E402
from eval_trainselected_on_dev import assert_disjoint_pool  # noqa: E402
from test_evo_significance import paired_ci, rowwise, rowwise_math  # noqa: E402

EV = ROOT / "results" / "v4" / "evolution"
NSGA = ROOT / "results" / "v4_1" / "evolution"

TASKS = {
    "code_laya": ("cascade", "results/v4/evolution/cache_code_laya.json",
                  "results/v4/evolution/cache_confirm_code_laya.json"),
    "code_vela": ("cascade", "results/v4/evolution/cache_code_vela.json",
                  "results/v4/evolution/cache_confirm_code_vela.json"),
    "math_laya": ("abstain", "results/v4/evolution/cache_math_laya.json",
                  "results/v4/evolution/cache_confirm_math_laya.json"),
    "math_vela": ("abstain", "results/v4/evolution/cache_math_vela.json",
                  "results/v4/evolution/cache_confirm_math_vela.json"),
}


def load(p):
    d = json.loads((ROOT / p).read_text(encoding="utf-8"))
    return d["rows"], d.get("split", "unknown")


def to_genome(kind, vec):
    if kind == "cascade":
        return CascadeGenome(lo=vec[0], hi=vec[1], use_second=int(round(vec[2])))
    return AbstainGenome(t=vec[0], use_margin=int(round(vec[1])), m=vec[2])


def select_binary(cands):
    """Pre-registered deploy rule (cascade): max dev accuracy, tie-break
    fewer calls, then genome order. Pure: unit-testable."""
    return sorted(cands, key=lambda t: (t[0], t[1], t[3]), reverse=True)[0][-1]


def select_abstain(cands):
    """Pre-registered deploy rule (abstain): max dev sel_accuracy, tie-break
    coverage, then genome order. Pure: unit-testable."""
    return sorted(cands, key=lambda t: (t[0], t[1], t[2]), reverse=True)[0][-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default=",".join(TASKS),
                    help="comma subset of: " + ",".join(TASKS))
    args = ap.parse_args()
    wanted = [t.strip() for t in args.tasks.split(",") if t.strip()]
    assert set(wanted) <= set(TASKS), f"unknown tasks: {wanted}"

    try:
        sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                             check=True, capture_output=True, text=True).stdout.strip()
    except Exception:
        sha = None
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "protocol": "v4.1 C-14: frozen train pareto -> dev select -> confirm report",
              "deploy_rule": {"cascade": "max dev accuracy, tie-break fewer calls, then genome order",
                              "abstain": "max dev sel_accuracy, tie-break coverage, then genome order"},
              "tasks": {}, "git_sha": sha,
              "note": ("search optimized accuracy-HV (cascade) as run; macro-F1 reported "
                       "alongside; confirm is report-only, never re-selected")}

    for key, (kind, dev_path, conf_path) in TASKS.items():
        if key not in wanted:
            continue
        ns = json.loads((NSGA / f"nsga2_train_{key}.json").read_text(encoding="utf-8"))
        train_cache = ns.get("cache") or (ns.get("train_cache") or {}).get("path")
        train_rows = json.loads((ROOT / train_cache).read_text(encoding="utf-8"))["rows"]
        dev_rows, dev_split = load(dev_path)
        conf_rows, conf_split = load(conf_path)
        assert_disjoint_pool(train_rows, dev_rows, f"{key} dev")
        assert_disjoint_pool(train_rows, conf_rows, f"{key} confirm")
        assert_disjoint_pool(dev_rows, conf_rows, f"{key} dev-vs-confirm")
        per_seed = {}
        for s in ns["seeds"]:
            cands = []
            for entry in ns["per_seed"][str(s)]["pareto"]:
                gg = to_genome(kind, entry["genome"])
                if kind == "cascade":
                    f = cascade_evaluate(gg, dev_rows)
                    cands.append((f["accuracy"], -f["calls_per_row"], f["macro_f1"],
                                  tuple(entry["genome"]), gg))
                else:
                    f = evaluate_math(gg, dev_rows)
                    cands.append((f["sel_accuracy"], f["coverage"],
                                  tuple(entry["genome"]), gg))
            deploy = (select_binary(cands) if kind == "cascade"
                      else select_abstain(cands))
            deploy_dev = (cascade_evaluate(deploy, dev_rows) if kind == "cascade"
                          else evaluate_math(deploy, dev_rows))
            deploy_conf = (cascade_evaluate(deploy, conf_rows) if kind == "cascade"
                           else evaluate_math(deploy, conf_rows))
            if kind == "cascade":
                base_dev = CascadeGenome(lo=0.0, hi=1.0, use_second=1)
                base_conf = CascadeGenome(lo=0.0, hi=1.0, use_second=1)
                ci_dev = paired_ci(rowwise(base_dev, dev_rows), rowwise(deploy, dev_rows))
                ci_conf = paired_ci(rowwise(base_conf, conf_rows), rowwise(deploy, conf_rows))
                ci_dev["comparison"] = ci_conf["comparison"] = "deploy_minus_switchonly"
            else:
                base = AbstainGenome(t=0.0, use_margin=0, m=0.0)
                ci_dev = paired_ci(rowwise_math(base, dev_rows), rowwise_math(deploy, dev_rows))
                ci_conf = paired_ci(rowwise_math(base, conf_rows), rowwise_math(deploy, conf_rows))
                ci_dev["comparison"] = ci_conf["comparison"] = "deploy_minus_never"
            per_seed[str(s)] = {
                "n_candidates": len(cands),
                "candidate_genomes": [list(t[-2]) for t in cands],
                "candidate_train_objectives": [list(e["objectives"])
                                               for e in ns["per_seed"][str(s)]["pareto"]],
                "deploy_genome": list(deploy.to_vector()),
                "deploy_dev": deploy_dev, "deploy_confirm": deploy_conf,
                "deploy_minus_base_dev": ci_dev, "deploy_minus_base_confirm": ci_conf,
                "splits": {"train": "search_train", "dev": dev_split, "confirm": conf_split}}
        report["tasks"][key] = {"per_seed": per_seed,
                                "candidate_source": f"nsga2_train_{key}.json frozen train pareto"}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for k, v in report["tasks"].items():
        for s, ps in v["per_seed"].items():
            print(k, s, "dev:", ps["deploy_minus_base_dev"]["est"], ps["deploy_minus_base_dev"]["ci95"],
                  "conf:", ps["deploy_minus_base_confirm"]["est"], ps["deploy_minus_base_confirm"]["ci95"])


if __name__ == "__main__":
    raise SystemExit(main())
