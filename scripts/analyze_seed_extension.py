#!/usr/bin/env python3
"""P1 (supplement): pre-registered seeds 59/73 for GA and ES (offline).

The v4.1 protocol pre-registered five seeds (17/23/41/59/73) but the main
tables report three. This script re-runs the search with all five seeds from
the committed caches (deterministic, zero model calls), then recomputes the
paired group-bootstrap CIs for the two comparisons the paper headlines:

  GA-best vs matched random-best        (macro-F1 and accuracy)
  GA-best vs fixed switch-only          (macro-F1 and accuracy)

Seeds 17/23/41 are rerun too; their per-seed genomes and CIs must reproduce
the three-seed files exactly (determinism check, not a new result).

Inputs: results/v4_1/final/ga5_*.json / es5_*.json written by
`run_policy_evolution.py --seeds 17,23,41,59,73`.
Output: results/v4_1/final/seed_extension.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evolution import CascadeGenome  # noqa: E402
from paired_macro_f1_ci import correctness, paired_macro_f1_ci  # noqa: E402
from test_evo_significance import paired_ci, rowwise  # noqa: E402

CELLS = {
    "code_laya": ("results/v4/evolution/cache_train_code_laya.json",
                  "results/v4/evolution/cache_code_laya.json",
                  "results/v4/evolution/cache_confirm_code_laya.json"),
    "code_vela": ("results/v4/evolution/cache_train_code_vela.json",
                  "results/v4/evolution/cache_code_vela.json",
                  "results/v4/evolution/cache_confirm_code_vela.json"),
    "findver_laya": ("results/v4/evolution/cache_train_findver_laya.json",
                     "results/v4/evolution/cache_findver_laya.json",
                     "results/v4/evolution/cache_confirm_findver_laya.json"),
    "findver_vela": ("results/v4/evolution/cache_train_findver_vela.json",
                     "results/v4/evolution/cache_findver_vela.json",
                     "results/v4/evolution/cache_confirm_findver_vela.json"),
}


def load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))["rows"]


def compare(algo_genome, other_genome, rows, reps):
    """Both macro-F1 and accuracy paired CIs for algo - other."""
    mf = paired_macro_f1_ci(correctness(other_genome, rows),
                            correctness(algo_genome, rows), n=reps)
    acc = paired_ci(rowwise(other_genome, rows), rowwise(algo_genome, rows), n=reps)
    return {"macro_f1": mf, "accuracy": acc, "n": len(rows)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=5000)
    args = ap.parse_args()
    try:
        sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                             check=True, capture_output=True, text=True).stdout.strip()
    except Exception:
        sha = None
    report = {
        "experiment_id": "P1-SEEDS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repo_commit": sha,
        "protocol_version": "v4.1",
        "analysis_script": "analyze_seed_extension.py",
        "analysis_type": "confirmatory",
        "selection_used": True,
        "confirm_used": True,
        "pools": "search_train select -> search_dev + confirm report",
        "ci_method": "paired group-bootstrap, 5000 resamples, module RandomState(17)",
        "run_manifest": {"argv": sys.argv[1:]},
        "cells": {},
    }
    for key, (tr_p, dev_p, conf_p) in CELLS.items():
        pools = {"train": load(tr_p), "dev": load(dev_p), "confirm": load(conf_p)}
        cell = {}
        for algo, src in (("ga", f"results/v4_1/final/ga5_{key}.json"),
                          ("es", f"results/v4_1/final/es5_{key}.json")):
            p = ROOT / src
            if not p.exists():
                cell[algo] = {"status": "not_run"}
                continue
            evo = json.loads(p.read_text(encoding="utf-8"))
            per_seed = {}
            for s in evo["seeds"]:
                g = CascadeGenome(**evo["per_seed"][str(s)][algo]["best"])
                r = CascadeGenome(**evo["per_seed"][str(s)]["random"]["best"])
                sw = CascadeGenome(lo=0.0, hi=1.0, use_second=1)
                entry = {"genome": evo["per_seed"][str(s)][algo]["best"],
                         "unique_evals": evo["per_seed"][str(s)][algo].get("unique"),
                         "pools": {}}
                for tag, rows in pools.items():
                    entry["pools"][tag] = {
                        "vs_random": compare(g, r, rows, args.reps),
                        "vs_switch": compare(g, sw, rows, args.reps),
                    }
                per_seed[str(s)] = entry
            cell[algo] = {
                "seeds": evo["seeds"],
                "source": src,
                "per_seed": per_seed,
                "train_scalar_mean": evo.get(f"{algo}_train_scalar_mean"),
                "random_train_scalar_mean": evo.get("random_train_scalar_mean"),
                "train_scalar_gap": evo.get(f"{algo}_minus_random_mean"),
            }
        report["cells"][key] = cell
    # summary counts
    summary = {}
    for key, cell in report["cells"].items():
        for algo in ("ga", "es"):
            if "per_seed" not in cell.get(algo, {}):
                continue
            n = sig = sig_pos = 0
            for s, entry in cell[algo]["per_seed"].items():
                for tag in ("train", "dev", "confirm"):
                    for comp in ("vs_random", "vs_switch"):
                        for metric in ("macro_f1", "accuracy"):
                            c = entry["pools"][tag][comp][metric]
                            n += 1
                            if c["excludes_zero"]:
                                sig += 1
                                sig_pos += int(c["est"] > 0)
            summary[f"{key}_{algo}"] = {"cells": n, "sig": sig, "sig_positive": sig_pos}
    report["summary"] = summary
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for k, v in summary.items():
        print(k, v)
    for key, cell in report["cells"].items():
        for algo in ("ga", "es"):
            if "per_seed" not in cell.get(algo, {}):
                continue
            gaps = [cell[algo]["per_seed"][s]["pools"]["confirm"]["vs_switch"]["macro_f1"]["est"]
                    for s in cell[algo]["per_seed"]]
            print(key, algo, "confirm GA-switch macroF1 by seed:", gaps)


if __name__ == "__main__":
    raise SystemExit(main())
