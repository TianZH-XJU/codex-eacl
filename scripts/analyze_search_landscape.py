#!/usr/bin/env python3
"""S-03 / EXP-03: search-landscape diagnostics from stored logs (offline).

No re-search: reads GA/ES/NSGA-II train result files and summarizes what the
stored logs already record — unique/generated/duplicate rates, best-so-far
trajectories, diversity collapse, mean-vs-best gaps, budget utilization.
Unanswerable-from-logs items (parent-child correlation, train-dev rank
correlation: no per-genome dev scores were stored) are marked unavailable
with the exact rerun that would be needed, not fabricated.
"""
from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EV = ROOT / "results/v4_1/evolution"


def best_so_far(log, key="best_scalar"):
    """Cumulative max of a per-generation log series + plateau generation."""
    if not log:
        raise ValueError("best_so_far: empty log (fail-closed, no silent series)")
    vals = [g[key] for g in log]
    best, plat, series = vals[0], 0, []
    for i, v in enumerate(vals):
        if v > best:
            best, plat = v, i
        series.append(round(best, 4))
    return {"first": vals[0], "last": vals[-1], "plateau_gen": plat,
            "n_gen_logs": len(vals), "series": series}


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    try:
        sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                             check=True, capture_output=True, text=True).stdout.strip()
    except Exception:
        sha = None
    fam = {}
    for f in sorted(glob.glob(str(EV / "evo_train_*.json"))):
        d = load(f)
        algo = d.get("algo", "ga")
        seeds = {}
        for s in d["seeds"]:
            e = d["per_seed"][str(s)]
            g, r = e[algo], e["random"]
            log = g.get("log", g.get("generations", []))
            div = [x.get("diversity") for x in log if x.get("diversity") is not None]
            seeds[str(s)] = {
                algo: {"unique": g.get("unique", g.get("n_unique_evals")),
                       "generated": (g.get("budget_state") or {}).get("generated"),
                       "duplicates": (g.get("budget_state") or {}).get("duplicate_hits"),
                       "best_so_far": best_so_far(log) if log else None,
                       "diversity_first": div[0] if div else None,
                       "diversity_last": div[-1] if div else None,
                       "mean_last": log[-1].get("mean_scalar") if log else None},
                "random": {"unique": r.get("unique", r.get("n_unique_evals")),
                           "note": "no per-gen history stored for random: endpoint only"},
                "gap_minus_random": d.get(f"{algo}_minus_random_mean")}
        fam[Path(f).stem] = {"algo": algo, "budget_unique": d["budget_unique"], "per_seed": seeds}
    for f in sorted(glob.glob(str(EV / "es_train_*.json"))):
        d = load(f)
        seeds = {}
        for s in d["seeds"]:
            e = d["per_seed"][str(s)]["es"]
            log = e.get("log", [])
            div = [x.get("diversity") for x in log if x.get("diversity") is not None]
            seeds[str(s)] = {"unique": e.get("unique", e.get("n_unique_evals")),
                             "best_so_far": best_so_far(log) if log else None,
                             "diversity_first": div[0] if div else None,
                             "diversity_last": div[-1] if div else None}
        fam[Path(f).stem] = {"algo": "es", "budget_unique": d["budget_unique"], "per_seed": seeds}
    for f in sorted(glob.glob(str(EV / "nsga2_train_*.json"))):
        d = load(f)
        seeds = {}
        for s in d["seeds"]:
            e = d["per_seed"][str(s)]
            seeds[str(s)] = {"unique": e.get("nsga2_unique"),
                             "stopped_early": e.get("nsga2_stopped_early"),
                             "hypervolume": e.get("nsga2_hv"),
                             "front_size": e.get("nsga2_front"),
                             "random_hv": e.get("random_hv"),
                             "note": ("endpoint only: per-generation HV/front trajectories were not "
                                      "stored by run_nsga2.py; see unavailable_from_logs")}
        fam[Path(f).stem] = {"algo": "nsga2", "budget_unique": d["budget_unique"],
                             "objectives": "quality-vs-cost HV", "per_seed": seeds}
    report = {"experiment_id": "EXP-03", "created_utc": datetime.now(timezone.utc).isoformat(),
              "repo_commit": sha, "protocol_version": "v4.1",
              "analysis_script": "analyze_search_landscape.py",
              "analysis_type": "exploratory", "selection_used": False, "confirm_used": False,
              "pools": "search_train logs only; no new search",
              "run_manifest": {"argv": sys.argv[1:]},
              "unavailable_from_logs": {
                  "parent_child_correlation": "per-child fitness not stored; needs score-level logging rerun",
                  "neutral_mutation_rate": "no per-mutation fitness deltas stored; same rerun as above",
                  "train_dev_rank_correlation": "no per-genome dev scores stored (only best reported); needs archive-eval rerun",
                  "random_trajectories": "random_search stores endpoint only, no history",
                  "nsga2_per_generation_trajectories": "nsga2_train files store endpoint HV/front only; per-generation front logs need a run_nsga2.py logging change + rerun"},
              "families": fam}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for k, v in fam.items():
        for s, ps in v["per_seed"].items():
            if v["algo"] == "nsga2":
                print(k, s, "unique:", ps.get("unique"), "hv:", ps.get("hypervolume"),
                      "front:", ps.get("front_size"), "stopped:", ps.get("stopped_early"))
                continue
            g = ps.get(v["algo"], ps)
            bsf = (g.get("best_so_far") or {})
            print(k, s, "u/g:", g.get("unique"), g.get("generated"),
                  "plateau:", bsf.get("plateau_gen"), "div:", g.get("diversity_first"), "->", g.get("diversity_last"))


if __name__ == "__main__":
    raise SystemExit(main())
