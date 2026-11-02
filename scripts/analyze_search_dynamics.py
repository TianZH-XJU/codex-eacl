#!/usr/bin/env python3
"""P1 (supplement): search-dynamics diagnostics the old logs could not answer.

`search_landscape.json` marks train-dev rank correlation, neutral-mutation rate
and parent-child correlation as unavailable because per-genome dev scores were
never stored. This script regenerates the same GA runs deterministically from
the committed caches (no model calls), keeps the archive of every evaluated
genome, and measures:

  * duplicate rate              (tracker.generated vs unique evaluations)
  * train-dev Spearman rank correlation over the archive
  * neutral-mutation rate       (one-step mutations with unchanged train scalar)
  * parent-child fitness correlation
  * archive headroom            (best dev scalar inside the archive vs the dev
                                 scalar of the train-best genome)

The last row separates "the search never sampled a good policy" from "the
search sampled one but could not identify it without dev labels".

Output: results/v4_1/final/search_dynamics.json
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evolution import BudgetTracker, CascadeGenome, evaluate, ga_run, mutate  # noqa: E402

CELLS = {
    "code_laya": ("results/v4/evolution/cache_train_code_laya.json",
                  "results/v4/evolution/cache_code_laya.json"),
    "code_vela": ("results/v4/evolution/cache_train_code_vela.json",
                  "results/v4/evolution/cache_code_vela.json"),
    "findver_laya": ("results/v4/evolution/cache_train_findver_laya.json",
                     "results/v4/evolution/cache_findver_laya.json"),
    "findver_vela": ("results/v4/evolution/cache_train_findver_vela.json",
                     "results/v4/evolution/cache_findver_vela.json"),
}
SEEDS = (17, 23, 41)
BUDGET = 24
POP = 8
GENS = 3
MUT = 0.2
CROSS = 0.8
LAM = 0.05
N_MUT = 8


def load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))["rows"]


def spearman(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 3 or len(set(a.tolist())) < 2 or len(set(b.tolist())) < 2:
        return None

    def rank(x):
        order = np.argsort(x, kind="mergesort")
        r = np.empty(len(x), dtype=float)
        r[order] = np.arange(1, len(x) + 1, dtype=float)
        s = x[order]
        i = 0
        while i < len(s):
            j = i
            while j + 1 < len(s) and s[j + 1] == s[i]:
                j += 1
            if j > i:
                r[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
            i = j + 1
        return r

    ra, rb = rank(a), rank(b)
    ra -= ra.mean()
    rb -= rb.mean()
    den = float(np.sqrt(np.sum(ra ** 2) * np.sum(rb ** 2)))
    return round(float(np.sum(ra * rb) / den), 4) if den > 0 else None


def pearson(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return round(float(np.corrcoef(a, b)[0, 1]), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    try:
        sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                             check=True, capture_output=True, text=True).stdout.strip()
    except Exception:
        sha = None
    gens_eff = min(GENS, max(0, (BUDGET - POP) // (POP - 1)))
    report = {
        "experiment_id": "P1-DYNAMICS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repo_commit": sha,
        "protocol_version": "v4.1",
        "analysis_script": "analyze_search_dynamics.py",
        "analysis_type": "exploratory",
        "selection_used": False,
        "confirm_used": False,
        "pools": "search_train search with selection_dev used only as a diagnostic oracle",
        "params": {"budget_unique": BUDGET, "pop": POP, "gens_eff": gens_eff,
                   "mutation_rate": MUT, "crossover_rate": CROSS, "lam": LAM,
                   "metric": "macro_f1", "neutral_mutations_per_genome": N_MUT},
        "run_manifest": {"argv": sys.argv[1:]},
        "cells": {},
    }
    for key, (tr_p, dev_p) in CELLS.items():
        train, dev = load(tr_p), load(dev_p)
        per_seed = {}
        for seed in SEEDS:
            tracker = BudgetTracker(max_unique_evals=BUDGET, max_model_calls=0)
            res = ga_run(train, population=POP, generations=gens_eff,
                         crossover_rate=CROSS, mutation_rate=MUT, elitism=1,
                         seed=seed, tracker=tracker, lam=LAM, metric="macro_f1")
            arch = res["archive"]
            tr_sc = [a["scalar"] for a in arch]
            dv_sc, dv_acc, dv_mf = [], [], []
            for a in arch:
                g = CascadeGenome(**a["genome"])
                f = evaluate(g, dev, lam=LAM, primary="macro_f1")
                dv_sc.append(f["scalar"])
                dv_acc.append(f["accuracy"])
                dv_mf.append(f["macro_f1"])
            rng = random.Random(seed * 7919)
            deltas, parent_child = [], []
            for a in arch:
                g = CascadeGenome(**a["genome"])
                base = a["scalar"]
                for _ in range(N_MUT):
                    m = mutate(rng, g, MUT)
                    fm = evaluate(m, train, lam=LAM, primary="macro_f1")
                    deltas.append(abs(fm["scalar"] - base))
                    parent_child.append((base, fm["scalar"]))
            best_train_idx = int(np.argmax(tr_sc))
            best_dev_idx = int(np.argmax(dv_sc))
            state = tracker.state()
            per_seed[str(seed)] = {
                "n_archive": len(arch),
                "generated": state["generated"],
                "unique_evals": state["unique_evals"],
                "duplicate_hits": state["duplicate_hits"],
                "duplicate_rate": round(state["duplicate_hits"] / max(state["generated"], 1), 4),
                "spearman_train_dev": spearman(tr_sc, dv_sc),
                "pearson_train_dev": pearson(tr_sc, dv_sc),
                "neutral_mutation_rate": round(float(np.mean([d < 1e-9 for d in deltas])), 4),
                "n_mutations": len(deltas),
                "parent_child_pearson": pearson([p for p, _ in parent_child],
                                                [c for _, c in parent_child]),
                "best_train_scalar": round(float(tr_sc[best_train_idx]), 4),
                "best_train_dev_scalar": round(float(dv_sc[best_train_idx]), 4),
                "archive_best_dev_scalar": round(float(dv_sc[best_dev_idx]), 4),
                "archive_dev_headroom": round(float(dv_sc[best_dev_idx] - dv_sc[best_train_idx]), 4),
                "train_best_is_dev_best": bool(best_train_idx == best_dev_idx),
                "best_train_dev_accuracy": round(float(dv_acc[best_train_idx]), 4),
                "best_train_dev_macro_f1": round(float(dv_mf[best_train_idx]), 4),
            }
        report["cells"][key] = {"per_seed": per_seed,
                                "spearman_mean": round(float(np.mean(
                                    [v["spearman_train_dev"] for v in per_seed.values()
                                     if v["spearman_train_dev"] is not None])), 4),
                                "neutral_mean": round(float(np.mean(
                                    [v["neutral_mutation_rate"] for v in per_seed.values()])), 4)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for k, c in report["cells"].items():
        print(k, "spearman(mean)", c["spearman_mean"], "neutral(mean)", c["neutral_mean"],
              "| per-seed:", [(s, v["spearman_train_dev"], v["neutral_mutation_rate"],
                               v["archive_dev_headroom"]) for s, v in c["per_seed"].items()])


if __name__ == "__main__":
    raise SystemExit(main())
