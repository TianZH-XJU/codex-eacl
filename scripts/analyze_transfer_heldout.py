#!/usr/bin/env python3
"""S-05 / EXP-05: held-out transfer evaluation (offline).

In-sample transfer (transfer_matrix.json) evaluates train-selected genomes on
the SAME train pools. This script evaluates the FROZEN train-selected best
genomes on HELD-OUT target pools (selection_dev and confirm) with paired CIs
vs the target's fixed switch-only baseline. No re-search, no confirm tuning:
genomes are loaded frozen; any pool that influenced the genome would
invalidate the cell (asserted disjoint by construction: train->dev, train->confirm).
Zero new model calls.
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

from evolution import CascadeGenome, evaluate  # noqa: E402
from eval_trainselected_on_dev import assert_disjoint_pool  # noqa: E402
from test_evo_significance import paired_ci, rowwise  # noqa: E402

NSGA = ROOT / "results" / "v4_1" / "evolution"


def load(p):
    d = json.loads((ROOT / p).read_text(encoding="utf-8"))
    return d["rows"]


SOURCES = ["code_laya", "code_vela", "findver_laya", "findver_vela"]
TARGETS = {
    "code_laya": ("results/v4/evolution/cache_code_laya.json",
                  "results/v4/evolution/cache_confirm_code_laya.json"),
    "code_vela": ("results/v4/evolution/cache_code_vela.json",
                  "results/v4/evolution/cache_confirm_code_vela.json"),
    "findver_laya": ("results/v4/evolution/cache_findver_laya.json",
                     "results/v4/evolution/cache_confirm_findver_laya.json"),
    "findver_vela": ("results/v4/evolution/cache_findver_vela.json",
                     "results/v4/evolution/cache_confirm_findver_vela.json"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    try:
        sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                             check=True, capture_output=True, text=True).stdout.strip()
    except Exception:
        sha = None
    genomes = {}
    for st in SOURCES:
        d = json.loads((NSGA / f"evo_train_{st}.json").read_text(encoding="utf-8"))
        train_cache = d["train_cache"]["path"]
        train_rows = load(train_cache)
        genomes[st] = {"genomes": {s: CascadeGenome(**d["per_seed"][str(s)]["ga"]["best"])
                                   for s in d["seeds"]},
                       "train_cache": train_cache, "train_rows": train_rows}
    matrix = {}
    for st in SOURCES:
        for tt in TARGETS:
            dev_rows = load(TARGETS[tt][0])
            conf_rows = load(TARGETS[tt][1])
            assert_disjoint_pool(genomes[st]["train_rows"], dev_rows, f"{st}->{tt} dev")
            assert_disjoint_pool(genomes[st]["train_rows"], conf_rows, f"{st}->{tt} confirm")
            cell = {}
            for tag, rows in (("dev", dev_rows), ("confirm", conf_rows)):
                fits = [evaluate(g, rows) for g in genomes[st]["genomes"].values()]
                base = CascadeGenome(lo=0.0, hi=1.0, use_second=1)
                # paired CI of mean genome vs fixed switch (same rows)
                cis = {}
                for s, g in genomes[st]["genomes"].items():
                    c = paired_ci(rowwise(base, rows), rowwise(g, rows))
                    c["comparison"] = "transfer_minus_switch"
                    cis[str(s)] = c
                ests = [c["est"] for c in cis.values()]
                pool = {"dev": TARGETS[tt][0], "confirm": TARGETS[tt][1]}[tag]
                cell[tag] = {"n": len(rows),
                             "pool": pool,
                             "source": {"evo_file": f"results/v4_1/evolution/evo_train_{st}.json",
                                        "train_cache": genomes[st]["train_cache"]},
                             "mean_macro_f1": round(float(sum(f["macro_f1"] for f in fits) / len(fits)), 4),
                             "mean_acc": round(float(sum(f["accuracy"] for f in fits) / len(fits)), 4),
                             "vs_switch_paired": {
                                 "mean_est": round(float(sum(ests) / len(ests)), 4),
                                 "all_ns": all(not c["excludes_zero"] for c in cis.values()),
                                 "per_seed": cis}}
            matrix[f"{st}_on_{tt}"] = cell
    report = {"experiment_id": "EXP-05", "created_utc": datetime.now(timezone.utc).isoformat(),
              "repo_commit": sha, "protocol_version": "v4.1",
              "pools": "frozen train-selected genomes -> held-out target dev/confirm; no re-search, no confirm tuning",
              "matrix": matrix}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for k, v in matrix.items():
        for tag in ("dev", "confirm"):
            c = v[tag]
            print(k, tag, "macroF1:", c["mean_macro_f1"],
                  "vs-switch mean_est:", c["vs_switch_paired"]["mean_est"],
                  "all_ns:", c["vs_switch_paired"]["all_ns"])


if __name__ == "__main__":
    raise SystemExit(main())
