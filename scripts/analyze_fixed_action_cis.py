#!/usr/bin/env python3
"""Fixed-action comparison under the task primary metric (offline).

E1 decomposes repair/harm by per-row accuracy, but CodeXGLUE and FinDVer are
scored by macro-F1. This script reports the paired difference between the two
fixed actions (noul-only vs switch-only) in BOTH metrics with group-bootstrap
CIs on every pool, so the paper can state where accuracy and macro-F1 disagree
instead of silently mixing them.

Output: results/v4_1/final/fixed_action_cis.json
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
    noul = CascadeGenome(lo=0.0, hi=1.0, use_second=0)
    switch = CascadeGenome(lo=0.0, hi=1.0, use_second=1)
    report = {
        "experiment_id": "P0-FIXED-METRIC",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repo_commit": sha,
        "protocol_version": "v4.1",
        "analysis_script": "analyze_fixed_action_cis.py",
        "analysis_type": "confirmatory",
        "selection_used": False,
        "confirm_used": True,
        "pools": "no selection; fixed actions evaluated on train/dev/confirm",
        "ci_method": "paired group-bootstrap, 5000 resamples, module RandomState(17)",
        "comparison": "switch-only minus noul-only (positive favours the second action)",
        "run_manifest": {"argv": sys.argv[1:]},
        "cells": {},
    }
    for key, (tr_p, dev_p, conf_p) in CELLS.items():
        cell = {}
        for tag, path in (("train", tr_p), ("dev", dev_p), ("confirm", conf_p)):
            rows = load(path)
            mf = paired_macro_f1_ci(correctness(noul, rows), correctness(switch, rows), n=args.reps)
            acc = paired_ci(rowwise(noul, rows), rowwise(switch, rows), n=args.reps)
            cell[tag] = {"n": len(rows), "macro_f1": mf, "accuracy": acc}
        report["cells"][key] = cell
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for k, c in report["cells"].items():
        print(k,
              "train mf", c["train"]["macro_f1"]["est"], c["train"]["macro_f1"]["ci95"],
              "| dev mf", c["dev"]["macro_f1"]["est"], c["dev"]["macro_f1"]["ci95"],
              "| conf mf", c["confirm"]["macro_f1"]["est"], c["confirm"]["macro_f1"]["ci95"],
              "| conf acc", c["confirm"]["accuracy"]["est"], c["confirm"]["accuracy"]["ci95"])


if __name__ == "__main__":
    raise SystemExit(main())
