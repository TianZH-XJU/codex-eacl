#!/usr/bin/env python3
"""E-02: frozen ES-best evaluation on dev + confirm (offline).

ES search ran on train only. This script freezes the train-selected ES best
genome per seed (no re-search, no tuning) and reports it on selection_dev and
confirm with paired CIs vs the fixed switch-only baseline on the same pools.
Cost (calls/row) reported alongside quality. Zero new model calls.
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

TASKS = {
    "code_laya": ("results/v4/evolution/cache_train_code_laya.json",
                  "results/v4/evolution/cache_code_laya.json",
                  "results/v4/evolution/cache_confirm_code_laya.json"),
    "code_vela": ("results/v4/evolution/cache_train_code_vela.json",
                  "results/v4/evolution/cache_code_vela.json",
                  "results/v4/evolution/cache_confirm_code_vela.json"),
}


def load(p):
    d = json.loads((ROOT / p).read_text(encoding="utf-8"))
    return d["rows"], d.get("split", "unknown")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    try:
        sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                             check=True, capture_output=True, text=True).stdout.strip()
    except Exception:
        sha = None
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "protocol": ("v4.1 E-02: frozen train-selected ES best -> dev/confirm report; "
                           "paired CIs are accuracy diffs; point dicts carry both accuracy and macro_f1 "
                           "(train selection used macro_f1 per es_train manifest, dev/confirm computed with accuracy default)"),
              "paired_bootstrap": {"reps": 5000, "seed": 17, "rng": "test_evo_significance module-global RandomState (call-order dependent)"},
              "deploy_rule": "none (single frozen genome per seed; no selection on dev)",
              "run_manifest": {"argv": sys.argv[1:]},
              "git_sha": sha, "tasks": {}}
    for key, (tr_p, dev_p, conf_p) in TASKS.items():
        d = json.loads((ROOT / f"results/v4_1/evolution/es_train_{key}.json").read_text(encoding="utf-8"))
        train_rows, train_split = load(tr_p)
        dev_rows, dev_split = load(dev_p)
        conf_rows, conf_split = load(conf_p)
        assert_disjoint_pool(train_rows, dev_rows, f"{key} dev")
        assert_disjoint_pool(train_rows, conf_rows, f"{key} confirm")
        assert_disjoint_pool(dev_rows, conf_rows, f"{key} dev-vs-confirm")
        per_seed = {}
        for s in d["seeds"]:
            g = CascadeGenome(**d["per_seed"][str(s)]["es"]["best"])
            base = CascadeGenome(lo=0.0, hi=1.0, use_second=1)
            fd, fc = evaluate(g, dev_rows), evaluate(g, conf_rows)
            ci_dev = paired_ci(rowwise(base, dev_rows), rowwise(g, dev_rows))
            ci_conf = paired_ci(rowwise(base, conf_rows), rowwise(g, conf_rows))
            ci_dev["comparison"] = ci_conf["comparison"] = "es_minus_switchonly"
            per_seed[str(s)] = {"es_genome": d["per_seed"][str(s)]["es"]["best"],
                                "es_train_fit": d["per_seed"][str(s)]["es"]["fit_train"],
                                "es_dev": fd, "es_confirm": fc,
                                "es_minus_switch_dev": ci_dev,
                                "es_minus_switch_confirm": ci_conf,
                                "splits": {"train": train_split, "dev": dev_split, "confirm": conf_split}}
        report["tasks"][key] = {"per_seed": per_seed,
                                "candidate_source": f"es_train_{key}.json frozen train best"}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for k, v in report["tasks"].items():
        for s, ps in v["per_seed"].items():
            print(k, s, "dev:", ps["es_minus_switch_dev"]["est"], ps["es_minus_switch_dev"]["ci95"],
                  "conf:", ps["es_minus_switch_confirm"]["est"], ps["es_minus_switch_confirm"]["ci95"])


if __name__ == "__main__":
    raise SystemExit(main())
