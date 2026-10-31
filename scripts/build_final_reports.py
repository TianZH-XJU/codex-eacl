#!/usr/bin/env python3
"""Build Stage 10 final summary artifacts (offline, from committed results only).

Writes results/final/model_comparison.{json,csv}, metrics_by_task.json,
paired_bootstrap.json, resource_profile.json, error_analysis.md.
Search-curve/pareto CSVs record not_run: no ES stage was reached (all gates failed).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FINAL = RES / "final"


def load(p):
    return json.loads((RES / p).read_text(encoding="utf-8"))


def paired_bootstrap(a, b, reps=5000, seed=17):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    diff = b - a
    rng = np.random.RandomState(seed)
    means = [float(np.mean(rng.choice(diff, size=len(diff), replace=True))) for _ in range(reps)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"diff_mean": round(float(diff.mean()), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "n": len(diff), "reps": reps, "seed": seed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/final")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    comparisons = [
        ("processbench", "Laya single vs Vela choice (train pilot, n=120)",
         "processbench/pilot_train_cache.jsonl", "vela_baseline_processbench_train.json", "single", "choice"),
        ("code_defect", "Laya single(noul) vs Vela noul (valid pilot, n=60)",
         "codexglue_defect/pilot_valid_cache.jsonl", "vela_baseline_code_valid.json", "single", "noul"),
        ("code_defect", "Laya switch vs Vela switch (valid pilot, n=60)",
         "codexglue_defect/pilot_valid_cache.jsonl", "vela_baseline_code_valid.json", "switch", "switch"),
        ("findver", "Laya single(noul) vs Vela noul (dev pilot, n=60)",
         "findver/pilot_dev_cache.jsonl", "vela_baseline_findver_dev.json", "single", "noul"),
        ("findver", "Laya switch vs Vela switch (dev pilot, n=60)",
         "findver/pilot_dev_cache.jsonl", "vela_baseline_findver_dev.json", "switch", "switch"),
    ]
    rows, boot = [], {}
    for task, label, laya_cache, vela_file, lstrat, vrep in comparisons:
        lrows = [json.loads(x) for x in (RES / laya_cache).read_text(encoding="utf-8").splitlines() if x.strip()]
        lby = {}
        for r in lrows:
            if r["strategy"] == lstrat:
                lby[(r.get("split"), r.get("position"))] = int(r["correct"])
        vrows = json.loads((RES / vela_file).read_text(encoding="utf-8"))["results"]
        vby = {(r["split"], r["position"]): int(r["correct"]) for r in vrows if r["rep"] == vrep}
        keys = sorted(set(lby) & set(vby))
        la = [lby[k] for k in keys]
        va = [vby[k] for k in keys]
        pb = paired_bootstrap(la, va)
        rows.append({"task": task, "comparison": label, "n_paired": len(keys),
                     "laya_accuracy": round(sum(la) / len(la), 4), "vela_accuracy": round(sum(va) / len(va), 4),
                     **{k: v for k, v in pb.items() if k != "reps"}})
        boot[label] = {"paired_bootstrap_vela_minus_laya": pb}

    with (out / "model_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["task", "comparison", "n_paired", "laya_accuracy", "vela_accuracy",
                                          "diff_mean", "ci95_lo", "ci95_hi"])
        w.writeheader()
        for r in rows:
            w.writerow({"task": r["task"], "comparison": r["comparison"], "n_paired": r["n_paired"],
                        "laya_accuracy": r["laya_accuracy"], "vela_accuracy": r["vela_accuracy"],
                        "diff_mean": r["diff_mean"], "ci95_lo": r["ci95"][0], "ci95_hi": r["ci95"][1]})
    (out / "model_comparison.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    (out / "paired_bootstrap.json").write_text(json.dumps(boot, indent=2) + "\n", encoding="utf-8")

    with (out / "algorithm_search_curves.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["algorithm", "status", "note"])
        w.writeheader()
        for a in ("cmaes", "de", "ga", "nsga2", "tpe", "random", "neat-trackB"):
            w.writerow({"algorithm": a, "status": "not_run",
                        "note": "blocked: Gate 4 (config sensitivity) failed; no stable quality signal"})

    with (out / "pareto_frontier.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["task", "policy", "accuracy", "mean_latency_ms", "status"])
        w.writeheader()
        for r in rows:
            w.writerow({"task": r["task"], "policy": "laya-fixed", "accuracy": r["laya_accuracy"],
                        "mean_latency_ms": "", "status": "measured-fixed-config-only"})
            w.writerow({"task": r["task"], "policy": "vela-fixed", "accuracy": r["vela_accuracy"],
                        "mean_latency_ms": "", "status": "measured-fixed-config-only"})

    sens = {}
    for name in ("sensitivity_pilot.summary.json",):
        p = RES / "processbench" / name
        if p.exists():
            sens = json.loads(p.read_text(encoding="utf-8"))
    (out / "metrics_by_task.json").write_text(json.dumps({
        "processbench_sensitivity_train": {k: {kk: v[kk] for kk in ("accuracy", "mean_delta_correct", "delta_ci95", "flips_vs_single", "mean_cumulative_ms", "errors")} for k, v in sens.get("configs", {}).items()},
        "gate4_note": "train pilot n=100; dev replication pending in sensitivity_devpilot.summary.json",
    }, indent=2) + "\n", encoding="utf-8")

    stairs = {}
    for size in (256, 512, 1024, 2048, 4096, 8192):
        p = RES / f"vela_staircase_{size}.json"
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            stairs[str(size)] = {"peak_rss_mb": d["peak_rss_mb"],
                                 "latency_ms": d["results"][0]["latency_ms"] if d["results"] else None}
    (out / "resource_profile.json").write_text(json.dumps({
        "hardware_policy": "CPU-only, 2 threads, single model process, RSS hard stop 3.5GB",
        "vela_torch_fp32_staircase_tokens": stairs,
        "note": "all Vela runs stayed under 2.0GB RSS; latency grows super-linearly with input tokens",
    }, indent=2) + "\n", encoding="utf-8")

    (out / "error_analysis.md").write_text(
        "# Error analysis (all gates)\n\n## ProcessBench Gate 1 (math Choice)\n"
        "- Tournament correct set identical to single; wide net-harmful; oracle +0.017.\n"
        "- SINGLE predictions collapse to early steps (113/120 on steps 0-5, zero sentinel).\n"
        "## Code Gate 1\n- Upgrades only reshuffle errors (each fixes 1, breaks 15); LR OOF below majority.\n"
        "## FinDVer Gate 1\n- Switch +0.05 raw over single but unpredictable (LR OOF below majority).\n"
        "## Gate 4 (config sensitivity, math)\n- Only significant effect: wide_max1024 HURTS (delta CI below zero).\n"
        "- order_reverse flips 18 but net delta CI covers zero. No repeatable quality gain.\n"
        "## Laya vs Vela baselines\n- Both models near/below chance on math Choice; both ~chance on code; "
        "Laya ahead on finance but weak. No credible model gap claim.\n", encoding="utf-8")
    print("wrote final artifacts to", out)


if __name__ == "__main__":
    raise SystemExit(main())
