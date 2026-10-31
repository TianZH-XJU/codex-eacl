#!/usr/bin/env python3
"""Offline figures from committed summaries (no model calls, no test data)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "results" / "final"
FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def main():
    rows = list(csv.DictReader((FINAL / "model_comparison.csv").open(encoding="utf-8")))
    labels = [f"{r['task']}\n({r['comparison'].split('(')[0].strip()[:28]})" for r in rows]
    ly = [float(r["laya_accuracy"]) for r in rows]
    ve = [float(r["vela_accuracy"]) for r in rows]
    x = range(len(rows))
    plt.figure(figsize=(10, 4.5))
    plt.bar([i - 0.2 for i in x], ly, width=0.4, label="Laya default")
    plt.bar([i + 0.2 for i in x], ve, width=0.4, label="Vela-2.0-0.3B default (Torch FP32)")
    plt.xticks(list(x), labels, fontsize=8)
    plt.ylabel("accuracy (paired rows)")
    plt.title("Paired default baselines: Laya vs Vela (identical inputs)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / "laya_vs_vela_accuracy.png", dpi=150)
    plt.close()

    sens = json.loads((ROOT / "results/processbench/sensitivity_pilot.summary.json").read_text(encoding="utf-8"))
    names = list(sens["configs"].keys())
    means = [sens["configs"][k]["mean_delta_correct"] for k in names]
    lo = [sens["configs"][k]["delta_ci95"][0] for k in names]
    hi = [sens["configs"][k]["delta_ci95"][1] for k in names]
    y = range(len(names))
    plt.figure(figsize=(8, 4))
    plt.errorbar(means, list(y), xerr=[[m - l for m, l in zip(means, lo)],
                                       [h - m for m, h in zip(means, hi)]], fmt="o", capsize=4)
    plt.axvline(0, color="k", linewidth=0.8)
    plt.yticks(list(y), names, fontsize=8)
    plt.xlabel("mean DeltaCorrect vs single (95% paired bootstrap CI, train n=100)")
    plt.title("Gate 4: config sensitivity (math Choice)")
    plt.tight_layout()
    plt.savefig(FIG / "sensitivity_delta.png", dpi=150)
    plt.close()
    print("wrote", sorted(p.name for p in FIG.glob("*.png")))


if __name__ == "__main__":
    raise SystemExit(main())
