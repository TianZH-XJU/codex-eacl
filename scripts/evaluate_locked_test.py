#!/usr/bin/env python3
"""One-shot evaluation of the frozen v4.1 locked test.

Applies only the policies frozen in results/v4_1/locked_test/protocol.json:
fixed Noul, fixed switch-only, and the train-selected GA genomes for seeds
17/23/41.  Reports macro-F1, accuracy, calls and paired group-bootstrap CIs.
No policy, threshold or metric is selected from these results.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "results/v4_1/locked_test"
RNG = np.random.RandomState(17)
CELLS = {
    "code_laya": ("code", "laya"),
    "code_vela": ("code", "vela"),
    "findver_laya": ("findver", "laya"),
    "findver_vela": ("findver", "vela"),
}


def macro_f1(gold, pred):
    f1s = []
    for cls in (False, True):
        tp = float(np.sum((gold == cls) & (pred == cls)))
        fp = float(np.sum((gold != cls) & (pred == cls)))
        fn = float(np.sum((gold == cls) & (pred != cls)))
        denom = 2.0 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else 2.0 * tp / denom)
    return float(np.mean(f1s))


def choice_bool(row, task):
    v = str(row.get("switch_choice", "")).lower()
    if task == "code":
        return v == "vulnerable"
    return v == "supported"


def metrics(rows, pred, task):
    gold = np.array([bool(r["gold"]) for r in rows], dtype=bool)
    pred = np.array(pred, dtype=bool)
    return {"accuracy": round(float(np.mean(gold == pred)), 4),
            "macro_f1": round(macro_f1(gold, pred), 4)}


def ga_pred(rows, task, g):
    out = []
    calls = 0
    switched = 0
    for r in rows:
        p = float(r["noul_p"])
        sw = (int(g["use_second"]) == 1) and (float(g["lo"]) <= p <= float(g["hi"]))
        noul = p >= 0.5
        out.append(choice_bool(r, task) if sw else noul)
        calls += 1 + int(sw)
        switched += int(sw)
    return out, calls, switched


def bootstrap_delta(rows, task, pred_a, pred_b, metric="macro_f1", n_boot=5000):
    by_group = {}
    for i, r in enumerate(rows):
        by_group.setdefault(r["group_id"], []).append(i)
    groups = sorted(by_group)
    diffs = []
    for _ in range(n_boot):
        sample = RNG.choice(groups, size=len(groups), replace=True)
        idx = [i for g in sample for i in by_group[g]]
        gold = np.array([bool(rows[i]["gold"]) for i in idx], dtype=bool)
        if len(set(gold.tolist())) < 2:
            continue
        a = np.array([pred_a[i] for i in idx], dtype=bool)
        b = np.array([pred_b[i] for i in idx], dtype=bool)
        if metric == "macro_f1":
            diffs.append(macro_f1(gold, a) - macro_f1(gold, b))
        else:
            diffs.append(float(np.mean(gold == a) - np.mean(gold == b)))
    est = (macro_f1(np.array([bool(r["gold"]) for r in rows], dtype=bool), np.array(pred_a, dtype=bool))
           - macro_f1(np.array([bool(r["gold"]) for r in rows], dtype=bool), np.array(pred_b, dtype=bool))) \
        if metric == "macro_f1" else float(np.mean(np.array([bool(r["gold"]) for r in rows], dtype=bool)
                                                   == np.array(pred_a, dtype=bool))
                                             - np.mean(np.array([bool(r["gold"]) for r in rows], dtype=bool)
                                                       == np.array(pred_b, dtype=bool)))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"est": round(float(est), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "n_boot": len(diffs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    protocol = json.loads((LOCK / "protocol.json").read_text(encoding="utf-8"))
    cells = {}
    for cell, (task, model) in CELLS.items():
        cache = json.loads((LOCK / f"cache_{cell}.json").read_text(encoding="utf-8"))
        rows = cache["rows"]
        noul_pred = [float(r["noul_p"]) >= 0.5 for r in rows]
        switch_pred = [choice_bool(r, task) for r in rows]
        out = {
            "n": len(rows), "n_groups": len({r["group_id"] for r in rows}),
            "fixed_noul": {**metrics(rows, noul_pred, task), "calls_per_row": 1.0},
            "fixed_switch": {**metrics(rows, switch_pred, task), "calls_per_row": 2.0},
            "ga": {},
        }
        for seed, g in protocol["genomes"][cell].items():
            pred, calls, switched = ga_pred(rows, task, g)
            m = metrics(rows, pred, task)
            out["ga"][seed] = {
                "genome": g, **m,
                "calls_per_row": round(calls / len(rows), 3),
                "switch_rate": round(switched / len(rows), 3),
                "delta_macro_f1_vs_switch": bootstrap_delta(rows, task, pred, switch_pred, "macro_f1"),
                "delta_macro_f1_vs_noul": bootstrap_delta(rows, task, pred, noul_pred, "macro_f1"),
                "delta_accuracy_vs_noul": bootstrap_delta(rows, task, pred, noul_pred, "accuracy"),
            }
        cells[cell] = out
    report = {"created_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
              "protocol_sha256": protocol["protocol_sha256"],
              "analysis_script": "evaluate_locked_test.py",
              "status": "one_shot_locked_test",
              "cells": cells}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for cell, c in cells.items():
        print(cell, "n", c["n"], "noul", c["fixed_noul"], "switch", c["fixed_switch"])
        for seed, g in c["ga"].items():
            print(" GA", seed, g["accuracy"], g["macro_f1"], "calls", g["calls_per_row"],
                  "dF1-switch", g["delta_macro_f1_vs_switch"])


if __name__ == "__main__":
    raise SystemExit(main())
