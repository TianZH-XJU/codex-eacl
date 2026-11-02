#!/usr/bin/env python3
"""P0 (supplement): task-level calibration of the Noul probability (offline).

Question the paper must answer before blaming "utility prediction": is the
Noul probability merely miscalibrated, or is it calibrated and still unable to
rank the incremental value of the switch action?

Protocol (zero new model calls, caches only):
- calibration target: P(gold = True) from the cached `noul_p`
  (Brier, NLL, ECE-15 equal width, adaptive ECE-10 equal mass, AUROC).
- calibration slope/intercept: logistic regression of gold on logit(p), fit on
  search_train, applied unchanged to selection_dev / confirm.
- temperature scaling: T fit on search_train by NLL (deterministic 1-D search),
  applied unchanged to dev/confirm; also group-aware 3-fold cross-fitted T
  inside train, so the in-train improvement is not an in-sample artefact.
- routing check: the paper's band policy class {lo <= p <= hi} is selected on
  dev and frozen on confirm twice, once with raw p and once with T-scaled p.
  Because T-scaling is strictly monotone on (0,1), the policy class is
  invariant; the experiment verifies this numerically instead of asserting it.
- utility contrast: AUROC of p for the *repair* event (switch fixes a noul
  error) vs AUROC of p for *gold*. A calibrated probability can rank its own
  correctness while carrying no information about the other action.

Output: one JSON with per-cell metrics + provenance. Never tunes on confirm.
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

from evolution import CascadeGenome, evaluate  # noqa: E402

LAM = 0.05
EPS = 1e-6

TASKS = {
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

BAND_GRID = [round(i / 20, 2) for i in range(21)]


def load(path):
    d = json.loads((ROOT / path).read_text(encoding="utf-8"))
    return d["rows"], d.get("split", "unknown")


def probs(rows):
    return np.array([float(r["noul_p"]) for r in rows], dtype=float)


def golds(rows):
    return np.array([1.0 if r["gold"] else 0.0 for r in rows], dtype=float)


def repair_labels(rows):
    """Y=1 iff the switch repairs a noul error (needs gold; labels only)."""
    y = []
    for r in rows:
        p = float(r["noul_p"])
        noul_ok = (p >= 0.5) == bool(r["gold"])
        sw_ok = bool(r.get("switch_correct", False))
        y.append(1.0 if (not noul_ok and sw_ok) else 0.0)
    return np.array(y, dtype=float)


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    z = np.clip(np.asarray(z, dtype=float), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))


def nll(p, y):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(p, y):
    return float(np.mean((np.asarray(p, dtype=float) - y) ** 2))


def ece_equal_width(p, y, bins=15):
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p < hi) if i < bins - 1 else (p >= lo) & (p <= hi)
        if not np.any(m):
            continue
        total += float(np.sum(m)) * abs(float(np.mean(y[m])) - float(np.mean(p[m])))
    return round(total / len(p), 4)


def ece_equal_mass(p, y, bins=10):
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(p)
    chunks = np.array_split(order, bins)
    total = 0.0
    for idx in chunks:
        if len(idx) == 0:
            continue
        total += len(idx) * abs(float(np.mean(y[idx])) - float(np.mean(p[idx])))
    return round(total / len(p), 4)


def auroc(score, y):
    """Rank-based AUROC; returns 0.5 when one class is absent."""
    score = np.asarray(score, dtype=float)
    y = np.asarray(y, dtype=float)
    n_pos = float(np.sum(y == 1))
    n_neg = float(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    ranks[order] = np.arange(1, len(score) + 1, dtype=float)
    # average ties
    s_sorted = score[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            avg = (i + 1 + j + 1) / 2.0
            ranks[order[i:j + 1]] = avg
        i = j + 1
    r_pos = float(np.sum(ranks[y == 1]))
    return round((r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg), 4)


def fit_temperature(p, y):
    """T>0 minimising NLL of sigmoid(logit(p)/T); deterministic grid + refine."""
    z = logit(p)
    best_t, best_v = 1.0, nll(sigmoid(z / 1.0), y)
    grid = np.exp(np.linspace(np.log(0.05), np.log(20.0), 400))
    for t in grid:
        v = nll(sigmoid(z / t), y)
        if v < best_v - 1e-12:
            best_t, best_v = float(t), v
    # local refinement (ternary search on log T)
    lo, hi = np.log(best_t / 1.3), np.log(best_t * 1.3)
    for _ in range(80):
        m1 = lo + (hi - lo) / 3
        m2 = hi - (hi - lo) / 3
        v1 = nll(sigmoid(z / np.exp(m1)), y)
        v2 = nll(sigmoid(z / np.exp(m2)), y)
        if v1 < v2:
            hi = m2
        else:
            lo = m1
    t = float(np.exp((lo + hi) / 2))
    return round(t, 4), round(nll(sigmoid(z / t), y), 4)


def fit_slope_intercept(p, y):
    """Logistic regression of gold on logit(p): returns (a, b) for a*z+b."""
    z = logit(p)
    a, b = 1.0, 0.0
    # Newton iterations (2 parameters, deterministic, no sklearn dependency)
    for _ in range(100):
        eta = a * z + b
        mu = sigmoid(eta)
        w = np.clip(mu * (1 - mu), 1e-9, None)
        r = y - mu
        g0 = float(np.sum(r * z))
        g1 = float(np.sum(r))
        h00 = float(np.sum(w * z * z))
        h01 = float(np.sum(w * z))
        h11 = float(np.sum(w))
        det = h00 * h11 - h01 * h01
        if abs(det) < 1e-12:
            break
        da = (h11 * g0 - h01 * g1) / det
        db = (-h01 * g0 + h00 * g1) / det
        a += da
        b += db
        if abs(da) < 1e-10 and abs(db) < 1e-10:
            break
    return round(float(a), 4), round(float(b), 4)


def band_policy(rows, scores, lo, hi):
    """CascadeGenome(lo, hi, use_second=1) applied with `scores` in place of p."""
    g = CascadeGenome(lo=lo, hi=hi, use_second=1)
    sub = []
    for r, s in zip(rows, scores):
        rr = dict(r)
        rr["noul_p"] = float(s)
        sub.append(rr)
    return evaluate(g, sub, lam=LAM, primary="macro_f1")


def select_band(dev_rows, dev_scores):
    """Grid-search the band on dev by the protocol scalar (macro-F1 - lam*cost)."""
    best = None
    for lo in BAND_GRID:
        for hi in BAND_GRID:
            if hi < lo:
                continue
            fit = band_policy(dev_rows, dev_scores, lo, hi)
            key = (fit["scalar"], fit["accuracy"])
            if best is None or key > best[0]:
                best = (key, lo, hi, fit)
    _, lo, hi, fit = best
    return {"lo": lo, "hi": hi, "dev_scalar": fit["scalar"], "dev_macro_f1": fit["macro_f1"],
            "dev_accuracy": fit["accuracy"], "dev_calls_per_row": fit["calls_per_row"]}


def cross_fitted_temperature(p, y, groups):
    """Group-aware 3-fold cross-fitted T (honest in-train calibration estimate)."""
    uniq = sorted(set(groups))
    fold_of = {g: i % 3 for i, g in enumerate(uniq)}
    folds = np.array([fold_of[g] for g in groups])
    out = np.zeros(len(p), dtype=float)
    ts = []
    for k in range(3):
        tr = folds != k
        va = folds == k
        if len(set(y[tr])) < 2 or not np.any(va):
            out[va] = p[va]
            ts.append(None)
            continue
        t, _ = fit_temperature(p[tr], y[tr])
        ts.append(t)
        out[va] = sigmoid(logit(p[va]) / t)
    return out, ts


def cell_report(train, dev, conf):
    ptr, ytr = probs(train), golds(train)
    pdev, ydev = probs(dev), golds(dev)
    pconf, yconf = probs(conf), golds(conf)
    groups = [r["group_id"] for r in train]

    t, train_nll_t = fit_temperature(ptr, ytr)
    a, b = fit_slope_intercept(ptr, ytr)
    cf, fold_ts = cross_fitted_temperature(ptr, ytr, groups)

    def metrics(p, y):
        return {"n": int(len(y)),
                "brier": round(brier(p, y), 4),
                "nll": round(nll(p, y), 4),
                "ece15": ece_equal_width(p, y, 15),
                "ece_adapt10": ece_equal_mass(p, y, 10),
                "auroc_gold": auroc(p, y),
                "mean_p": round(float(np.mean(p)), 4),
                "pos_rate": round(float(np.mean(y)), 4)}

    scaled = {
        "train_crossfit": cf,
        "train_insample": sigmoid(logit(ptr) / t),
        "dev": sigmoid(logit(pdev) / t),
        "confirm": sigmoid(logit(pconf) / t),
    }
    lin = {
        "train": sigmoid(a * logit(ptr) + b),
        "dev": sigmoid(a * logit(pdev) + b),
        "confirm": sigmoid(a * logit(pconf) + b),
    }

    # routing: same band class, raw vs T-scaled p, selected on dev, frozen on confirm
    sel_raw = select_band(dev, pdev)
    sel_t = select_band(dev, scaled["dev"])
    conf_raw = band_policy(conf, pconf, sel_raw["lo"], sel_raw["hi"])
    conf_t = band_policy(conf, scaled["confirm"], sel_t["lo"], sel_t["hi"])
    noul_only = evaluate(CascadeGenome(lo=0.0, hi=1.0, use_second=0), conf, lam=LAM,
                         primary="macro_f1")
    switch_only = evaluate(CascadeGenome(lo=0.0, hi=1.0, use_second=1), conf, lam=LAM,
                           primary="macro_f1")

    def net_utility(rows, scores, lo, hi):
        rep = reg = sw = 0
        for r, s in zip(rows, scores):
            if not (lo <= float(s) <= hi):
                continue
            sw += 1
            p = float(r["noul_p"])
            noul_ok = (p >= 0.5) == bool(r["gold"])
            sw_ok = bool(r.get("switch_correct", False))
            if not noul_ok and sw_ok:
                rep += 1
            elif noul_ok and not sw_ok:
                reg += 1
        return {"repair": rep, "regression": reg, "net": round((rep - reg) / len(rows), 4),
                "n_switched": sw}

    return {
        "n": {"train": len(train), "dev": len(dev), "confirm": len(conf)},
        "temperature": {"T": t, "train_nll_at_T": train_nll_t, "fold_T": fold_ts},
        "slope_intercept": {"a": a, "b": b},
        "raw": {"train": metrics(ptr, ytr), "dev": metrics(pdev, ydev),
                "confirm": metrics(pconf, yconf)},
        "temp_scaled": {"train_crossfit": metrics(scaled["train_crossfit"], ytr),
                        "train_insample": metrics(scaled["train_insample"], ytr),
                        "dev": metrics(scaled["dev"], ydev),
                        "confirm": metrics(scaled["confirm"], yconf)},
        "linear_recal": {"train": metrics(lin["train"], ytr), "dev": metrics(lin["dev"], ydev),
                         "confirm": metrics(lin["confirm"], yconf)},
        "utility_vs_calibration": {
            "auroc_p_for_gold_train": auroc(ptr, ytr),
            "auroc_p_for_repair_train": auroc(ptr, repair_labels(train)),
            "auroc_p_for_gold_confirm": auroc(pconf, yconf),
            "auroc_p_for_repair_confirm": auroc(pconf, repair_labels(conf)),
            "n_repair": {"train": int(np.sum(repair_labels(train))),
                         "dev": int(np.sum(repair_labels(dev))),
                         "confirm": int(np.sum(repair_labels(conf)))},
        },
        "routing_raw_p": {"dev_selected": sel_raw, "confirm": conf_raw,
                          "confirm_net": net_utility(conf, pconf, sel_raw["lo"], sel_raw["hi"])},
        "routing_temp_p": {"dev_selected": sel_t, "confirm": conf_t,
                           "confirm_net": net_utility(conf, scaled["confirm"], sel_t["lo"], sel_t["hi"])},
        "fixed_confirm": {"noul_only": noul_only, "switch_only": switch_only},
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
    cells = {}
    for key, (tr_p, dev_p, conf_p) in TASKS.items():
        train, _ = load(tr_p)
        dev, dev_split = load(dev_p)
        conf, conf_split = load(conf_p)
        cells[key] = cell_report(train, dev, conf)
        cells[key]["splits"] = {"train": "search_train", "dev": dev_split,
                                "confirm": conf_split}
    report = {
        "experiment_id": "P0-CALIB",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repo_commit": sha,
        "protocol_version": "v4.1",
        "analysis_script": "analyze_calibration.py",
        "analysis_type": "exploratory",
        "selection_used": True,
        "confirm_used": True,
        "pools": "T/slope fit on search_train; dev used only to select the band; confirm frozen",
        "run_manifest": {"argv": sys.argv[1:]},
        "notes": ("Temperature scaling and linear-in-logit recalibration are strictly "
                  "monotone on (0,1); the band policy class {lo<=p<=hi} is therefore "
                  "invariant up to threshold reparameterisation. The routing rows verify "
                  "this numerically rather than assuming it."),
        "cells": cells,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for k, c in cells.items():
        r = c["raw"]
        print(k,
              "| ECE raw train/dev/conf:", r["train"]["ece15"], r["dev"]["ece15"], r["confirm"]["ece15"],
              "| ECE T train/dev/conf:", c["temp_scaled"]["train_crossfit"]["ece15"],
              c["temp_scaled"]["dev"]["ece15"], c["temp_scaled"]["confirm"]["ece15"],
              "| T:", c["temperature"]["T"],
              "| AUC gold vs repair (conf):", c["utility_vs_calibration"]["auroc_p_for_gold_confirm"],
              c["utility_vs_calibration"]["auroc_p_for_repair_confirm"])


if __name__ == "__main__":
    raise SystemExit(main())
