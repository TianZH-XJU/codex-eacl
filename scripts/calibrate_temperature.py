#!/usr/bin/env python3
"""Single-temperature scaling diagnostic on cached SINGLE probabilities.

Fit ONE temperature T on TRAIN cache only (minimize NLL of the true label
under p_T propto p^(1/T)). Evaluate raw (T=1) vs calibrated on DEV:
NLL, multiclass Brier, ECE (15 equal-width bins), plus a re-fit of the
confidence-threshold cascade on TRAIN calibrated confidence applied to DEV.

Fully offline: no Laya calls.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

EPS = 1e-12
N_BINS = 15


def tempered(probs, T):
    pw = {k: (max(v, EPS) ** (1.0 / T)) for k, v in probs.items()}
    z = sum(pw.values())
    return {k: v / z for k, v in pw.items()}


def nll_of_rows(rows, T):
    total = 0.0
    for r in rows:
        probs = r.get("initial_probabilities") or {}
        gold = r.get("gold_normalized") or r.get("gold", "")
        if not probs:
            total += -math.log(EPS)
            continue
        pt = tempered(probs, T)
        p = pt.get(gold, pt.get(gold.replace(" ", "_"), EPS))
        total += -math.log(max(p, EPS))
    return total / max(len(rows), 1)


def dev_metrics(single_rows, T):
    nlls, brier_sum, confs, hits = [], 0.0, [], []
    for r in single_rows:
        probs = r.get("initial_probabilities") or {}
        gold = r.get("gold_normalized") or r.get("gold", "")
        pt = tempered(probs, T) if probs else {}
        p_true = pt.get(gold, pt.get(gold.replace(" ", "_"), 0.0)) if pt else 0.0
        nlls.append(-math.log(max(p_true, EPS)))
        if pt:
            brier_sum += sum((p - (1.0 if k == gold or k == gold.replace(" ", "_") else 0.0)) ** 2
                             for k, p in pt.items())
        c = max(pt.values()) if pt else 0.0
        confs.append(c)
        # correctness of SINGLE itself
        hits.append(1.0 if r.get("correct") else 0.0)
    n = len(single_rows)
    # ECE over confidence bins
    ece = 0.0
    for b in range(N_BINS):
        lo, hi = b / N_BINS, (b + 1) / N_BINS
        idx = [i for i, c in enumerate(confs) if (c > lo and c <= hi) or (b == 0 and c == 0.0)]
        if idx:
            acc = sum(hits[i] for i in idx) / len(idx)
            conf = sum(confs[i] for i in idx) / len(idx)
            ece += len(idx) / n * abs(acc - conf)
    return {
        "nll": round(float(sum(nlls) / n), 4) if n else None,
        "brier": round(float(brier_sum / n), 4) if n else None,
        "ece_15bin": round(float(ece), 4),
    }


def single_rows_of(cache_rows):
    return [r for r in cache_rows if r.get("strategy") == "single"]


def fit_threshold(train_single, dev_groups, key, upgrade_target):
    vals = sorted({float(r.get(key) or 0.0) for r in train_single})
    best_t, best_acc = None, -1.0
    # need per-example correctness of single vs upgrade target on train
    return vals, best_t, best_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-cache", required=True)
    ap.add_argument("--dev-cache", required=True)
    ap.add_argument("--upgrade-target", default="tournament")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    def load(p):
        return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]

    train, dev = load(args.train_cache), load(args.dev_cache)
    tr_single = single_rows_of(train)
    de_single = single_rows_of(dev)

    def group(rows):
        by = {}
        for r in rows:
            by.setdefault(r["index"], []).append(r)
        return by

    train_by, dev_by = group(train), group(dev)

    res = minimize_scalar(lambda T: nll_of_rows(tr_single, T), bounds=(0.1, 10.0),
                          method="bounded", options={"xatol": 1e-3})
    T = float(res.x)

    raw = dev_metrics(de_single, 1.0)
    cal = dev_metrics(de_single, T)
    train_raw_nll = round(nll_of_rows(tr_single, 1.0), 4)
    train_cal_nll = round(nll_of_rows(tr_single, T), 4)

    # Re-fit confidence-threshold cascade on TRAIN (raw vs calibrated), eval on DEV.
    def add_calibrated(rows, T):
        out = []
        for r in rows:
            probs = r.get("initial_probabilities") or {}
            pt = tempered(probs, T) if probs else {}
            r2 = dict(r)
            r2["calibrated_confidence"] = max(pt.values()) if pt else 0.0
            out.append(r2)
        return out

    tr_cal = add_calibrated(tr_single, T)

    def fit_eval(key):
        vals = sorted({float(r.get(key) or 0.0) for r in tr_cal})
        best_t, best_acc = None, -1.0
        tr_map = {r["index"]: r for r in tr_cal}
        for t in vals:
            ok = tot = 0
            for i, items in train_by.items():
                v = float(tr_map[i].get(key) or 0.0)
                pred = "single" if v >= t else args.upgrade_target
                row = next(x for x in items if x["strategy"] == pred)
                ok += int(row["correct"])
                tot += 1
            acc = ok / tot if tot else 0.0
            if acc > best_acc:
                best_acc, best_t = acc, t
        # eval on dev with calibrated conf
        de_map = {r["index"]: (max(tempered(r.get("initial_probabilities") or {}, T).values()) if r.get("initial_probabilities") else 0.0) for r in de_single}
        ok = tot = 0
        lats, calls = [], []
        for i, items in dev_by.items():
            pred = "single" if de_map[i] >= best_t else args.upgrade_target
            row = next(x for x in items if x["strategy"] == pred)
            ok += int(row["correct"])
            tot += 1
            lats.append(row["cumulative_latency_ms"])
            calls.append(row["cumulative_calls"])
        return {
            "threshold": round(float(best_t), 4) if best_t is not None else None,
            "train_accuracy": round(best_acc, 4),
            "dev_accuracy": round(ok / tot, 4) if tot else None,
            "dev_mean_cumulative_ms": round(float(sum(lats)) / len(lats), 1) if lats else None,
            "dev_mean_cumulative_calls": round(float(sum(calls)) / len(calls), 3) if calls else None,
        }

    cal_route = fit_eval("calibrated_confidence")

    # Raw-confidence threshold on same split for direct comparison.
    def fit_eval_raw():
        vals = sorted({float(r.get("initial_answer_confidence") or 0.0) for r in tr_single})
        best_t, best_acc = None, -1.0
        tr_map = {r["index"]: r for r in tr_single}
        for t in vals:
            ok = tot = 0
            for i, items in train_by.items():
                v = float(tr_map[i].get("initial_answer_confidence") or 0.0)
                pred = "single" if v >= t else args.upgrade_target
                row = next(x for x in items if x["strategy"] == pred)
                ok += int(row["correct"])
                tot += 1
            acc = ok / tot if tot else 0.0
            if acc > best_acc:
                best_acc, best_t = acc, t
        de_map = {r["index"]: float(r.get("initial_answer_confidence") or 0.0) for r in de_single}
        ok = tot = 0
        lats, calls = [], []
        for i, items in dev_by.items():
            pred = "single" if de_map[i] >= best_t else args.upgrade_target
            row = next(x for x in items if x["strategy"] == pred)
            ok += int(row["correct"])
            tot += 1
            lats.append(row["cumulative_latency_ms"])
            calls.append(row["cumulative_calls"])
        return {
            "threshold": round(float(best_t), 4) if best_t is not None else None,
            "train_accuracy": round(best_acc, 4),
            "dev_accuracy": round(ok / tot, 4) if tot else None,
            "dev_mean_cumulative_ms": round(float(sum(lats)) / len(lats), 1) if lats else None,
            "dev_mean_cumulative_calls": round(float(sum(calls)) / len(calls), 3) if calls else None,
        }

    raw_route = fit_eval_raw()

    out = {
        "train_cache": args.train_cache,
        "dev_cache": args.dev_cache,
        "n_train_single": len(tr_single),
        "n_dev_single": len(de_single),
        "temperature_T": round(T, 4),
        "train_nll_raw_T1": train_raw_nll,
        "train_nll_calibrated": train_cal_nll,
        "dev_raw_T1": raw,
        "dev_calibrated": cal,
        "upgrade_target": args.upgrade_target,
        "route_raw_confidence": raw_route,
        "route_calibrated_confidence": cal_route,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
