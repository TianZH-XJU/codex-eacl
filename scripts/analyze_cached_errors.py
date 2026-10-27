#!/usr/bin/env python3
"""Offline error analysis on frozen caches (no Laya calls, no test access).

Replays locked controllers fitted on the TRAIN cache only and diagnoses
per-class / per-sample behavior on DEV and CONFIRMATION separately.
Confirmation output is post-hoc explanation only — never used for tuning.

Outputs per split: per-class error table, per-sample diagnosis, plus a
global extreme-latency list (strategy latency >= 60 s, raw values kept).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_controller_baselines import features, single_row  # noqa: E402
from run_three_action import STRATEGIES, choose_action, estimate_train_costs, group, load_cache  # noqa: E402

LAM = 2e-05
EXTREME_MS = 60000.0


def fit_locked_controllers(train_by):
    """Deterministic replay of the locked config (unweighted, C=1.0/0.2)."""
    X = np.array([features(single_row(train_by[i])) for i in train_by], dtype=float)
    num = {}
    for s in STRATEGIES:
        y = np.array([int(next(x for x in train_by[i] if x["strategy"] == s)["correct"]) for i in train_by])
        clf = LogisticRegression(max_iter=1000, random_state=17, class_weight=None, C=1.0)
        clf.fit(StandardScaler().fit_transform(X), y)
        num[s] = clf
    scaler = StandardScaler().fit(X)
    cats = np.array([[single_row(train_by[i]).get("initial_choice") or ""] for i in train_by])
    enc = OneHotEncoder(handle_unknown="ignore").fit(cats)
    X2 = np.hstack([scaler.transform(X), enc.transform(cats).toarray()])
    ch = {}
    for s in STRATEGIES:
        y = np.array([int(next(x for x in train_by[i] if x["strategy"] == s)["correct"]) for i in train_by])
        clf = LogisticRegression(max_iter=1000, random_state=17, class_weight=None, C=0.2)
        clf.fit(X2, y)
        ch[s] = clf
    # confidence threshold from train (upgrade=tournament), as in baselines.
    vals = sorted({float(single_row(train_by[i]).get("initial_answer_confidence") or 0.0) for i in train_by})
    best_t, best_acc = None, -1.0
    for t in vals:
        ok = sum(int(next(x for x in train_by[i]
                          if x["strategy"] == ("single" if float(single_row(train_by[i]).get("initial_answer_confidence") or 0.0) >= t else "tournament"))["correct"]) for i in train_by)
        if ok / len(train_by) > best_acc:
            best_acc, best_t = ok / len(train_by), t
    costs = estimate_train_costs(train_by)
    return {"num": num, "ch": (scaler, enc, ch), "threshold": best_t, "costs": costs}


def make_predictors(ctrl, train_by):
    """Build stateless predictor closures with their own fitted transformers."""
    Xtr = np.array([features(single_row(train_by[i])) for i in train_by], dtype=float)
    sc = StandardScaler().fit(Xtr)
    num = ctrl["num"]
    scaler_c, enc, ch = ctrl["ch"]
    costs = ctrl["costs"]

    def p_num(row):
        X = sc.transform(np.array([features(row)], dtype=float))
        out = {}
        for s in STRATEGIES:
            cls = list(num[s].classes_)
            p = num[s].predict_proba(X)[0]
            out[s] = float(p[cls.index(1)]) if 1 in cls else 0.0
        return out

    def p_ch(row):
        Xn = scaler_c.transform(np.array([features(row)], dtype=float))
        C = enc.transform(np.array([[row.get("initial_choice") or ""]])).toarray()
        X = np.hstack([Xn, C])
        out = {}
        for s in STRATEGIES:
            cls = list(ch[s].classes_)
            p = ch[s].predict_proba(X)[0]
            out[s] = float(p[cls.index(1)]) if 1 in cls else 0.0
        return out

    def p_thr(row):
        v = float(row.get("initial_answer_confidence") or 0.0)
        return "single" if v >= ctrl["threshold"] else "tournament"

    return p_num, p_ch, p_thr, costs


def analyze_split(by, p_num, p_ch, p_thr, costs):
    per_class = {}
    samples = []
    for idx, items in by.items():
        srow = single_row(items)
        corr = {s: bool(next(x for x in items if x["strategy"] == s)["correct"]) for s in STRATEGIES}
        ch_num = choose_action(p_num(srow), costs, LAM)
        ch_ch = choose_action(p_ch(srow), costs, LAM)
        ch_thr = p_thr(srow)
        chosen = {"numeric_only": ch_num, "numeric_choice": ch_ch, "conf_threshold": ch_thr,
                  "fixed_single": "single", "fixed_wide": "wide", "fixed_tournament": "tournament"}
        correct_of = {m: corr[a] for m, a in chosen.items()}
        any_correct = any(corr.values())
        all_wrong = not any_correct
        alternatives = [s for s in STRATEGIES if corr[s]]
        samples.append({
            "index": idx,
            "gold": srow.get("gold_normalized") or srow.get("gold", ""),
            "initial_choice": srow.get("initial_choice"),
            "correct": corr,
            "chosen": chosen,
            "chosen_correct": correct_of,
            "has_correct_alternative": any_correct,
            "correct_alternatives": alternatives,
            "all_wrong": all_wrong,
        })
        lab = srow.get("gold_normalized") or srow.get("gold", "")
        d = per_class.setdefault(lab, {"n": 0, "single_correct": 0, "wide_correct": 0, "tournament_correct": 0,
                                       "oracle_correct": 0, "all_wrong": 0, "methods": {}})
        d["n"] += 1
        d["single_correct"] += int(corr["single"])
        d["wide_correct"] += int(corr["wide"])
        d["tournament_correct"] += int(corr["tournament"])
        d["oracle_correct"] += int(any_correct)
        d["all_wrong"] += int(all_wrong)
        for m, a in chosen.items():
            md = d["methods"].setdefault(m, {"chosen_correct": 0, "avoidable_errors": 0, "mix": {s: 0 for s in STRATEGIES}})
            md["mix"][a] += 1
            md["chosen_correct"] += int(corr[a])
            if (not corr[a]) and any_correct:
                md["avoidable_errors"] += 1
    return {"per_class": per_class, "samples": samples}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-cache", required=True)
    ap.add_argument("--dev-cache", required=True)
    ap.add_argument("--confirmation-cache", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    train_by = group(load_cache(args.train_cache))
    dev_by = group(load_cache(args.dev_cache))
    conf_by = group(load_cache(args.confirmation_cache))

    ctrl = fit_locked_controllers(train_by)
    p_num, p_ch, p_thr, costs = make_predictors(ctrl, train_by)

    extremes = []
    for path in (args.train_cache, args.dev_cache, args.confirmation_cache):
        for r in load_cache(path):
            if (r.get("latency_ms") or 0) >= EXTREME_MS:
                extremes.append({"cache": Path(path).name, "index": r["index"], "strategy": r["strategy"],
                                 "latency_ms": r["latency_ms"], "cumulative_latency_ms": r["cumulative_latency_ms"],
                                 "correct": bool(r["correct"]), "text": (r.get("text") or "")[:80]})

    out = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "locked_config": {"class_weight": "none", "C_numeric": 1.0, "C_choice": 0.2, "lambda_per_ms": LAM},
        "extreme_latency_threshold_ms": EXTREME_MS,
        "extreme_latency_note": "raw values kept; not deleted or truncated",
        "latency_extremes": sorted(extremes, key=lambda r: -r["latency_ms"]),
        "dev": analyze_split(dev_by, p_num, p_ch, p_thr, costs),
        "confirmation_posthoc": analyze_split(conf_by, p_num, p_ch, p_thr, costs),
        "confirmation_note": "post-hoc explanation only; not used for tuning or selection",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    # console digest
    for split in ("dev", "confirmation_posthoc"):
        data = out[split]
        n = len(data["samples"])
        print(f"== {split} (n={n}) ==")
        for m in ("fixed_single", "fixed_wide", "fixed_tournament", "conf_threshold", "numeric_only", "numeric_choice"):
            ok = sum(s["chosen_correct"][m] for s in data["samples"])
            av = sum(1 for s in data["samples"] if (not s["chosen_correct"][m]) and s["has_correct_alternative"])
            print(f"  {m:18s} acc={ok/n:.4f} avoidable_errors={av}")
        print(f"  all_wrong={sum(s['all_wrong'] for s in data['samples'])}")


if __name__ == "__main__":
    raise SystemExit(main())
