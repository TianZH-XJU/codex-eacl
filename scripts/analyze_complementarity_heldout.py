#!/usr/bin/env python3
"""P0-3 补充实验：留出池上的动作互补性复核（离线，零新模型调用）。

回答两个问题：
1. search_train 上观察到的 repair/harmful 关系能否在 selection_dev / confirm 重现
   （方向沿用 train 的较优固定动作，不在留出池重新选择基准）？
2. 在主指标 macro-F1 上，两种固定动作的差值是否显示可实现的净空间？

输入：cache_train_{code,findver}_{laya,vela}.json 与对应 dev/confirm 缓存。
输出：results/v4_1/evolution/complementarity_heldout.json
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

rng = np.random.RandomState(17)

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


def load(p):
    d = json.loads((ROOT / p).read_text(encoding="utf-8"))
    return d["rows"], d.get("split", "unknown")


def cell_stats(rows, base):
    """base: 'noul' or 'switch' (frozen direction from train).
    Returns repair/harmful/net/disagreement/headroom with group-bootstrap CIs."""
    def _ok(r):
        try:
            v = float(r["noul_p"])
        except (TypeError, ValueError):
            return False  # fail-closed: unusable output counts as wrong
        if not math.isfinite(v) or not 0.0 <= v <= 1.0:
            return False
        return (v >= 0.5) == bool(r["gold"])
    noul_ok = [_ok(r) for r in rows]
    sw_ok = [bool(r.get("switch_correct", False)) for r in rows]
    n = len(rows)
    if base == "noul":
        rep = sum(1 for a, b in zip(noul_ok, sw_ok) if not a and b)
        harm = sum(1 for a, b in zip(noul_ok, sw_ok) if a and not b)
        orac = sum(1 for a, b in zip(noul_ok, sw_ok) if a or b) / n
    else:
        rep = sum(1 for a, b in zip(noul_ok, sw_ok) if b is False and a)
        harm = sum(1 for a, b in zip(noul_ok, sw_ok) if b and not a)
        orac = sum(1 for a, b in zip(noul_ok, sw_ok) if a or b) / n
    by_g = {}
    for r, a, b in zip(rows, noul_ok, sw_ok):
        if base == "noul":
            d = (int((not a) and b), int(a and (not b)))
        else:
            d = (int((not b) and a), int(b and (not a)))
        by_g.setdefault(r["group_id"], []).append(d)
    groups = sorted(by_g)
    est = (rep - harm) / n
    boots = []
    for _ in range(5000):
        s = rng.choice(groups, size=len(groups), replace=True)
        flat = [x for g in s for x in by_g[g]]
        boots.append((sum(x[0] for x in flat) - sum(x[1] for x in flat)) / len(flat))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"n": n, "repair": rep, "harmful": harm, "net": rep - harm,
            "net_pp": round(est, 4), "net_ci95": [round(float(lo), 4), round(float(hi), 4)],
            "mcnemar_exact_p": round(mcnemar_exact(rep, harm), 4),
            "disagreement": round(sum(1 for a, b in zip(noul_ok, sw_ok) if a != b) / n, 4),
            "oracle_headroom_acc": round(orac - ((sum(sw_ok) if base == "switch" else sum(noul_ok)) / n), 4)}


def mcnemar_exact(b, c):
    """Two-sided exact conditional (McNemar) p-value for net repair:
    b = #(wrong->right), c = #(right->wrong); under H0 each discordant row is
    equally likely to be repair or regression, so b ~ Binomial(b+c, 0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    from math import comb
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def macro_f1(rows, use_switch):
    """Macro-F1 of fixed action on this pool (use_switch False -> Noul only)."""
    tp = tn = fp = fn = 0
    for r in rows:
        if use_switch:
            ok = bool(r.get("switch_correct", False))
        else:
            try:
                v = float(r["noul_p"])
            except (TypeError, ValueError):
                v = float("nan")
            ok = math.isfinite(v) and 0.0 <= v <= 1.0 and (v >= 0.5) == bool(r["gold"])
        if ok and bool(r["gold"]):
            tp += 1
        elif ok:
            tn += 1
        elif bool(r["gold"]):
            fn += 1
        else:
            fp += 1
    f1p = 2 * tp / max(2 * tp + fp + fn, 1)
    f1n = 2 * tn / max(2 * tn + fp + fn, 1)
    return (f1p + f1n) / 2


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
              "repo_commit": sha, "protocol_version": "v4.1",
              "design": ("frozen train-selected fixed-action direction; repair/harmful and "
                         "headroom recomputed on dev and confirm; macro-F1 fixed-action deltas"),
              "run_manifest": {"argv": sys.argv[1:]},
              "analysis_script": "analyze_complementarity_heldout.py",
              "analysis_type": "confirmatory",
              "selection_used": False, "confirm_used": True,
              "pools": "frozen train direction; dev=selection, confirm=held-out check",
              "cells": {}}
    for key, (tr_p, dev_p, conf_p) in TASKS.items():
        train, tr_split = load(tr_p)
        dev, dev_split = load(dev_p)
        conf, conf_split = load(conf_p)
        # frozen direction: better fixed action on TRAIN (no re-selection on held-out)
        tr_noul = macro_f1(train, False)
        tr_sw = macro_f1(train, True)
        base = "noul" if tr_noul >= tr_sw else "switch"
        cell = {"base_action_frozen": base,
                "train_macro_f1": {"noul": round(tr_noul, 4), "switch": round(tr_sw, 4)},
                "splits": {"train": tr_split, "dev": dev_split, "confirm": conf_split}}
        for tag, rows in (("train", train), ("dev", dev), ("confirm", conf)):
            st = cell_stats(rows, base)
            st["macro_f1_noul"] = round(macro_f1(rows, False), 4)
            st["macro_f1_switch"] = round(macro_f1(rows, True), 4)
            st["macro_f1_delta_switch_minus_noul"] = round(st["macro_f1_switch"] - st["macro_f1_noul"], 4)
            cell[tag] = st
        report["cells"][key] = cell
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for k, c in report["cells"].items():
        print(k, "base:", c["base_action_frozen"],
              "| train net:", c["train"]["net"], c["train"]["net_ci95"],
              "| dev net:", c["dev"]["net"], c["dev"]["net_ci95"],
              "| conf net:", c["confirm"]["net"], c["confirm"]["net_ci95"],
              "| conf macroF1 d:", c["confirm"]["macro_f1_delta_switch_minus_noul"])


if __name__ == "__main__":
    raise SystemExit(main())
