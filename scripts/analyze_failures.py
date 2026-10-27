#!/usr/bin/env python3
"""Offline failure-mode analysis on the SINGLE-first dev cache (no Laya calls)."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np


def load_cache(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-cache", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = load_cache(args.dev_cache)
    by = {}
    for r in rows:
        by.setdefault(r["index"], []).append(r)

    def get(items, s):
        return next(x for x in items if x["strategy"] == s)

    cats = {"single_only_correct": 0, "wide_only_fix": 0, "tournament_only_fix": 0,
            "both_fix": 0, "single_correct_upgrade_wrong": 0, "all_wrong": 0, "all_correct": 0}
    upgrade_gain_conf, upgrade_gain_margin, upgrade_gain_ent = [], [], []
    upgrade_loss_conf, upgrade_loss_margin = [], []
    wide_only_rows, tour_only_rows = [], []
    for i, items in by.items():
        s, w, t = get(items, "single"), get(items, "wide"), get(items, "tournament")
        sr = single_row_feat = s
        if s["correct"] and w["correct"] and t["correct"]:
            cats["all_correct"] += 1
        elif not s["correct"] and not w["correct"] and not t["correct"]:
            cats["all_wrong"] += 1
        if (not s["correct"]) and w["correct"] and (not t["correct"]):
            cats["wide_only_fix"] += 1
            wide_only_rows.append(i)
        if (not s["correct"]) and (not w["correct"]) and t["correct"]:
            cats["tournament_only_fix"] += 1
            tour_only_rows.append(i)
        if (not s["correct"]) and w["correct"] and t["correct"]:
            cats["both_fix"] += 1
        if s["correct"] and ((not w["correct"]) or (not t["correct"])):
            cats["single_correct_upgrade_wrong"] += 1
        # gain = target(tournament) fixes single error; loss = single right but tournament wrong
        if (not s["correct"]) and t["correct"]:
            upgrade_gain_conf.append(s["initial_answer_confidence"] or 0)
            upgrade_gain_margin.append(s["initial_margin"] or 0)
            upgrade_gain_ent.append(s["initial_entropy_confidence"] or 0)
        if s["correct"] and (not t["correct"]):
            upgrade_loss_conf.append(s["initial_answer_confidence"] or 0)
            upgrade_loss_margin.append(s["initial_margin"] or 0)

    def stats(x):
        a = np.array(x, dtype=float)
        return {"n": len(a), "mean": round(float(a.mean()), 4) if len(a) else None,
                "p25": round(float(np.percentile(a, 25)), 4) if len(a) else None,
                "p50": round(float(np.percentile(a, 50)), 4) if len(a) else None,
                "p75": round(float(np.percentile(a, 75)), 4) if len(a) else None}

    # Binned upgrade value by confidence decile.
    bins = {}
    for i, items in by.items():
        s, t = get(items, "single"), get(items, "tournament")
        c = float(s["initial_answer_confidence"] or 0.0)
        b = f"{int(c*10)/10:.1f}-{int(c*10)/10+0.1:.1f}"
        d = bins.setdefault(b, {"n": 0, "single_correct": 0, "tournament_correct": 0, "gain": 0})
        d["n"] += 1
        d["single_correct"] += int(s["correct"])
        d["tournament_correct"] += int(t["correct"])
        d["gain"] += int((not s["correct"]) and t["correct"])

    out = {
        "n_dev": len(by),
        "categories": cats,
        "upgrade_gain_confidence": stats(upgrade_gain_conf),
        "upgrade_gain_margin": stats(upgrade_gain_margin),
        "upgrade_gain_entropy_conf": stats(upgrade_gain_ent),
        "upgrade_loss_confidence": stats(upgrade_loss_conf),
        "upgrade_loss_margin": stats(upgrade_loss_margin),
        "confidence_bins": bins,
        "wide_only_fix_indices": wide_only_rows,
        "tournament_only_fix_indices": tour_only_rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
