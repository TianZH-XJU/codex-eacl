#!/usr/bin/env python3
"""E5 minimal failure-landscape mapper (MAP-Elites-lite, fully offline).

Descriptor grid: (task, model, conf-band). Cell severity = hard joint-failure
rate for binary tasks (both noul and switch wrong), 1 - accuracy for math.
Genome = cell index; the 18-cell grid is exhaustively evaluated (= complete
MAP-Elites fill on this tiny grid; archive keeps max severity per cell).
Discover on search_train caches, validate on search_dev then confirm.
A cell counts as confirmed iff S_validate >= S_discover - 0.15.
No model calls.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

CONF_BANDS = ("low", "mid", "high")


def as_bool(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.strip().lower() in ("true", "1", "yes")
    return bool(x)


def as_float(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def conf_of_binary(row):
    p = as_float(row.get("noul_p"))
    if p is None:
        return None
    c = abs(p - 0.5)
    if c < 0.1:
        return "low"
    if c < 0.25:
        return "mid"
    return "high"


def conf_of_math(row):
    m = as_float(row.get("margin"))
    if m is None:
        return None
    if m < 0.05:
        return "low"
    if m < 0.15:
        return "mid"
    return "high"


def is_binary(row):
    return "noul_p" in row


def load_cache(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return d["rows"], {"file": Path(path).name, "task": d.get("task"),
                       "model": d.get("model"), "split": d.get("split")}


def bucketize(files):
    """Return {(task, model, conf): [rows]} across the given cache files.
    Rows with missing conf go to an explicit "missing" band (fail-closed:
    they count in severity denominators instead of silently vanishing)."""
    rooms = {}
    for f in files:
        rows, meta = load_cache(f)
        task = meta["task"]
        tshort = {"codexglue_defect": "code", "findver": "findver",
                  "processbench": "math"}.get(task, task)
        for r in rows:
            c = conf_of_binary(r) if is_binary(r) else conf_of_math(r)
            if c is None:
                c = "missing"
            rooms.setdefault((tshort, meta["model"], c), []).append(r)
    return rooms


def severity(rows):
    """(severity, n, detail). Binary: both-wrong rate. Math: 1 - accuracy.
    Missing noul_p counts as both-wrong (fail-closed): a row with no usable
    model output cannot be rescued by any policy."""
    if not rows:
        return 0.0, 0, {}
    if is_binary(rows[0]):
        fixed = []
        for r in rows:
            p = as_float(r.get("noul_p"))
            if p is None:
                fixed.append((False, False))
                continue
            noul_ok = (p >= 0.5) == as_bool(r["gold"])
            fixed.append((noul_ok, bool(r.get("switch_correct", False))))
        n = len(fixed)
        both_wrong = sum(1 for no, sw in fixed if not no and not sw)
        only_noul = sum(1 for no, sw in fixed if no and not sw)
        only_switch = sum(1 for no, sw in fixed if not no and sw)
        both_ok = sum(1 for no, sw in fixed if no and sw)
        return (both_wrong / n if n else 0.0), n, {
            "both_wrong": both_wrong, "only_noul": only_noul,
            "only_switch": only_switch, "both_ok": both_ok}
    acc = sum(1 for r in rows if r.get("correct") is True) / len(rows)
    return 1.0 - acc, len(rows), {"acc": round(acc, 4)}


def wilson_lb(k, n, z=1.96):
    if n == 0:
        return 0.0
    p = k / n
    den = 1 + z * z / n
    center = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return round(max(0.0, (center - half) / den), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-glob", default="results/v4/evolution/cache_train_*.json")
    ap.add_argument("--dev-glob", default="results/v4/evolution/cache_[a-z]*_*.json")
    ap.add_argument("--confirm-glob", default="results/v4/evolution/cache_confirm_*.json")
    ap.add_argument("--min-n", type=int, default=8)
    ap.add_argument("--tol", type=float, default=0.15)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import glob as _glob
    train_files = sorted(f for f in _glob.glob(str(ROOT / args.train_glob))
                         if "/cache_train_" in f.replace("\\", "/"))
    dev_files = sorted(f for f in _glob.glob(str(ROOT / args.dev_glob))
                       if "/cache_train_" not in f.replace("\\", "/")
                       and "/cache_confirm_" not in f.replace("\\", "/")
                       and "/cache_test_" not in f.replace("\\", "/")
                       and "cache_" in f.replace("\\", "/"))
    conf_files = [f for f in _glob.glob(str(ROOT / args.confirm_glob))
                  if "/cache_test_" not in f.replace("\\", "/")]
    assert conf_files, f"no confirm caches matched {args.confirm_glob}"
    assert train_files, f"no train caches matched {args.train_glob}"
    disc = bucketize(train_files)
    dev = bucketize(dev_files)
    conf = bucketize(conf_files)

    cells = []
    for cell in sorted(disc):
        s, n, det = severity(disc[cell])
        if n < args.min_n:
            continue
        s_dev, n_dev, _ = severity(dev.get(cell, []))
        s_con, n_con, _ = severity(conf.get(cell, []))
        confirmed_dev = (n_dev >= args.min_n) and (s_dev >= s - args.tol)
        confirmed_con = (n_con >= args.min_n) and (s_con >= s - args.tol)
        cells.append({"cell": list(cell), "n_discover": n,
                      "severity_discover": round(s, 4),
                      "wilson_lb_discover": wilson_lb(int(round(s * n)), n),
                      "detail_discover": det,
                      "n_dev": n_dev, "severity_dev": round(s_dev, 4),
                      "confirmed_dev": confirmed_dev,
                      "n_confirm": n_con, "severity_confirm": round(s_con, 4),
                      "confirmed_confirm": confirmed_con,
                      "genome": {"kind": "cell_index",
                                 "space": ["task", "model", "conf"],
                                 "note": "exhaustive 18-cell evaluation = full archive fill"}})
    cells.sort(key=lambda c: (-c["severity_discover"], -c["n_discover"]))
    report = {"discover": "search_train caches", "validate": ["search_dev", "confirm"],
              "min_n": args.min_n, "tol": args.tol,
              "n_cells_eligible": len(cells),
              "n_confirmed_dev": sum(1 for c in cells if c["confirmed_dev"]),
              "n_confirmed_confirm": sum(1 for c in cells if c["confirmed_confirm"]),
              "cells": cells,
              "status": "passed" if cells else "empty_archive"}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for c in cells[:8]:
        print(c["cell"], "S=", c["severity_discover"], "n=", c["n_discover"],
              "dev:", c["confirmed_dev"], "conf:", c["confirmed_confirm"])
    print(json.dumps({"eligible": len(cells),
                      "confirmed_dev": report["n_confirmed_dev"],
                      "confirmed_confirm": report["n_confirmed_confirm"]}))


if __name__ == "__main__":
    raise SystemExit(main())
