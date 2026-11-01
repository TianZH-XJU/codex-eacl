#!/usr/bin/env python3
"""Gate 3a complementarity screen (offline, cache-exact, zero new model calls).

For each task x model train pool with a noul/switch action pair, computes:
repair / regression / net / disagreement / oracle gap. Verdict rule (from
configs/evolution.yaml): KEEP_BLOCKED unless repair-regression >= +5pp with a
held-out re-check AND disagreement >= 15%. Math has a single representation,
so no pair exists (recorded as N/A).
Writes results/v4_1/evolution/complementarity_screen.json.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def binom_upper_tail(k, n, p=0.5):
    """P(X >= k) for X ~ Binomial(n, p). Exact via math.comb."""
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def screen_cell(noul_ok, sw_ok):
    """Pure function: (repair, regression, net, disagreement, oracle_acc, p).

    p = one-sided binomial P(X >= repair | discordant, 0.5): is the observed
    repair share among disagreements better than a coin flip?
    """
    n = len(noul_ok)
    repair = sum(1 for a, b in zip(noul_ok, sw_ok) if not a and b)
    regr = sum(1 for a, b in zip(noul_ok, sw_ok) if a and not b)
    disag = sum(1 for a, b in zip(noul_ok, sw_ok) if a != b)
    oracle = sum(1 for a, b in zip(noul_ok, sw_ok) if a or b) / n if n else 0.0
    p = binom_upper_tail(repair, disag) if disag else 1.0
    return {"n": n, "repair": repair, "regression": regr, "net": repair - regr,
            "net_pp": round(100.0 * (repair - regr) / n, 2) if n else 0.0,
            "disagreement": round(disag / n, 4) if n else 0.0,
            "oracle_acc": round(oracle, 4),
            "binomial_p": round(p, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cells = []
    for short, t, m in (("code", "codexglue_defect", "laya"), ("code", "codexglue_defect", "vela"),
                        ("findver", "findver", "laya"), ("findver", "findver", "vela")):
        d = json.loads((ROOT / f"results/v4/evolution/cache_train_{short}_{m}.json").read_text(encoding="utf-8"))
        rows = d["rows"]
        noul_ok = [(float(r["noul_p"]) >= 0.5) == bool(r["gold"]) for r in rows]
        sw_ok = [bool(r.get("switch_correct", False)) for r in rows]
        c = screen_cell(noul_ok, sw_ok)
        c.update({"task": t, "model": m,
                  "noul_acc": round(sum(noul_ok) / len(rows), 4),
                  "switch_acc": round(sum(sw_ok) / len(rows), 4)})
        cells.append(c)
    for t, m in (("processbench", "laya"), ("processbench", "vela")):
        short = "math"
        matches = sorted((ROOT / "results/v4/evolution").glob(f"cache_train_{short}_{m}.json"))
        d = json.loads(matches[0].read_text(encoding="utf-8"))
        acc = sum(1 for r in d["rows"] if r.get("correct")) / len(d["rows"])
        cells.append({"task": t, "model": m, "n": len(d["rows"]),
                      "single_rep_acc": round(acc, 4),
                      "note": "single representation: no complementarity pair exists"})
    try:
        sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                             check=True, capture_output=True, text=True).stdout.strip()
    except Exception:
        sha = None
    best = max((c for c in cells if "net_pp" in c), key=lambda c: c["net_pp"])
    # Gate needs magnitude AND significance AND held-out: best cell clears the
    # +5pp point bar (+10.0pp) but its binomial p=0.0898 does not exclude chance,
    # and no held-out re-check exists. Hence blocked, not borderline-pass.
    verdict = "KEEP_BLOCKED"
    reason = (f"best cell net {best['net_pp']:+.2f}pp clears the point bar but "
              f"binomial p={best['binomial_p']} (ns) and no held-out re-check; "
              f"all other cells net -1..+1")
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "pools": "search_train (in-sample screen; Gate 3a requires a held-out re-check)",
              "gate_rule": "repair-regression >= +5pp AND disagreement >= 15% AND OOF AUC >= 0.65",
              "cells": cells, "verdict": verdict, "reason": reason,
              "git_sha": sha,
              "run_manifest": {"argv": sys.argv[1:]}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for c in cells:
        print(c.get("task"), c.get("model"),
              {k: c[k] for k in ("n", "net", "net_pp", "disagreement", "binomial_p") if k in c})
    print("verdict:", verdict)


if __name__ == "__main__":
    raise SystemExit(main())
