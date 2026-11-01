#!/usr/bin/env python3
"""Traditional-controller baselines on committed policy caches (offline).

Binary tasks (code/findver): fixed uncertainty-band thresholds + an in-sample
LR router on [p, |p-0.5|] predicting P(switch fixes | row). Both are compared
vs the GA-best genome per seed with paired group-bootstrap CIs. In-sample
selection is disclosed for BOTH sides (GA-best is also in-sample selected).
Math task: fixed abstain thresholds vs never-abstain, paired CI.
No model calls.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evolution import CascadeGenome, decide  # noqa: E402
from test_evo_significance import paired_ci, rowwise, rowwise_math  # noqa: E402
from evolution import AbstainGenome  # noqa: E402

EV = ROOT / "results/v4/evolution"


def band_correct(rows, lo, hi):
    from evolution import decide as _decide
    out = []
    for r in rows:
        if r.get("noul_p") is None:
            out.append((r["group_id"], 0))
            continue
        p = r["noul_p"]
        if lo <= p <= hi:
            out.append((r["group_id"], int(bool(r.get("switch_correct", False)))))
        else:
            out.append((r["group_id"], int((p >= 0.5) == bool(r["gold"]))))
    return out


def lr_router_correct(rows):
    X, y, groups = [], [], []
    for r in rows:
        if r.get("noul_p") is None:
            continue
        p = r["noul_p"]
        noul_ok = (p >= 0.5) == bool(r["gold"])
        fix = bool(r.get("switch_correct", False)) and not noul_ok
        X.append([p, abs(p - 0.5)])
        y.append(int(fix))
        groups.append(r["group_id"])
    if len(set(y)) < 2:
        return None  # no variation: router cannot learn
    clf = LogisticRegression(max_iter=1000).fit(np.array(X), np.array(y))
    pred = clf.predict(np.array(X))
    out, i = [], 0
    for r in rows:
        if r.get("noul_p") is None:
            out.append((r["group_id"], 0))
            continue
        p = r["noul_p"]
        if pred[i]:
            out.append((r["group_id"], int(bool(r.get("switch_correct", False)))))
        else:
            out.append((r["group_id"], int((p >= 0.5) == bool(r["gold"]))))
        i += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    report = {}
    for t in ("code", "findver"):
        for m in ("laya", "vela"):
            cache = json.loads((EV / f"cache_{t}_{m}.json").read_text(encoding="utf-8"))
            rows = cache["rows"]
            evo = json.loads((EV / f"evo_{t}_{m}.json").read_text(encoding="utf-8"))
            # threshold grid (in-sample best, disclosed)
            grid = {}
            for d in (0.1, 0.2, 0.3):
                v = [x for _, x in band_correct(rows, 0.5 - d, 0.5 + d)]
                grid[str(d)] = round(float(np.mean(v)), 4)
            best_d = max(grid, key=lambda k: grid[k])
            rep = {"task": f"{t}/{m}", "threshold_grid": grid, "best_d": best_d,
                   "best_acc": grid[best_d]}
            lr = lr_router_correct(rows)
            if lr is not None:
                lr_acc = round(float(np.mean([x for _, x in lr])), 4)
                rep["lr_router_acc"] = lr_acc
                # paired CI: LR router vs fixed noul-only
                base = rowwise(CascadeGenome(lo=0.0, hi=1.0, use_second=0), rows)
                rep["lr_vs_noulonly"] = paired_ci(base, lr)
            # paired CI: GA-best (per seed) vs threshold-best and vs LR
            tb = band_correct(rows, 0.5 - float(best_d), 0.5 + float(best_d))
            per_seed = {}
            for s in evo["seeds"]:
                gb = CascadeGenome(**evo["per_seed"][str(s)]["ga"]["best"])
                gvec = rowwise(gb, rows)
                drow = {"ga_minus_threshbest": paired_ci(tb, gvec)}
                if lr is not None:
                    drow["ga_minus_lr"] = paired_ci(lr, gvec)
                per_seed[str(s)] = drow
            rep["per_seed_ga_vs_controls"] = per_seed
            report[f"{t}_{m}"] = rep
    # math: fixed abstain thresholds vs never-abstain
    for m in ("laya", "vela"):
        cache = json.loads((EV / f"cache_math_{m}.json").read_text(encoding="utf-8"))
        rows = cache["rows"]
        from evolution import evaluate_math as _em
        base_vec = []
        for r in rows:
            if r.get("top1") is None:
                base_vec.append((r["group_id"], 0))
            else:
                base_vec.append((r["group_id"], int(bool(r.get("correct", False)))))
        th = {}
        for t in (0.3, 0.5, 0.7):
            g = AbstainGenome(t=t, use_margin=0, m=0.0)
            th[str(t)] = _em(g, rows)
        best_t = max(th, key=lambda k: th[k]["sel_accuracy"])
        rep = {"thresholds": th, "best_t": best_t,
               "never_abstain_acc": round(float(np.mean([x for _, x in base_vec])), 4)}
        # paired CI on ANSWERED-correct? No: compare per-row correct (abstain=0) vs never-abstain
        gb = AbstainGenome(t=float(best_t), use_margin=0, m=0.0)
        rep["threshbest_vs_never"] = paired_ci(
            base_vec, [(g, 0 if (r.get("top1") or 1.0) < gb.t else v)
                       for (g, v), r in zip(base_vec, rows)])
        report[f"math_{m}"] = rep
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for k, v in report.items():
        if "per_seed_ga_vs_controls" in v:
            print(k, {s: {kk: (vv["est"], vv["ci95"], "SIG" if vv["excludes_zero"] else "ns")
                                      for kk, vv in d.items()}
                          for s, d in v["per_seed_ga_vs_controls"].items()})
        else:
            print(k, "best_t:", v["best_t"], v["threshbest_vs_never"])


if __name__ == "__main__":
    raise SystemExit(main())
