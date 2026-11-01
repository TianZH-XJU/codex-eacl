#!/usr/bin/env python3
"""P0-3: paired group-bootstrap CIs for macro-F1 differences (offline).

Background: test_evo_significance.paired_ci bootstraps per-row *accuracy*
(correct/incorrect), so its CIs are accuracy-difference CIs, not macro-F1
CIs — even when the fitness scalar was macro-F1 driven. This script computes
paired macro-F1-difference CIs from the same caches: resample groups with
replacement, rebuild the TP/FP/TN/FN contingency from (correct, gold) pairs
for each policy, take macro-F1(A) - macro-F1(B).

Comparisons (frozen train-selected GA bests, no re-search):
  GA-best vs fixed switch-only, and GA-best vs matched random-best,
  on train (=search), dev (=selection_dev) and confirm pools.
Zero new model calls.
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
sys.path.insert(0, str(ROOT / "scripts"))

from evolution import CascadeGenome, decide  # noqa: E402

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
    return d["rows"]


def correctness(g, rows):
    """Per-row (group_id, correct, key, gold) for a CascadeGenome."""
    out = []
    for r in rows:
        if r.get("noul_p") is None:
            out.append((r["group_id"], 0, r["key"], r["gold"]))
            continue
        ok, _ = decide(g, r["noul_p"], bool(r.get("switch_correct", False)), r["gold"])
        out.append((r["group_id"], int(ok), r["key"], r["gold"]))
    return out


def macro_f1(pairs):
    """Macro-F1 from (group_id, correct, key, gold) tuples.

    Predicted label is defined by correctness accounting identities:
    correct&gold=TP, correct&~gold=TN, wrong&gold=FN, wrong&~gold=FP.
    """
    tp = sum(1 for _, c, _, g in pairs if c and g)
    tn = sum(1 for _, c, _, g in pairs if c and not g)
    fp = sum(1 for _, c, _, g in pairs if not c and not g)
    fn = sum(1 for _, c, _, g in pairs if not c and g)
    f1pos = 2 * tp / max(2 * tp + fp + fn, 1)
    f1neg = 2 * tn / max(2 * tn + fp + fn, 1)
    return (f1pos + f1neg) / 2


def paired_macro_f1_ci(a_tuples, b_tuples, n=5000):
    """Paired group-bootstrap CI of macroF1(b) - macroF1(a)."""
    if len(a_tuples) != len(b_tuples):
        raise SystemExit("length mismatch")
    ka = [t[2] for t in a_tuples]
    kb = [t[2] for t in b_tuples]
    if ka != kb:
        raise SystemExit("row-id order mismatch (unpaired comparison refused)")
    ga = [t[3] for t in a_tuples]
    gb = [t[3] for t in b_tuples]
    if ga != gb:
        raise SystemExit("gold mismatch (unpaired comparison refused)")
    groups = sorted({t[0] for t in a_tuples})
    da = {g: [t for t in a_tuples if t[0] == g] for g in groups}
    db = {g: [t for t in b_tuples if t[0] == g] for g in groups}
    flat = lambda d, s: [t for g in s for t in d[g]]
    est = macro_f1(flat(db, groups)) - macro_f1(flat(da, groups))
    boots = []
    for _ in range(n):
        s = rng.choice(groups, size=len(groups), replace=True)
        boots.append(macro_f1(flat(db, s)) - macro_f1(flat(da, s)))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"est": round(float(est), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "excludes_zero": bool(lo > 0 or hi < 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=5000)
    args = ap.parse_args()
    try:
        sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                             check=True, capture_output=True, text=True).stdout.strip()
    except Exception:
        sha = None
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "repo_commit": sha, "protocol_version": "v4.1",
              "method": ("paired group-bootstrap (5000 resamples) macro-F1 difference; "
                         "CIs are macro-F1 CIs, unlike test_evo_significance accuracy CIs"),
              "run_manifest": {"argv": sys.argv[1:]},
              "tasks": {}}
    for key, (tr_p, dev_p, conf_p) in TASKS.items():
        evo = json.loads((ROOT / f"results/v4_1/evolution/evo_train_{key}.json").read_text(encoding="utf-8"))
        pools = {"train": load(tr_p), "dev": load(dev_p), "confirm": load(conf_p)}
        per_seed = {}
        for s in evo["seeds"]:
            gb = CascadeGenome(**evo["per_seed"][str(s)]["ga"]["best"])
            rb = CascadeGenome(**evo["per_seed"][str(s)]["random"]["best"])
            sw = CascadeGenome(lo=0.0, hi=1.0, use_second=1)
            cell = {}
            for tag, rows in pools.items():
                ci_sw = paired_macro_f1_ci(correctness(sw, rows), correctness(gb, rows), n=args.reps)
                ci_sw["comparison"] = "ga_minus_switchonly_macro_f1"
                ci_rnd = paired_macro_f1_ci(correctness(rb, rows), correctness(gb, rows), n=args.reps)
                ci_rnd["comparison"] = "ga_minus_random_macro_f1"
                cell[tag] = {"ga_minus_switchonly": ci_sw, "ga_minus_random": ci_rnd, "n": len(rows)}
            per_seed[str(s)] = cell
        report["tasks"][key] = {"per_seed": per_seed,
                                "source": f"evo_train_{key}.json frozen train bests"}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for k, v in report["tasks"].items():
        for s, ps in v["per_seed"].items():
            d, c = ps["dev"]["ga_minus_switchonly"], ps["confirm"]["ga_minus_switchonly"]
            print(k, s, "dev:", d["est"], d["ci95"], "SIG" if d["excludes_zero"] else "ns",
                  "| conf:", c["est"], c["ci95"], "SIG" if c["excludes_zero"] else "ns")


if __name__ == "__main__":
    raise SystemExit(main())
