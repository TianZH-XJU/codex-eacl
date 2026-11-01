#!/usr/bin/env python3
"""Recompute cited CIs from committed row files (offline audit evidence).

Writes results/v4_1/manifests/recomputed_cis.json with paired bootstrap CIs
for the A1 intent comparisons and restricted-vs-cached agreement rates for the
surrogate three-way validation. All inputs are committed files; the script is
deterministic (seed 17, 5000 reps).
"""
from __future__ import annotations

import glob
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.RandomState(17)


def git_sha(path):
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                              check=True, capture_output=True, text=True).stdout.strip()
    except Exception:
        return None


def paired_ci(diffs, n=5000):
    diffs = np.array(diffs, dtype=float)
    boots = [float(np.mean(rng.choice(diffs, size=len(diffs), replace=True))) for _ in range(n)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"est": round(float(np.mean(diffs)), 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)]}


def main():
    out = {"created_utc": datetime.now(timezone.utc).isoformat(),
           "method": "grouped rows as stored (intent rows carry group_id); paired bootstrap, seed 17, 5000 reps",
           "project_git_sha": git_sha(ROOT), "a1_paired": {}, "surrogate": {}}
    for name in ("paired_clinc150", "paired_banking77"):
        d = json.loads((ROOT / "results/v4/native" / f"{name}.json").read_text(encoding="utf-8"))
        rows = d["rows"]
        la = [int(r["laya_correct"]) for r in rows]
        va = [int(r["vela_correct"]) for r in rows]
        out["a1_paired"][name] = {"n": len(rows),
                                  "laya_acc": round(sum(la) / len(la), 4),
                                  "vela_acc": round(sum(va) / len(va), 4),
                                  "vela_minus_laya": paired_ci([v - l for l, v in zip(la, va)])}
    for f in sorted(glob.glob(str(ROOT / "results/v2/menu/surrogate_*.json"))):
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        rs = [r for r in d["results"] if r["variant"].startswith("tfidf") and r["error"] is None]
        out["surrogate"][Path(f).stem] = {
            "agreement": round(sum(r["agree_real_vs_cached"] for r in rs) / len(rs), 4),
            "n": len(rs), "bar": 0.9,
            "verdict": "rejected" if sum(r["agree_real_vs_cached"] for r in rs) / len(rs) < 0.9 else "pass"}
    dest = ROOT / "results/v4_1/manifests/recomputed_cis.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("a1_paired", "surrogate")}, indent=2)[:1500])


if __name__ == "__main__":
    raise SystemExit(main())
