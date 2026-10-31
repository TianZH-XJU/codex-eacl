#!/usr/bin/env python3
"""Offline OFAT paired analysis vs A2 semantic baselines (no model calls)."""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def main():
    base = {}
    for task, fname in (("code_defect", "laya_code.json"), ("findver", "laya_findver.json")):
        d = load(ROOT / "results/v2/schema_ablation" / fname)
        base[task] = {r["row_key"]: r for r in d["results"] if r["rep"] == "semantic_noul"}
    for f in sorted(glob.glob("results/v2/schema_ablation/laya_*_h96.json")) + \
             sorted(glob.glob("results/v2/schema_ablation/laya_*_max384.json")):
        d = load(f)
        name = Path(f).stem.replace("laya_", "")
        task = "code_defect" if "_code_" in f else "findver"
        b = base[task]
        assert set(r["row_key"] for r in d["results"] if r["rep"] == "semantic_noul") == set(b), f"row mismatch {name}"
        n = len(b)
        diffs, flips, fix, brk = [], 0, 0, 0
        for r in d["results"]:
            if r["rep"] != "semantic_noul":
                continue
            c1 = int(r["correct"])
            c0 = int(b[r["row_key"]]["correct"])
            diffs.append(c1 - c0)
            flips += int(r["pred"] != b[r["row_key"]]["pred"])
            fix += int(c1 and not c0)
            brk += int((not c1) and c0)
        rng = np.random.RandomState(17)
        means = [float(np.mean(rng.choice(diffs, size=n, replace=True))) for _ in range(5000)]
        lo, hi = np.percentile(means, [2.5, 97.5])
        print(f"{name}: n={n} base_acc={sum(int(v['correct']) for v in b.values())/n:.3f} "
              f"cfg_acc={sum(int(r['correct']) for r in d['results'] if r['rep']=='semantic_noul')/n:.3f} "
              f"flips={flips} fix={fix} break={brk} mean_delta={np.mean(diffs):+.3f} CI=[{lo:+.3f},{hi:+.3f}]")


if __name__ == "__main__":
    raise SystemExit(main())
