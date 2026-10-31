#!/usr/bin/env python3
"""Build the evaluated configuration bank from committed results (offline).

Every (task, model, rep, config) measured so far becomes a bank entry with
quality, cost, flips and group stats. No new inference. Optimizer selection
regret against this bank is therefore zero by construction for covered cells;
the finding is whether the bank contains anything worth selecting.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v2/evolutionary_bank.json"


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    bank = []
    # A2 schema ablations (semantic/legacy/choice).
    alaya = sorted(glob.glob(str(ROOT / "results/v2/schema_ablation/laya_*.json")))
    avela = sorted(glob.glob(str(ROOT / "results/v2/schema_ablation/vela_*.json")))
    for f in alaya + avela:
        if Path(f).stem.startswith(("laya_code", "vela_code")):
            task = "code_defect"
        elif Path(f).stem.startswith(("laya_findver", "vela_findver")):
            task = "findver"
        else:
            continue
        if f.endswith(("_h96.json", "_max384.json", "_fixed.json")):
            continue
        model = "laya" if Path(f).stem.startswith("laya_") else "vela"
        d = load(f)
        for rep in sorted({r["rep"] for r in d["results"]}):
            sub = [r for r in d["results"] if r["rep"] == rep and r["error"] is None]
            if not sub:
                continue
            acc = sum(r["correct"] for r in sub) / len(sub)
            lat = [r["latency_ms"] for r in sub]
            bank.append({"entry": f"{task}|{model}|{rep}|default-budget", "task": task,
                         "model": model, "rep": rep, "config": "native-default", "n": len(sub),
                         "accuracy": round(acc, 4),
                         "mean_latency_ms": round(sum(lat) / len(lat), 1)})

    # A2 fixed-parser reruns.
    for f, model in [("laya_findver_choice_fixed.json", "laya"), ("vela_findver_choice_fixed.json", "vela")]:
        d = load(ROOT / "results/v2/schema_ablation" / f)
        sub = [r for r in d["results"] if r["error"] is None]
        acc = sum(r["correct"] for r in sub) / len(sub)
        lat = [r["latency_ms"] for r in sub]
        bank.append({"entry": f"findver|{model}|choice-fixed-parser|default-budget", "task": "findver",
                     "model": model, "rep": "choice", "config": "native-default-fixed-parser", "n": len(sub),
                     "accuracy": round(acc, 4), "mean_latency_ms": round(sum(lat) / len(lat), 1)})

    # OFAT budget variants (Laya semantic Noul only).
    for f in sorted(glob.glob(str(ROOT / "results/v2/schema_ablation/*_h96.json"))) + \
             sorted(glob.glob(str(ROOT / "results/v2/schema_ablation/*_max384.json"))):
        stem = Path(f).stem
        if stem.startswith("laya_code"):
            task = "code_defect"
        elif stem.startswith("laya_findver"):
            task = "findver"
        else:
            continue
        cfg = "head_max_len=96" if stem.endswith("h96") else "max_len=384"
        d = load(f)
        sub = [r for r in d["results"] if r["rep"] == "semantic_noul" and r["error"] is None]
        acc = sum(r["correct"] for r in sub) / len(sub) if sub else None
        lat = [r["latency_ms"] for r in sub]
        bank.append({"entry": f"{task}|laya|semantic_noul|{cfg}", "task": task, "model": "laya",
                     "rep": "semantic_noul", "config": cfg, "n": len(sub),
                     "accuracy": round(acc, 4) if acc is not None else None,
                     "mean_latency_ms": round(sum(lat) / len(lat), 1) if lat else None})

    out = {"created": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
           "n_entries": len(bank),
           "note": "every entry measured with real model calls; no surrogate fitness anywhere",
           "entries": bank}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    for e in bank:
        print(f"{e['entry']:55s} acc={e['accuracy']} ms={e['mean_latency_ms']} n={e['n']}")


if __name__ == "__main__":
    raise SystemExit(main())
