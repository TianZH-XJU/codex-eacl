#!/usr/bin/env python3
"""Offline paired full-vs-menu analysis on v4 intent dev runs (no model calls)."""
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.RandomState(17)


def load(p):
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def paired(a, b):
    assert len(a) == len(b)
    ca = np.array([r["correct"] for r in a], dtype=float)
    cb = np.array([r["correct"] for r in b], dtype=float)
    d = cb - ca
    means = [float(np.mean(rng.choice(d, size=len(d), replace=True))) for _ in range(5000)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"n": len(a), "first_acc": round(float(ca.mean()), 4), "second_acc": round(float(cb.mean()), 4),
            "diff": round(float(d.mean()), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "both_ok": int(((ca == 1) & (cb == 1)).sum()), "both_bad": int(((ca == 0) & (cb == 0)).sum()),
            "first_only": int(((ca == 1) & (cb == 0)).sum()), "second_only": int(((ca == 0) & (cb == 1)).sum())}


def by_key(rows):
    return {r["row_id"]: r for r in rows}


def main():
    out = {}
    for task in ("banking77", "clinc150"):
        for model in ("laya", "vela"):
            full = load(ROOT / f"results/v2/native/{model}_{task}_fulldev.jsonl")
            menu = load(ROOT / f"results/v2/native/{model}_{task}_menu.jsonl")
            fm, mm = by_key(full), by_key(menu)
            common = sorted(set(fm) & set(mm))
            fa = [fm[k] for k in common]
            ma = [mm[k] for k in common]
            lat = lambda rs: round(float(np.mean([r["cumulative_latency_ms"] for r in rs])), 1)
            out[f"{task}_{model}_menu_vs_full"] = {**paired(fa, ma),
                "mean_ms_full": lat(fa), "mean_ms_menu": lat(ma),
                "recall_hit": round(sum(r.get("menu_recall_hit", 0) for r in ma) / len(ma), 4)}
    # cross-model on full and on menu
    for task in ("banking77", "clinc150"):
        for rep, pat in (("full", "*_fulldev.jsonl"), ("menu", "*_menu.jsonl")):
            files = {"laya": None, "vela": None}
            for f in glob.glob(str(ROOT / f"results/v2/native/*_{task}_{rep}*.jsonl")):
                if f.split("_")[0].endswith(("laya", "vela")) or "/laya_" in f.replace("\\", "/"):
                    files["laya" if "laya_" in Path(f).name else "vela"] = f
            la = load(files["laya"]) if files["laya"] else []
            va = load(files["vela"]) if files["vela"] else []
            lm = by_key(la)
            vm = by_key(va)
            common = sorted(set(lm) & set(vm))
            if common:
                out[f"{task}_laya_vs_vela_{rep}"] = paired([lm[k] for k in common], [vm[k] for k in common])
                out[f"{task}_laya_vs_vela_{rep}"]["mean_ms"] = {
                    "laya": round(float(np.mean([lm[k]["cumulative_latency_ms"] for k in common])), 1),
                    "vela": round(float(np.mean([vm[k]["cumulative_latency_ms"] for k in common])), 1)}
    Path("results/v2/native/paired_menu_full.json").write_text(json.dumps(out, indent=2) + "\n")
    for k, v in out.items():
        print(k, "n=", v.get("n"), "first=", v.get("first_acc"), "second=", v.get("second_acc"),
              "diff=", v.get("diff"), "ci=", v.get("ci95"))


if __name__ == "__main__":
    raise SystemExit(main())
