#!/usr/bin/env python3
"""Stage 3: paired Laya-vs-Vela default comparison on identical rows (offline).

Recomputes correctness from raw predictions + pilot gold (explicit
normalization) rather than trusting any stored flags. Paired accuracy,
discordant counts, latency/cost side-by-side, per-split tables.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def norm(s):
    return str(s or "").replace(" ", "_").lower()


def load_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--laya-cache", required=True)
    ap.add_argument("--vela-baseline", required=True)
    ap.add_argument("--laya-strategy", default="single")
    ap.add_argument("--vela-rep", default=None,
                    help="Vela rep to compare (default: choice for processbench, noul otherwise)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    laya = [r for r in load_jsonl(args.laya_cache) if r["strategy"] == args.laya_strategy]
    vela_all = json.loads(Path(args.vela_baseline).read_text(encoding="utf-8"))["results"]
    rep = args.vela_rep or ("choice" if any(r["rep"] == "choice" for r in vela_all) else "noul")
    vela = [r for r in vela_all if r["rep"] == rep]
    v_by_key = {(r["split"], r["position"]): r for r in vela}

    rows, missing = [], 0
    for r in laya:
        key = (r["split"], r.get("position"))
        v = v_by_key.get(key)
        if v is None:
            missing += 1
            continue
        gold = norm(r.get("gold_rendered", "") or r.get("gold_render", ""))
        l_ok = norm(r.get("choice")) == gold
        v_ok = norm(v.get("pred")) == gold if v.get("rep") != "noul" else ((v.get("pred") == "pos") == (gold in ("vulnerable", "supported")))
        rows.append({"key": key, "row_id": r.get("row_id"), "gold": gold,
                     "laya_pred": r.get("choice"), "laya_correct": bool(l_ok),
                     "laya_ms": r.get("cumulative_latency_ms"), "laya_calls": r.get("cumulative_calls"),
                     "vela_pred": v.get("pred"), "vela_correct": bool(v_ok),
                     "vela_ms": v.get("latency_ms"), "vela_usage": v.get("usage")})

    n = len(rows)
    l_acc = sum(r["laya_correct"] for r in rows) / n if n else 0.0
    v_acc = sum(r["vela_correct"] for r in rows) / n if n else 0.0
    both_ok = sum(r["laya_correct"] and r["vela_correct"] for r in rows)
    both_bad = sum((not r["laya_correct"]) and (not r["vela_correct"]) for r in rows)
    l_only = sum(r["laya_correct"] and (not r["vela_correct"]) for r in rows)
    v_only = sum((not r["laya_correct"]) and r["vela_correct"] for r in rows)
    out = {
        "laya_cache": args.laya_cache, "vela_baseline": args.vela_baseline,
        "laya_strategy": args.laya_strategy, "vela_rep": rep,
        "n_paired": n, "missing_vela_rows": missing,
        "laya_accuracy": round(l_acc, 4), "vela_accuracy": round(v_acc, 4),
        "accuracy_diff_vela_minus_laya": round(v_acc - l_acc, 4),
        "both_correct": both_ok, "both_wrong": both_bad,
        "laya_only": l_only, "vela_only": v_only,
        "mean_latency_ms": {
            "laya": round(sum(r["laya_ms"] or 0 for r in rows) / n, 1) if n else None,
            "vela": round(sum(r["vela_ms"] or 0 for r in rows) / n, 1) if n else None},
        "rows": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("n_paired", "laya_accuracy", "vela_accuracy",
          "accuracy_diff_vela_minus_laya", "both_correct", "both_wrong",
          "laya_only", "vela_only", "mean_latency_ms")}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
