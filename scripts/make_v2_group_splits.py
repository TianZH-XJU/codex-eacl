#!/usr/bin/env python3
"""A0 group manifests for the post-gate diagnostic track (offline, no model calls).

- CodeXGLUE: official train rows grouped by canonical function hash into
  search_train/search_dev/confirm; cross-split duplicate rows listed as excluded.
- FinDVer: testmini reports NOT in official test -> search_train/search_dev/confirm.
- ProcessBench: train problems minus all history-pilot problem groups ->
  search_train/search_dev/confirm.
Locked test splits are never read for labels (only testmini/test report-ID overlap,
already audited, is reused for exclusion).
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from group_oof import function_group_id, problem_group_id  # noqa: E402


def load_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def stratified_group_split(groups, seed, fractions=(0.7, 0.15, 0.15), strat_key=None):
    """groups: {group_id: info}. Deterministic group-disjoint split."""
    by_stratum = defaultdict(list)
    for gid, info in groups.items():
        by_stratum[strat_key(info) if strat_key else ""].append(gid)
    rng = random.Random(seed)
    out = {"search_train": [], "search_dev": [], "confirm": []}
    for stratum in sorted(by_stratum):
        items = list(by_stratum[stratum])
        rng.shuffle(items)
        n = len(items)
        n_dev = max(1, int(round(n * fractions[1]))) if n >= 3 else 0
        n_conf = max(1, int(round(n * fractions[2]))) if n >= 3 else 0
        out["confirm"].extend(items[:n_conf])
        out["search_dev"].extend(items[n_conf:n_conf + n_dev])
        out["search_train"].extend(items[n_conf + n_dev:])
    return out


def code_manifest(args):
    raw = Path(args.code_raw)
    js_all = json.loads((raw / "function.json").read_text(encoding="utf-8"))
    official = {}
    for split, fn in (("train", "train.txt"), ("valid", "valid.txt"), ("test", "test.txt")):
        official[split] = [int(x) for x in (raw / fn).read_text(encoding="utf-8").split()]
    groups = {}
    for idx in official["train"]:
        g = function_group_id(js_all[idx]["func"])
        groups.setdefault(g, {"idx": [], "labels": []})
        groups[g]["idx"].append(idx)
        groups[g]["labels"].append(js_all[idx]["target"])
    # cross-split duplicates (in valid/test): excluded from any new search pool
    train_funcs = {function_group_id(js_all[i]["func"]) for i in official["train"]}
    leaked = sorted({i for s in ("valid", "test") for i in official[s]
                     if function_group_id(js_all[i]["func"]) in train_funcs})
    splits = stratified_group_split(groups, args.seed,
                                     strat_key=lambda info: sum(info["labels"]) > 0)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "rule": "official-train only, grouped by canonical function hash; seed 17",
                "n_groups": len(groups), "excluded_cross_split_leaked_idx": leaked,
                "members": {g: sorted(info["idx"]) for g, info in groups.items()},
                "splits": {k: sorted(v) for k, v in splits.items()},
                "split_sizes_groups": {k: len(v) for k, v in splits.items()}}
    out = Path(args.out_dir, "codexglue_function_groups.json")
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def findver_manifest(args):
    root = Path(args.findver_root)
    testmini = json.loads((root / "data/testmini.json").read_text(encoding="utf-8"))
    test_ids = {r["report"] for r in json.loads((root / "data/test.json").read_text(encoding="utf-8"))}
    clean = [r for r in testmini if r["report"] not in test_ids]
    groups = {}
    for r in clean:
        groups.setdefault(r["report"], {"ids": [], "subsets": set(), "labels": []})
        groups[r["report"]]["ids"].append(r["example_id"])
        groups[r["report"]]["subsets"].add(r["subset"])
        groups[r["report"]]["labels"].append(bool(r["entailment_label"]))
    splits = stratified_group_split(
        groups, args.seed,
        strat_key=lambda info: (tuple(sorted(info["subsets"])), sum(info["labels"]) > 0))
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "rule": "testmini reports disjoint from official test reports; grouped by report; seed 17",
                "n_testmini_reports_total": len({r["report"] for r in testmini}),
                "n_excluded_shared_with_official_test": len({r["report"] for r in testmini} & test_ids),
                "n_clean_reports": len(groups),
                "splits": {k: sorted(v) for k, v in splits.items()},
                "split_sizes_reports": {k: len(v) for k, v in splits.items()}}
    out = Path(args.out_dir, "findver_report_groups.json")
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def history_pilot_row_ids():
    """All ProcessBench row ids ever touched by pilots/smoke (offline, from summaries)."""
    res = ROOT / "results/processbench"
    ids = set()
    for name in ("pilot_cache.summary.json", "sensitivity_pilot.summary.json",
                 "sensitivity_devpilot.summary.json"):
        p = res / name
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            ids.update(d.get("sample_ids", []))
    # smoke scripts used gsm8k rows 0..29 + long-sample math-1 + a few.
    ids.update(f"gsm8k-{i}" for i in range(30))
    ids.add("math-1")
    return ids


def processbench_manifest(args):
    split = json.loads(Path(args.pb_split).read_text(encoding="utf-8"))
    raw = {}
    for fn in ("gsm8k.json", "math.json", "olympiadbench.json", "omnimath.json"):
        for r in json.loads(Path(args.pb_raw, fn).read_text(encoding="utf-8")):
            raw[r["id"]] = (fn, r["problem"], r["label"])
    touched = history_pilot_row_ids()
    touched_groups = set()
    for rid in touched:
        if rid in raw:
            touched_groups.add(problem_group_id(raw[rid][1]))
    groups = {}
    for item in split["rows"]["train"]:
        fn, problem, label = raw[item["id"]]
        g = problem_group_id(problem)
        if g in touched_groups:
            continue
        groups.setdefault(g, {"rows": [], "err": False, "source": fn})
        groups[g]["rows"].append(item["id"])
        groups[g]["err"] = groups[g]["err"] or (label != -1)
    splits = stratified_group_split(groups, args.seed,
                                     strat_key=lambda info: (info["source"], info["err"]))
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "rule": "train problems minus all history-pilot problem groups; grouped by problem hash; seed 17",
                "n_history_touched_row_ids": len(touched),
                "n_history_touched_groups": len(touched_groups),
                "n_remaining_groups": len(groups),
                "splits": {k: sorted(v) for k, v in splits.items()},
                "split_sizes_groups": {k: len(v) for k, v in splits.items()}}
    out = Path(args.out_dir, "processbench_problem_groups.json")
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code-raw", default="data/raw/codexglue_defect")
    ap.add_argument("--findver-root", default="external/FinDVer")
    ap.add_argument("--pb-split", default="results/processbench/split_manifest.json")
    ap.add_argument("--pb-raw", default="data/raw/processbench")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out-dir", default="results/v2/data_manifest")
    args = ap.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    c = code_manifest(args)
    f = findver_manifest(args)
    p = processbench_manifest(args)
    print(json.dumps({
        "code_groups": c["n_groups"], "code_leaked_excluded": len(c["excluded_cross_split_leaked_idx"]),
        "findver_clean_reports": f["n_clean_reports"],
        "processbench_remaining_groups": p["n_remaining_groups"]}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
