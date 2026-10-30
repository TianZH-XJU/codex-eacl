#!/usr/bin/env python3
"""CodeXGLUE defect split audit + project-grouped manifest (offline).

Reads function.json + official index lists. Reports: official sizes, label
balance, exact-duplicate functions (within/across splits), and project overlap
across train/valid/test. Writes a manifest; does NOT create new splits yet
(adapter Gate 0 first). No Laya calls.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw/codexglue_defect")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", default="results/codexglue_defect/split_audit.json")
    args = ap.parse_args()

    raw = Path(args.raw_dir)
    js_all = json.loads((raw / "function.json").read_text(encoding="utf-8"))
    splits = {}
    for split, fn in (("train", "train.txt"), ("valid", "valid.txt"), ("test", "test.txt")):
        idx = [int(x) for x in (raw / fn).read_text(encoding="utf-8").split()]
        splits[split] = [{"idx": i, **js_all[i]} for i in idx]

    projects = defaultdict(lambda: defaultdict(int))
    func_first_seen = {}
    dup_across, dup_within = 0, 0
    for split, rows in splits.items():
        for r in rows:
            projects[r.get("project", "?")][split] += 1
            if r["func"] in func_first_seen:
                if func_first_seen[r["func"]] == split:
                    dup_within += 1
                else:
                    dup_across += 1
            else:
                func_first_seen[r["func"]] = split

    cross_projects = [p for p, d in projects.items() if len(d) > 1]
    audit = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sizes": {k: len(v) for k, v in splits.items()},
        "label_balance": {k: dict(Counter(r["target"] for r in v)) for k, v in splits.items()},
        "exact_duplicate_func_strings_total": sum(1 for c in Counter(r["func"] for rows in splits.values() for r in rows).values() if c > 1),
        "dup_rows_within_same_split": dup_within,
        "dup_rows_across_splits": dup_across,
        "n_projects": len(projects),
        "projects_spanning_multiple_splits": len(cross_projects),
        "cross_split_projects": sorted(cross_projects)[:50],
        "note": "official split kept for Gate 0; project-deduped controller split decided after adapter check",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: audit[k] for k in ("sizes", "label_balance", "dup_rows_within_same_split",
          "dup_rows_across_splits", "n_projects", "projects_spanning_multiple_splits")}, indent=2))
    print(f"Wrote audit: {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
