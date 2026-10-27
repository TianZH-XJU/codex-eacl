#!/usr/bin/env python3
"""Problem-grouped train/dev/test split manifest for ProcessBench.

Groups by problem text so trajectories of the same problem never cross
splits. Stratified by (source file, error-vs-all-correct). Test stays locked.
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

from prepare_processbench import (  # noqa: E402
    FILES, PROCESSBENCH_GIT_COMMIT, REPO_ID, REVISION, validate_row,
)

SPLIT_FRACTIONS = {"train": 0.6, "dev": 0.2, "test": 0.2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw/processbench")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", default="results/processbench/split_manifest.json")
    args = ap.parse_args()

    problems = {}  # problem text -> {"rows": [(file, id, label)], "source": file}
    for filename in FILES:
        rows = json.loads(Path(args.raw_dir, filename).read_text(encoding="utf-8"))
        for row in rows:
            ok, _ = validate_row(row)
            assert ok, row.get("id")
            p = problems.setdefault(row["problem"], {"rows": [], "source": filename})
            p["rows"].append({"file": filename, "id": row["id"], "label": row["label"]})
    # A problem appearing in two files would break source stratification; check.
    by_id = {}
    for problem, info in problems.items():
        for r in info["rows"]:
            assert r["id"] not in by_id, f"duplicate id {r['id']}"
            by_id[r["id"]] = problem

    strata = defaultdict(list)
    for problem, info in problems.items():
        has_error = any(r["label"] != -1 for r in info["rows"])
        strata[(info["source"], has_error)].append(problem)

    rng = random.Random(args.seed)
    splits = {"train": [], "dev": [], "test": []}
    for key in sorted(strata):
        items = list(strata[key])
        rng.shuffle(items)
        n = len(items)
        n_test = int(round(n * SPLIT_FRACTIONS["test"]))
        n_dev = int(round(n * SPLIT_FRACTIONS["dev"]))
        splits["test"].extend(items[:n_test])
        splits["dev"].extend(items[n_test:n_test + n_dev])
        splits["train"].extend(items[n_test + n_dev:])

    # verify disjointness and coverage
    sets = {k: set(v) for k, v in splits.items()}
    assert sets["train"].isdisjoint(sets["dev"]) and sets["train"].isdisjoint(sets["test"]) and sets["dev"].isdisjoint(sets["test"])
    assert sum(len(v) for v in sets.values()) == len(problems)

    def rows_of(problem_list):
        out = []
        for p in problem_list:
            out.extend(problems[p]["rows"])
        return out

    row_splits = {k: rows_of(v) for k, v in splits.items()}
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repo_id": REPO_ID, "revision": REVISION,
        "processbench_git_commit": PROCESSBENCH_GIT_COMMIT,
        "seed": args.seed, "fractions": SPLIT_FRACTIONS,
        "rule": "group by problem text; stratified by (source file, error-vs-all-correct)",
        "n_problems": {k: len(v) for k, v in splits.items()},
        "n_rows": {k: len(v) for k, v in row_splits.items()},
        "problems": splits,
        "rows": row_splits,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"n_problems": manifest["n_problems"], "n_rows": manifest["n_rows"]}, indent=2))
    print(f"Wrote manifest: {out}")


if __name__ == "__main__":
    raise SystemExit(main())
