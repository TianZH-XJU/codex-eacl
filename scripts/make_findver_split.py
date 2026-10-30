#!/usr/bin/env python3
"""FinDVer report-grouped controller train/dev carve-out (offline).

Carves train/dev ONLY from testmini, grouped by report (stratified by
subset x label), so no report crosses train/dev. Test split untouched.
No Laya calls.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--findver-root", default="external/FinDVer")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--dev-report-fraction", type=float, default=0.2)
    ap.add_argument("--out", default="results/findver/controller_split_manifest.json")
    args = ap.parse_args()

    rows = json.loads(Path(args.findver_root, "data", "testmini.json").read_text(encoding="utf-8"))
    by_report = defaultdict(list)
    for r in rows:
        by_report[r["report"]].append(r["example_id"])

    def strat_key(ids):
        subs = sorted({r["subset"] for r in rows if r["example_id"] in set(ids)})
        labs = sorted({r["entailment_label"] for r in rows if r["example_id"] in set(ids)})
        return (tuple(subs), tuple(labs))

    strata = defaultdict(list)
    for rep, ids in by_report.items():
        strata[strat_key(ids)].append(rep)

    rng = random.Random(args.seed)
    dev_reports, train_reports = set(), set()
    for key in sorted(strata, key=str):
        reps = list(strata[key])
        rng.shuffle(reps)
        n_dev = max(1, int(round(len(reps) * args.dev_report_fraction)))
        dev_reports.update(reps[:n_dev])
        train_reports.update(reps[n_dev:])

    assert dev_reports.isdisjoint(train_reports)
    by_id = {r["example_id"]: r for r in rows}
    dev_ids = sorted(i for rep in dev_reports for i in by_report[rep])
    train_ids = sorted(i for rep in train_reports for i in by_report[rep])
    assert len(dev_ids) + len(train_ids) == len(rows)

    from collections import Counter
    def dist(ids):
        return {"n": len(ids),
                "subsets": dict(Counter(by_id[i]["subset"] for i in ids)),
                "labels": {str(k): v for k, v in Counter(by_id[i]["entailment_label"] for i in ids).items()}}
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": "testmini only; test split untouched",
        "seed": args.seed, "dev_report_fraction": args.dev_report_fraction,
        "rule": "group by report; stratified by (subsets, labels) of report",
        "n_reports": {"train": len(train_reports), "dev": len(dev_reports)},
        "train": dist(train_ids), "dev": dist(dev_ids),
        "train_example_ids": train_ids, "dev_example_ids": dev_ids,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"n_reports": manifest["n_reports"], "train": manifest["train"],
                      "dev": manifest["dev"]}, indent=2))
    print(f"Wrote manifest: {out}")


if __name__ == "__main__":
    raise SystemExit(main())
