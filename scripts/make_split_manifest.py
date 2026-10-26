#!/usr/bin/env python3
"""Create a fixed, stratified, mutually exclusive train/dev manifest."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external" / "laya" / "research" / "benchmarks" / "tournament"))
import benchmark as bench

DATASET_TO_REPO = {
    "banking77": "mteb/banking77",
    "clinc150": "clinc/clinc_oos",
    "massive": "SetFit/amazon_massive_intent_en-US",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="banking77", choices=sorted(bench.DATASETS))
    ap.add_argument("--source-split", default="train")
    ap.add_argument("--train-per-class", type=int, default=5)
    ap.add_argument("--dev-per-class", type=int, default=3)
    ap.add_argument("--shot-sizes", default="1,3,5")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    load, _ = bench.DATASETS[args.dataset]
    names, rows = load(args.source_split)
    # group row indices by gold label
    by_label = defaultdict(list)
    for i, (_text, gold) in enumerate(rows):
        by_label[gold].append(i)

    shot_sizes = [int(x) for x in args.shot_sizes.split(",")]
    rng = random.Random(args.seed)
    train_indices = []
    dev_indices = []
    per_class = {}
    for label in sorted(by_label):
        idxs = list(by_label[label])
        rng.shuffle(idxs)
        needed = args.train_per_class + args.dev_per_class
        if len(idxs) < needed:
            raise SystemExit(f"Class {label!r} has only {len(idxs)} rows, need {needed}")
        train_pool = idxs[:args.train_per_class]
        dev_pool = idxs[args.train_per_class:args.train_per_class + args.dev_per_class]
        train_indices.extend(train_pool)
        dev_indices.extend(dev_pool)
        per_class[label] = {
            "train_pool": train_pool,
            "dev": dev_pool,
            "nested_train": {str(k): train_pool[:k] for k in shot_sizes if k <= len(train_pool)},
        }

    # Nested shot subsets are prefixes of the same shuffled train pool.
    max_shot = max(shot_sizes)
    nested = {}
    for k in shot_sizes:
        if k <= max_shot:
            nested[str(k)] = []
    # Build nested subsets per class then flatten in class order.
    for label in sorted(by_label):
        tp = per_class[label]["train_pool"]
        for k in shot_sizes:
            if k <= len(tp):
                nested.setdefault(str(k), []).extend(tp[:k])

    train_set = set(train_indices)
    dev_set = set(dev_indices)
    assert train_set.isdisjoint(dev_set), "train/dev overlap detected"

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "source_split": args.source_split,
        "dataset_revision": bench.REVISIONS.get(DATASET_TO_REPO.get(args.dataset)),
        "seed": args.seed,
        "stratification": "group by gold label; shuffle each class with the seed; select train_per_class then dev_per_class",
        "train_per_class": args.train_per_class,
        "dev_per_class": args.dev_per_class,
        "shot_sizes": shot_sizes,
        "n_classes": len(by_label),
        "n_train_total": len(train_indices),
        "n_dev_total": len(dev_indices),
        "train_indices_all": train_indices,
        "dev_indices": dev_indices,
        "nested_train_indices": {str(k): nested[str(k)] for k in shot_sizes if str(k) in nested},
        "per_class": per_class,
        "disjoint": True,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("dataset", "n_classes", "n_train_total", "n_dev_total", "disjoint", "nested_train_indices")}, indent=2))
    print(f"Wrote manifest: {out}")


if __name__ == "__main__":
    raise SystemExit(main())
PY