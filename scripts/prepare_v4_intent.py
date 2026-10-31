#!/usr/bin/env python3
"""V4 intent-task prep: CLINC150 + BANKING77 inspect-only with group splits.

- CLINC150 (clinc/clinc_oos, plus config, pinned revision): label audit incl OOS,
  exact-duplicate utterance groups, group-safe search/dev/confirm from train.
- BANKING77 (mteb/banking77, pinned revision): new search/dev/test group manifest
  from official train only (old dev/confirmation never reused as test).
No model calls. No test labels used for any selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from group_oof import normalize_text, sha256_hex  # noqa: E402

DATASETS = {
    "clinc150": {"repo": "clinc/clinc_oos", "revision": "155b9c710419136e17307b80d0a13e68cd46b4ec",
                 "config": "plus", "config_dir": "data/raw/clinc150"},
    "banking77": {"repo": "mteb/banking77", "revision": "18072d2685ea682290f7b8924d94c62acc19c0b2",
                  "config": None, "config_dir": "data/raw/banking77"},
}


def load_rows(ds, split):
    import pyarrow.parquet as pq
    spec = DATASETS[ds]
    if ds == "clinc150":
        path = Path(spec["config_dir"]) / spec["config"] / f"{split}-00000-of-00001.parquet"
        table = pq.read_table(path)
        names = json.loads(pq.read_schema(path).metadata[b"huggingface"])["info"]["features"]["intent"]["names"]
        return [{"text": r["text"], "label": names[r["intent"]]} for r in table.to_pylist()], names
    path = Path(spec["config_dir"]) / "data" / f"{split}-00000-of-00001.parquet"
    table = pq.read_table(path)
    rows = table.to_pylist()
    return [{"text": r["text"], "label": r["label_text"]} for r in rows], sorted({r["label_text"] for r in rows})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["clinc150", "banking77"])
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out-dir", default="results/v4/data_manifest")
    args = ap.parse_args()

    spec = DATASETS[args.dataset]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_split = {}
    for split in ("train", "validation", "test") if args.dataset == "clinc150" else ("train", "test"):
        try:
            rows, _names = load_rows(args.dataset, split)
        except FileNotFoundError:
            rows, _names = [], []
        per_split[split] = rows

    # group by normalized utterance
    groups = defaultdict(list)
    for split, rows in per_split.items():
        for i, r in enumerate(rows):
            groups[(normalize_text(r["text"]), r["label"])].append({"split": split, "idx": i,
                                                                    "text": r["text"], "label": r["label"]})
    spans = {g: sorted({m["split"] for m in ms}) for g, ms in groups.items()}
    cross = {g: ms for g, ms in groups.items() if len(spans[g]) > 1}

    if args.dataset == "clinc150":
        oos = sum(1 for rows in per_split.values() for r in rows if r["label"] == "oos")
    else:
        oos = 0

    # search/dev/confirm from TRAIN groups only, stratified by label presence
    train_groups = {g: ms for g, ms in groups.items() if all(m["split"] == "train" for m in ms)}
    by_label = defaultdict(list)
    for g, ms in train_groups.items():
        by_label[ms[0]["label"]].append(g)
    rng = random.Random(args.seed)
    splits = {"search_train": [], "search_dev": [], "confirm": []}
    for lab in sorted(by_label):
        items = list(by_label[lab])
        rng.shuffle(items)
        n = len(items)
        n_dev = max(1, int(round(n * 0.15))) if n >= 3 else 0
        n_conf = max(1, int(round(n * 0.15))) if n >= 3 else 0
        splits["confirm"].extend(items[:n_conf])
        splits["search_dev"].extend(items[n_conf:n_conf + n_dev])
        splits["search_train"].extend(items[n_conf + n_dev:])

    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "dataset": args.dataset, "repo": spec["repo"], "revision": spec["revision"],
                "sizes": {k: len(v) for k, v in per_split.items()},
                "label_counts_train": dict(Counter(r["label"] for r in per_split.get("train", []))),
                "n_exact_duplicate_groups": sum(1 for ms in groups.values() if len(ms) > 1),
                "n_cross_split_groups": len(cross),
                "oos_rows_total": oos,
                "search_rule": "train groups only, stratified by label, seed 17",
                "splits": {k: sorted(v) for k, v in splits.items()},
                "split_sizes_groups": {k: len(v) for k, v in splits.items()},
                "note": "utterance groups are exact-normalized-text; paraphrase clusters not merged (documented limitation)"}
    out = out_dir / f"{args.dataset}_groups.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("sizes", "n_exact_duplicate_groups", "n_cross_split_groups",
          "oos_rows_total", "split_sizes_groups")}, indent=2))
    print(f"Wrote manifest: {out}")


if __name__ == "__main__":
    raise SystemExit(main())
