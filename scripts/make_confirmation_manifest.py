#!/usr/bin/env python3
"""Freeze a 77-sample confirmation set: 1 per class from official-train rows
unused by any controller train/dev subset so far. No Laya calls; index bookkeeping only."""
from __future__ import annotations

import argparse
import json
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external" / "laya" / "research" / "benchmarks" / "tournament"))
import benchmark as bench  # noqa: E402


def git_sha(cwd):
    try:
        return subprocess.run(["git", "-C", str(cwd), "rev-parse", "HEAD"],
                              check=True, capture_output=True, text=True).stdout.strip()
    except Exception:
        return None


def git_dirty(cwd):
    try:
        return bool(subprocess.run(["git", "-C", str(cwd), "status", "--porcelain"],
                                   check=True, capture_output=True, text=True).stdout.strip())
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-manifest", default="results/splits/banking77_train_dev_manifest.json")
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--out", default="results/splits/banking77_confirmation_manifest.json")
    args = ap.parse_args()

    split = json.loads(Path(args.split_manifest).read_text(encoding="utf-8"))
    used = set(split["train_indices_all"]) | set(split["dev_indices"])
    assert set(split["train_indices_all"]).isdisjoint(set(split["dev_indices"]))

    load, _ = bench.DATASETS["banking77"]
    names, rows = load(split.get("source_split", "train"))
    by_label = {}
    for i, (_text, gold) in enumerate(rows):
        by_label.setdefault(gold, []).append(i)

    rng = random.Random(args.seed)
    confirmation = []
    per_class = {}
    for label in sorted(by_label):
        candidates = [i for i in by_label[label] if i not in used]
        assert candidates, f"no unused rows for class {label!r}"
        pick = rng.choice(candidates)
        confirmation.append(pick)
        per_class[label] = {"confirmation_index": pick, "n_unused_candidates": len(candidates)}

    assert len(confirmation) == len(by_label)
    assert not (set(confirmation) & used)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "banking77",
        "source_split": split.get("source_split", "train"),
        "dataset_revision": split.get("dataset_revision"),
        "seed": args.seed,
        "rule": "1 per class, uniform choice among official-train rows unused by train_all/dev subsets",
        "split_manifest": args.split_manifest,
        "n_classes": len(by_label),
        "n_confirmation": len(confirmation),
        "confirmation_indices": sorted(confirmation),
        "per_class": per_class,
        "project_git_sha": git_sha(ROOT),
        "working_tree_dirty_at_freeze": git_dirty(ROOT),
        "laya_commit": git_sha(ROOT / "external" / "laya"),
        "model_revision": "7b928d828b7b0e022f929d9bd2e44165aa270148",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("n_classes", "n_confirmation", "seed",
          "project_git_sha", "working_tree_dirty_at_freeze")}, indent=2))
    print(f"Wrote manifest: {out}")


if __name__ == "__main__":
    raise SystemExit(main())
