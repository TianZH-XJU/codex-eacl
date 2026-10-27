#!/usr/bin/env python3
"""Reserve a fresh one-example-per-class NEAT confirmation set.

This is index bookkeeping only: no Laya inference. The new manifest excludes
all initial train-pool rows, all dev rows, and the previous (already inspected)
confirmation set. Run only after NEAT's OOF configuration is locked; never use
this set for feature or hyperparameter selection.
"""
from __future__ import annotations
import argparse, hashlib, json, random, subprocess, sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def choose_confirmation_indices(rows, used_indices, seed):
    by_label = defaultdict(list)
    for i, (_text, gold) in enumerate(rows):
        by_label[str(gold)].append(i)
    used = set(map(int, used_indices))
    rng = random.Random(seed)
    selected, per_class = [], {}
    for label in sorted(by_label):
        candidates = [i for i in by_label[label] if i not in used]
        if not candidates:
            raise ValueError(f"No unused official-train row left for class {label!r}")
        idx = rng.choice(candidates)
        selected.append(idx)
        per_class[label] = {"index": idx, "n_candidates_after_exclusion": len(candidates)}
    if len(selected) != len(by_label) or (set(selected) & used):
        raise AssertionError("confirmation selection is incomplete or overlaps a used split")
    return sorted(selected), per_class

def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def git_sha(path):
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                              check=True, capture_output=True, text=True).stdout.strip()
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-manifest", default="results/splits/banking77_train_dev_manifest.json")
    ap.add_argument("--old-confirmation-manifest", default="results/splits/banking77_confirmation_manifest.json")
    ap.add_argument("--seed", type=int, default=47)
    ap.add_argument("--exclude-manifest", action="append", default=[],
                    help="Additional prior confirmation manifest to exclude (may be repeated).")
    ap.add_argument("--out", default="results/splits/banking77_neat_confirmation_manifest.json")
    args = ap.parse_args()
    split_path, old_path, out_path = [(ROOT / p).resolve() for p in
                                      (args.split_manifest, args.old_confirmation_manifest, args.out)]
    if out_path.exists():
        raise SystemExit(f"Refusing to overwrite existing manifest: {out_path}")
    for p in (split_path, old_path):
        if not p.is_file():
            raise SystemExit(f"Required manifest missing: {p}")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    old = json.loads(old_path.read_text(encoding="utf-8"))
    if split.get("dataset") != "banking77" or old.get("dataset") != "banking77":
        raise SystemExit("Both input manifests must describe BANKING77")
    if split.get("dataset_revision") != old.get("dataset_revision"):
        raise SystemExit("Dataset revision mismatch between manifests")
    used = set(map(int, split["train_indices_all"])) | set(map(int, split["dev_indices"]))
    old_conf = set(map(int, old["confirmation_indices"]))
    if used & old_conf:
        raise SystemExit("The existing confirmation set overlaps train/dev unexpectedly")
    used |= old_conf
    additional_exclusions = []
    for excluded_path_arg in args.exclude_manifest:
        excluded_path = (ROOT / excluded_path_arg).resolve()
        if excluded_path == out_path:
            raise SystemExit("Output manifest cannot also be an exclusion manifest")
        if not excluded_path.is_file():
            raise SystemExit(f"Additional exclusion manifest missing: {excluded_path}")
        excluded = json.loads(excluded_path.read_text(encoding="utf-8"))
        if excluded.get("dataset") != "banking77" or excluded.get("dataset_revision") != split.get("dataset_revision"):
            raise SystemExit(f"Additional exclusion manifest dataset/revision mismatch: {excluded_path}")
        excluded_indices = set(map(int, excluded.get("confirmation_indices", [])))
        if not excluded_indices:
            raise SystemExit(f"Additional exclusion manifest has no confirmation indices: {excluded_path}")
        used |= excluded_indices
        additional_exclusions.append({
            "path": str(excluded_path.relative_to(ROOT)),
            "sha256": sha256_file(excluded_path),
            "n_indices": len(excluded_indices),
        })
    sys.path.insert(0, str(ROOT / "external" / "laya" / "research" / "benchmarks" / "tournament"))
    import benchmark as bench
    load, _ = bench.DATASETS["banking77"]
    names, rows = load(split.get("source_split", "train"))
    chosen, per_class = choose_confirmation_indices(rows, used, args.seed)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "banking77", "source_split": split.get("source_split", "train"),
        "dataset_revision": split.get("dataset_revision"),
        "seed": args.seed, "n_classes": len(per_class), "n_confirmation": len(chosen),
        "rule": "one official-train row/class, excluding all train-pool, dev and old-confirmation indices",
        "split_manifest": str(split_path.relative_to(ROOT)),
        "split_manifest_sha256": sha256_file(split_path),
        "old_confirmation_manifest": str(old_path.relative_to(ROOT)),
        "old_confirmation_manifest_sha256": sha256_file(old_path),
        "excluded_index_counts": {
            "train_pool": len(split["train_indices_all"]),
            "dev": len(split["dev_indices"]),
            "previous_confirmation": len(old_conf),
            "additional_confirmation_manifests": len(additional_exclusions),
            "additional_confirmation_indices_total": sum(x["n_indices"] for x in additional_exclusions),
            "union": len(used),
        },
        "additional_excluded_manifests": additional_exclusions,
        "confirmation_indices": chosen, "per_class": per_class,
        "project_git_sha_at_creation": git_sha(ROOT),
        "model_revision": "7b928d828b7b0e022f929d9bd2e44165aa270148",
        "status": "reserved_not_inferred",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in (
        "dataset", "seed", "n_classes", "n_confirmation", "excluded_index_counts", "status")}, indent=2))
    print(f"Wrote manifest: {out_path}")

if __name__ == "__main__":
    raise SystemExit(main())
