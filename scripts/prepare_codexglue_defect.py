#!/usr/bin/env python3
"""Prepare CodeXGLUE Defect Detection (Devign origin), inspect-only stage.

Downloads function.json (official Google Drive) + train/valid/test index lists
(pinned CodeXGLUE commit), rebuilds the three jsonl splits with the official
preprocess rule, and validates schema, split sizes, label balance, exact-duplicate
functions and project distribution. No Laya calls, no model training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO = "microsoft/CodeXGLUE"
COMMIT = "ac74a62802a0dd159b3258c78a2df8ad36cdf2b9"  # main HEAD as of 2026-10-27
RAW_BASE = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/Code-Code/Defect-detection/dataset"
GDRIVE_ID = "1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF"
INDEX_FILES = {"train": "train.txt", "valid": "valid.txt", "test": "test.txt"}
EXPECTED_SIZES = {"train": 21854, "valid": 2732, "test": 2732}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_text(url, path):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r, Path(path).open("wb") as f:
        f.write(r.read())


def ensure_gdrive(path):
    if Path(path).exists() and Path(path).stat().st_size > 0:
        return False
    try:
        subprocess.run([sys.executable, "-m", "gdown", GDRIVE_ID, "-O", str(path)],
                       check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as e:
        raise SystemExit(f"Google Drive download failed (need gdown + network): {e}")
    return True


def validate_row(row, idx):
    if not isinstance(row.get("func"), str) or not row["func"].strip():
        return False, "empty func"
    if row.get("target") not in (0, 1):
        return False, "target not in {0,1}"
    if row.get("idx") != idx:
        return False, "idx mismatch"
    return True, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect-only", action="store_true")
    ap.add_argument("--raw-dir", default="data/raw/codexglue_defect")
    ap.add_argument("--manifest", default="results/codexglue_defect/dataset_manifest.json")
    ap.add_argument("--report", default="results/codexglue_defect/schema_report.json")
    args = ap.parse_args()
    if not args.inspect_only:
        raise SystemExit("--inspect-only is currently the only supported mode")

    raw = Path(args.raw_dir)
    raw.mkdir(parents=True, exist_ok=True)
    ensure_gdrive(raw / "function.json")
    index_sets, index_hashes = {}, {}
    for split, filename in INDEX_FILES.items():
        path = raw / filename
        if not path.exists():
            download_text(f"{RAW_BASE}/{filename}", path)
        index_hashes[filename] = sha256(path)
        index_sets[split] = {int(x) for x in path.read_text(encoding="utf-8").split()}

    js_all = json.loads((raw / "function.json").read_text(encoding="utf-8"))
    # Official preprocess.py rule: enumerate function.json, assign by index set.
    splits, bad, seen_idx = {}, 0, set()
    for split, idx_set in index_sets.items():
        rows = []
        for idx in sorted(idx_set):
            row = dict(js_all[idx])
            row["idx"] = idx
            ok, why = validate_row(row, idx)
            if not ok:
                bad += 1
                print(f"INVALID {split}/{idx}: {why}")
                continue
            rows.append(row)
            seen_idx.add(idx)
        splits[split] = rows
    assert len(seen_idx) == sum(len(v) for v in index_sets.values()), "index sets must be disjoint"
    assert len(seen_idx) == len(js_all), "index sets must cover function.json"

    sizes = {k: len(v) for k, v in splits.items()}
    assert sizes == EXPECTED_SIZES, f"size mismatch: {sizes}"

    func_counts = Counter(r["func"] for rows in splits.values() for r in rows)
    exact_dup_funcs = sum(1 for c in func_counts.values() if c > 1)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repo": REPO, "commit": COMMIT, "gdrive_id": GDRIVE_ID,
        "task": "binary classification: target 1 = insecure/vulnerable, 0 = secure",
        "sizes": sizes,
        "label_balance": {k: dict(Counter(r["target"] for r in v)) for k, v in splits.items()},
        "n_invalid_rows": bad,
        "exact_duplicate_func_strings": exact_dup_funcs,
        "projects": {k: dict(Counter(r.get("project", "?") for r in v)) for k, v in splits.items()},
        "func_char_len": {},
        "official_metric": "accuracy (evaluator/evaluator.py)",
    }
    for split, rows in splits.items():
        lens = sorted(len(r["func"]) for r in rows)
        report["func_char_len"][split] = {"min": lens[0], "p50": lens[len(lens) // 2],
                                          "p95": lens[int(len(lens) * 0.95)], "max": lens[-1]}
    manifest = {"created_utc": report["created_utc"], "repo": REPO, "commit": COMMIT,
                "gdrive_id": GDRIVE_ID, "function_json_sha256": sha256(raw / "function.json"),
                "index_sha256": index_hashes, "sizes": sizes,
                "license_note": "verify CodeXGLUE/Devign terms before release; Drive data has no separate license file",
                "status": "inspected" if bad == 0 else "invalid_rows_found"}
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"sizes": sizes, "exact_duplicate_func_strings": exact_dup_funcs,
                      "n_invalid_rows": bad}, indent=2))
    print(f"Wrote manifest: {args.manifest}")
    if bad:
        raise SystemExit(f"{bad} invalid rows")


if __name__ == "__main__":
    raise SystemExit(main())
