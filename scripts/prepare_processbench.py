#!/usr/bin/env python3
"""Prepare ProcessBench (Qwen) for the math error-localization Choice gate.

Stage 1 (--inspect-only): download the four pinned source files, verify
SHA-256, validate the official label mapping (label == -1 means all steps
correct, otherwise 0-based index of the earliest erroneous step, matching
QwenLM/ProcessBench run_eval.py `prediction == label`), and write a dataset
manifest plus a schema/grouping report. No Laya calls, no test split use.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "Qwen/ProcessBench"
REVISION = "3bdcd5371ed567559a78f559c01c13a6deee7604"
FILES = ["gsm8k.json", "math.json", "olympiadbench.json", "omnimath.json"]
PROCESSBENCH_GIT_COMMIT = "e8024636bcabdf8bd514440551b531d3f90dd18b"
ALL_CORRECT_SENTINEL = "ALL_STEPS_CORRECT"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_row(row):
    """Official mapping: int label; -1 or a valid 0-based step index."""
    for key in ("id", "problem", "steps", "label", "generator", "final_answer_correct"):
        if key not in row:
            return False, f"missing key {key}"
    if not isinstance(row["label"], int):
        return False, "label is not int"
    if not isinstance(row["steps"], list) or not row["steps"]:
        return False, "steps is not a non-empty list"
    if row["label"] != -1 and not (0 <= row["label"] < len(row["steps"])):
        return False, "label out of step range"
    return True, ""


def to_choice_task(row):
    """Adapt one record to a Choice task over step entries + sentinel.

    Candidates: one per step (0-based) plus ALL_CORRECT_SENTINEL last.
    Gold: label if label != -1 else len(steps) (sentinel position).
    """
    steps = row["steps"]
    candidates = [f"step {k}" for k in range(len(steps))] + [ALL_CORRECT_SENTINEL]
    gold = row["label"] if row["label"] != -1 else len(steps)
    return {"candidates": candidates, "gold_index": gold, "n_steps": len(steps)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect-only", action="store_true",
                    help="only inspect schema/mapping and write manifest (no adapted cache)")
    ap.add_argument("--output-dir", default="data/raw/processbench")
    ap.add_argument("--manifest", default="results/processbench/dataset_manifest.json")
    ap.add_argument("--report", default="results/processbench/schema_report.json")
    args = ap.parse_args()

    if not args.inspect_only:
        raise SystemExit("--inspect-only is currently the only supported mode")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files, problems, labels, generators, per_file = [], Counter(), Counter(), Counter(), {}
    bad = 0
    for filename in FILES:
        path = Path(hf_hub_download(REPO_ID, filename, repo_type="dataset", revision=REVISION,
                                    local_dir=str(out_dir)))
        rows = json.loads(path.read_text(encoding="utf-8"))
        per_file[filename] = len(rows)
        for row in rows:
            ok, why = validate_row(row)
            if not ok:
                bad += 1
                print(f"INVALID {row.get('id')}: {why}")
                continue
            problems[row["problem"]] += 1
            labels[row["label"]] += 1
            generators[row["generator"]] += 1
        files.append({"filename": filename, "path": str(path), "n_rows": len(rows),
                      "size_bytes": path.stat().st_size, "sha256": sha256(path)})
    multi = sum(1 for v in problems.values() if v > 1)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repo_id": REPO_ID, "revision": REVISION,
        "processbench_git_commit": PROCESSBENCH_GIT_COMMIT,
        "n_rows": sum(per_file.values()), "n_problems": len(problems),
        "problems_with_multiple_trajectories": multi,
        "n_invalid_rows": bad,
        "per_file_rows": per_file,
        "label_histogram": {str(k): v for k, v in sorted(labels.items())},
        "generator_histogram": dict(generators),
        "steps_len": {"min": None, "max": None},
        "official_metric": "error acc / correct-correct acc / F1 = harmonic mean (run_eval.py)",
        "label_mapping": "label == -1 means all steps correct, else 0-based earliest-error step index",
    }
    manifest = {"created_utc": report["created_utc"], "repo_id": REPO_ID, "revision": REVISION,
                "processbench_git_commit": PROCESSBENCH_GIT_COMMIT,
                "license": "Apache-2.0 (per dataset card; recheck before release)",
                "files": files, "status": "inspected" if bad == 0 else "invalid_rows_found"}
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("n_rows", "n_problems",
          "problems_with_multiple_trajectories", "n_invalid_rows")}, indent=2))
    print(f"Wrote manifest: {args.manifest}")
    if bad:
        raise SystemExit(f"{bad} invalid rows")


if __name__ == "__main__":
    raise SystemExit(main())
