#!/usr/bin/env python3
"""Prepare FinDVer claim verification, inspect-only stage.

Uses the local FinDVer clone at the pinned commit. Validates schema
(entailment bool, subset values, relevant_context indices resolve against the
actual report files), counts rows/labels per subset and split, audits report
overlap between testmini and test, and checks report-file coverage. No Laya
calls; test stays locked (only read for the overlap audit).
"""
from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

FINX_ROOT = Path(__file__).resolve().parents[1] / "external" / "FinDVer"
SUBSETS = {"ie", "numeric", "knowledge"}


def git_sha(path):
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                              check=True, capture_output=True, text=True).stdout.strip()
    except Exception:
        return None


def validate_row(row, reports_dir):
    for key in ("example_id", "subset", "statement", "entailment_label",
                "relevant_context", "report"):
        if key not in row:
            return False, f"missing key {key}"
    # explanation is auxiliary (not needed for the verification decision);
    # counted separately, never invalidating.
    if row["subset"] not in SUBSETS:
        return False, f"bad subset {row['subset']}"
    if not isinstance(row["entailment_label"], bool):
        return False, "entailment_label is not bool"
    if not isinstance(row["relevant_context"], list) or not row["relevant_context"]:
        return False, "relevant_context is not a non-empty list"
    if not isinstance(row["statement"], str) or not row["statement"].strip():
        return False, "empty statement"
    report_path = reports_dir / row["report"]
    if not report_path.exists():
        return False, f"report file missing: {row['report']}"
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        n_ctx = len(report["context"])
        if any(not isinstance(i, int) or not (0 <= i < n_ctx) for i in row["relevant_context"]):
            return False, "relevant_context index out of range"
    except (OSError, ValueError, KeyError) as e:  # noqa: BLE001
        return False, f"report unreadable: {e}"
    return True, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect-only", action="store_true")
    ap.add_argument("--findver-root", default=str(FINX_ROOT))
    ap.add_argument("--manifest", default="results/findver/dataset_manifest.json")
    ap.add_argument("--report", default="results/findver/schema_report.json")
    args = ap.parse_args()
    if not args.inspect_only:
        raise SystemExit("--inspect-only is currently the only supported mode")

    root = Path(args.findver_root)
    reports_dir = root / "financial_reports"
    data = {}
    for split in ("testmini", "test"):
        rows = json.loads((root / "data" / f"{split}.json").read_text(encoding="utf-8"))
        data[split] = rows

    bad, per_split, missing_expl = 0, {}, 0
    for split, rows in data.items():
        id_seen, ok_rows = set(), []
        for row in rows:
            if row["example_id"] in id_seen:
                bad += 1
                print(f"INVALID {split}: duplicate id {row['example_id']}")
                continue
            id_seen.add(row["example_id"])
            if "explanation" not in row:
                missing_expl += 1
            ok, why = validate_row(row, reports_dir)
            if not ok:
                bad += 1
                print(f"INVALID {split}/{row['example_id']}: {why}")
                continue
            ok_rows.append(row)
        per_split[split] = {
            "n": len(ok_rows),
            "subsets": dict(Counter(r["subset"] for r in ok_rows)),
            "labels": {str(k): v for k, v in Counter(r["entailment_label"] for r in ok_rows).items()},
            "n_reports": len({r["report"] for r in ok_rows}),
        }
        data[split] = ok_rows

    ids_mini = {r["example_id"] for r in data["testmini"]}
    ids_test = {r["example_id"] for r in data["test"]}
    rep_mini = {r["report"] for r in data["testmini"]}
    rep_test = {r["report"] for r in data["test"]}

    commit = git_sha(root)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repo": "yilunzhao/FinDVer", "commit": commit,
        "per_split": per_split,
        "example_id_overlap_testmini_test": len(ids_mini & ids_test),
        "report_overlap_testmini_test": len(rep_mini & rep_test),
        "report_overlap_note": "shared reports across official splits: carve controller "
                               "train/dev from testmini grouped by report; never train on test",
        "n_invalid_rows": bad,
        "rows_missing_explanation_auxiliary": missing_expl,
        "task": "claim verification: entailment_label True = supported by report evidence",
        "official_subsets": {"ie": "information extraction", "numeric": "numerical reasoning",
                             "knowledge": "knowledge-intensive reasoning"},
        "evidence": "relevant_context indices resolve against financial_reports/<report>.context[] entries",
    }
    manifest = {"created_utc": report["created_utc"], "repo": "yilunzhao/FinDVer",
                "commit": commit,
                "license_note": "repo LICENSE is MIT (code); data terms need recheck before release",
                "per_split_rows": {k: v["n"] for k, v in per_split.items()},
                "status": "inspected" if bad == 0 else "invalid_rows_found"}
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("per_split", "example_id_overlap_testmini_test",
          "report_overlap_testmini_test", "n_invalid_rows")}, indent=2))
    print(f"Wrote manifest: {args.manifest}")
    if bad:
        raise SystemExit(f"{bad} invalid rows")


if __name__ == "__main__":
    raise SystemExit(main())
