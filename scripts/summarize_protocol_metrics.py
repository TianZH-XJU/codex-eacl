#!/usr/bin/env python3
"""S-06: unified protocol-metrics summary (offline).

Walks results/v4_1/evolution/*.json and records, per file, the fields the
v4.1 protocol requires (track, split/pool, metric, seeds/budget, primary
score, manifest/provenance presence). Heterogeneous schemas are tolerated:
missing fields are omitted, never guessed or null-filled. This is an index,
not an analysis — numbers live in the source files.
"""
from __future__ import annotations

import argparse
import glob
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def summarize(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {"file": Path(path).name}
    for k in ("experiment_id", "status", "track", "metric", "protocol",
              "protocol_version", "pools", "budget_unique", "gens_effective",
              "gens_rule", "verdict"):
        if k in d:
            out[k] = d[k]
    if "seeds" in d:
        out["seeds"] = d["seeds"]
    for k in ("train_cache", "dev_cache", "confirm_cache", "cache"):
        if k in d:
            v = d[k]
            out[k] = v.get("path", v) if isinstance(v, dict) else v
    if "tasks" in d and isinstance(d["tasks"], dict):
        out["tasks"] = sorted(d["tasks"])
    if "cells" in d:
        if isinstance(d["cells"], dict):
            out["cells"] = sorted(d["cells"])
        elif isinstance(d["cells"], list):
            out["n_cells"] = len(d["cells"])
    if "matrix" in d and isinstance(d["matrix"], dict):
        out["n_cells"] = len(d["matrix"])
    if "families" in d and isinstance(d["families"], dict):
        out["families"] = sorted(d["families"])
    rm = d.get("run_manifest", {})
    out["has_run_manifest"] = bool(rm)
    out["has_git_sha"] = bool(d.get("git_sha") or d.get("repo_commit") or rm.get("project_git_sha"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    try:
        sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                             check=True, capture_output=True, text=True).stdout.strip()
    except Exception:
        sha = None
    files = {}
    for f in sorted(glob.glob(str(ROOT / "results/v4_1/evolution/*.json"))):
        try:
            files[Path(f).stem] = summarize(f)
        except Exception as e:  # never fail the index on one bad file
            files[Path(f).stem] = {"file": Path(f).name, "error": f"{type(e).__name__}: {e}"}
    report = {"experiment_id": "S-06", "created_utc": datetime.now(timezone.utc).isoformat(),
              "repo_commit": sha, "protocol_version": "v4.1",
              "note": "index only: missing fields omitted, never inferred",
              "files": files}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    n_ok = sum(1 for v in files.values() if "error" not in v)
    print(f"indexed {n_ok}/{len(files)} files")
    missing_manifest = [k for k, v in files.items() if not v.get("has_run_manifest") and "error" not in v]
    print("without run_manifest/project_git_sha:", missing_manifest)


if __name__ == "__main__":
    raise SystemExit(main())
