#!/usr/bin/env python3
"""Stage 1A: download pinned Vela 2.0 0.3B Torch-minimal snapshot (local only).

Pins revision d6f03aa9baca4f9017fa8f8e9c2764d3bed3f409, downloads the最小 Torch
file set via snapshot_download, records per-file SHA-256 into
results/model_manifest/vela.json. No model loading here (see smoke step).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import snapshot_download

REPO = "vllm-sr/Vela-2.0-0.3B"
REVISION = "d6f03aa9baca4f9017fa8f8e9c2764d3bed3f409"
ALLOW = ["model.safetensors", "config.json", "calibration.json", "tokenizer.json",
         "tokenizer_config.json", "special_tokens_map.json",
         "configuration_vela2.py", "modeling_vela2.py", "vela2_inference.py",
         "README.md", "USAGE.md", "TRAINING.md", "EVALUATION.md", "LICENSE",
         "NOTICE", "DISTRIBUTION_TERMS.md", "LICENSING_STATUS.md",
         "NOTICE_SOURCES.json", "MODIFICATIONS.md"]
REMOTE_CODE = ["configuration_vela2.py", "modeling_vela2.py", "vela2_inference.py"]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="models/vela")
    ap.add_argument("--out", default="results/model_manifest/vela.json")
    args = ap.parse_args()

    free_gb = shutil.disk_usage(Path(args.cache_dir).resolve().anchor).free / 2**30
    if free_gb < 8.0:
        raise SystemExit(f"refusing Vela download: only {free_gb:.1f} GB free")

    snap = Path(snapshot_download(repo_id=REPO, revision=REVISION, allow_patterns=ALLOW,
                                  local_dir=args.cache_dir))
    files = {}
    for rel in ALLOW:
        p = snap / rel
        if p.exists():
            files[rel] = {"bytes": p.stat().st_size, "sha256": sha256(p)}
    missing = [r for r in ALLOW if r not in files]
    manifest = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "model_repo": REPO, "pinned_revision": REVISION,
                "local_snapshot": str(snap.resolve()),
                "requested_files": ALLOW, "present_files": files,
                "missing_files": missing,
                "remote_code_files": {f: files[f]["sha256"] for f in REMOTE_CODE if f in files},
                "remote_code_note": "trust_remote_code executes these on load; audited before first load",
                "license_note": "weights/code/docs Apache-2.0 per model card; tokenizer under Gemma Terms of Use; see LICENSE/NOTICE/DISTRIBUTION_TERMS in snapshot",
                "status": "downloaded" if not missing else "partial_download"}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"snapshot": str(snap), "n_present": len(files),
                      "missing": missing, "status": manifest["status"]}, indent=2))
    if missing:
        print(f"WARNING: missing files will be needed before load: {missing}")


if __name__ == "__main__":
    raise SystemExit(main())
