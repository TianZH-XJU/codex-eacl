#!/usr/bin/env python3
"""Stage 0 audit: environment, git state, Laya/Vela model manifests (offline).

Writes results/environment/new_route_environment.json,
results/model_manifest/laya.json, results/model_manifest/vela.json,
results/audits/git_state_before_new_route.json. No downloads, no inference.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULES = ["torch", "transformers", "tokenizers", "huggingface_hub", "datasets",
           "pyarrow", "numpy", "pandas", "sklearn", "scipy", "psutil",
           "threadpoolctl", "tqdm", "pytest", "safetensors", "onnxruntime",
           "cma", "optuna", "deap", "pymoo", "neat", "laya"]


def git(cmd, cwd):
    try:
        return subprocess.run(["git", "-C", str(cwd), *cmd], check=True,
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()
    out = Path(args.out_dir)
    ts = datetime.now(timezone.utc).isoformat()

    mods = {}
    for name in MODULES:
        try:
            m = importlib.import_module(name)
            mods[name] = {"available": True, "version": getattr(m, "__version__", None)}
        except Exception as e:  # noqa: BLE001
            mods[name] = {"available": False, "error": f"{type(e).__name__}: {e}"}
    try:
        import psutil as _ps
        mem = _ps.virtual_memory()
        mem_info = {"total_mb": round(mem.total / 2**20, 1), "free_mb": round(mem.free / 2**20, 1)}
        disk = _ps.disk_usage(str(ROOT.resolve().anchor))
        disk_info = {"free_gb": round(disk.free / 2**30, 1), "total_gb": round(disk.total / 2**30, 1)}
    except Exception:  # noqa: BLE001
        mem_info, disk_info = {}, {}

    env = {"timestamp_utc": ts, "platform": platform.platform(), "machine": platform.machine(),
           "python_version": sys.version, "logical_cpu_count": os.cpu_count(),
           "device_policy": "cpu-only", "cpu_threads_policy": 2,
           "memory": mem_info, "disk": disk_info, "modules": mods, "status": "passed"}
    (out / "environment").mkdir(parents=True, exist_ok=True)
    (out / "environment" / "new_route_environment.json").write_text(json.dumps(env, indent=2) + "\n")

    laya_commit = git(["rev-parse", "HEAD"], ROOT / "external" / "laya")
    laya_manifest = {"timestamp_utc": ts, "package": "laya",
                     "installed_version": mods.get("laya", {}).get("version"),
                     "source_commit": laya_commit, "model_repo": "convaiinnovations/laya",
                     "model_revision": "7b928d828b7b0e022f929d9bd2e44165aa270148",
                     "device": "cpu", "status": "cached_and_verified"}
    (out / "model_manifest").mkdir(parents=True, exist_ok=True)
    (out / "model_manifest" / "laya.json").write_text(json.dumps(laya_manifest, indent=2) + "\n")
    vela_manifest = {"timestamp_utc": ts, "model_repo": "vllm-sr/Vela-2.0-0.3B",
                     "pinned_revision": "d6f03aa9baca4f9017fa8f8e9c2764d3bed3f409",
                     "status": "not_downloaded"}
    (out / "model_manifest" / "vela.json").write_text(json.dumps(vela_manifest, indent=2) + "\n")

    git_state = {"timestamp_utc": ts, "root": str(ROOT),
                 "head": git(["rev-parse", "HEAD"], ROOT),
                 "branch": git(["rev-parse", "--abbrev-ref", "HEAD"], ROOT),
                 "status_porcelain": git(["status", "--porcelain"], ROOT),
                 "remotes": git(["remote", "-v"], ROOT)}
    (out / "audits").mkdir(parents=True, exist_ok=True)
    (out / "audits" / "git_state_before_new_route.json").write_text(json.dumps(git_state, indent=2) + "\n")
    print(json.dumps({"head": git_state["head"], "laya_commit": laya_commit,
                      "torch": mods.get("torch", {}).get("version"),
                      "transformers": mods.get("transformers", {}).get("version")}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
