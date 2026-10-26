#!/usr/bin/env python3
"""Inspect CPU-only environment and write a machine-readable report."""
from __future__ import annotations
import argparse, importlib, json, os, platform, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
MODULES = ["torch", "laya", "transformers", "huggingface_hub", "neat", "datasets", "pyarrow", "numpy", "pandas", "sklearn", "scipy", "psutil", "threadpoolctl"]
def get_cpu_model():
    try:
        for line in Path("/proc/cpuinfo").read_text(errors="ignore").splitlines():
            if line.lower().startswith("model name"): return line.split(":", 1)[1].strip()
    except OSError: pass
    return platform.processor() or None
def get_git_sha():
    try: return subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError): return None
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/environment.json")
    parser.add_argument("--require-laya", action="store_true")
    parser.add_argument("--require-neat", action="store_true")
    args = parser.parse_args()
    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "git_sha": get_git_sha(),
      "python_version": sys.version, "platform": platform.platform(), "machine": platform.machine(),
      "cpu_model": get_cpu_model(), "logical_cpu_count": os.cpu_count(),
      "cpu_threads_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS")},
      "modules": {}, "device": "cpu", "status": "passed", "notes": []}
    for name in MODULES:
        try:
            mod = importlib.import_module(name)
            report["modules"][name] = {"available": True, "version": getattr(mod, "__version__", None)}
            if name == "torch":
                report["torch_version"] = mod.__version__
                report["torch_cuda_available"] = bool(mod.cuda.is_available())
                report["torch_cpu_threads"] = int(mod.get_num_threads())
        except Exception as exc: report["modules"][name] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    missing = []
    if args.require_laya and not report["modules"]["laya"]["available"]: missing.append("laya")
    if args.require_neat and not report["modules"]["neat"]["available"]: missing.append("neat")
    if missing: report["status"] = "failed"; report["notes"].append("Missing required modules: " + ", ".join(missing))
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n"); print(json.dumps(report, indent=2))
    return 1 if missing else 0
if __name__ == "__main__": raise SystemExit(main())
