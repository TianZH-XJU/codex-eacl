#!/usr/bin/env python3
"""Run the frozen locked-test manifests with one model (CPU-only).

Reads results/v4_1/locked_test/manifest_{code,findver}.json and the frozen
protocol, runs Noul then Choice for every row, and writes per-cell caches.
No labels are used to choose anything; the protocol is already frozen.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from models.registry import get_adapter  # noqa: E402

OUT = ROOT / "results/v4_1/locked_test"
MODEL_TASKS = {"code": "code_defect", "findver": "findver"}


def git_sha(path):
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                              check=True, capture_output=True, text=True).stdout.strip()
    except Exception:
        return None


def prevent_sleep():
    try:
        return bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001))
    except Exception:
        return False


def allow_sleep():
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
    except Exception:
        pass


def clean_p(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) and 0.0 <= v <= 1.0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["laya", "vela"])
    ap.add_argument("--tasks", default="code,findver")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()

    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_before_test_calls":
        raise SystemExit("protocol is not frozen")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_available():
        raise RuntimeError("CPU-only run refused: CUDA is available")

    agent = get_adapter(args.model).load(device="cpu")
    proc = psutil.Process()
    peak = proc.memory_info().rss
    sleep_on = prevent_sleep()
    try:
        for task in [t.strip() for t in args.tasks.split(",") if t.strip()]:
            manifest = json.loads((OUT / f"manifest_{task}.json").read_text(encoding="utf-8"))
            noul_q = protocol["templates"][task]["noul"]
            choice_q = protocol["templates"][task]["choice"]
            rows = []
            for pos, row in enumerate(manifest["rows"]):
                rec = {"key": row["key"], "gold": bool(row["gold"]),
                       "group_id": row["group_id"], "meta": row.get("meta", {})}
                errs = {}
                # Noul.
                try:
                    ans, _, one = agent.predict(MODEL_TASKS[task], {"message": row["text"]},
                                                {"q": noul_q}, prompt_version="locked-test-v1")
                    inner = ans.get("q", ans) if isinstance(ans, dict) else {}
                    if not one.ok:
                        errs["noul"] = f"model_error:{one.error}"
                    elif not one.parse_ok:
                        errs["noul"] = f"parse_error:{one.error}"
                    rec["noul_p"] = clean_p(inner.get("noul"))
                    if rec["noul_p"] is None and "noul" not in errs:
                        errs["noul"] = "parse_error:noul_missing_or_invalid"
                except Exception as e:  # noqa: BLE001
                    errs["noul"] = f"model_error:{type(e).__name__}: {e}"
                    rec["noul_p"] = None
                # Choice.
                try:
                    ans, _, one = agent.predict(MODEL_TASKS[task], {"message": row["text"]},
                                                {"q": choice_q}, prompt_version="locked-test-v1")
                    inner = ans.get("q", ans) if isinstance(ans, dict) else {}
                    if not one.ok:
                        errs["switch"] = f"model_error:{one.error}"
                    elif not one.parse_ok:
                        errs["switch"] = f"parse_error:{one.error}"
                    rec["switch_choice"] = str(inner.get("choice", "")).replace(" ", "_").lower() or None
                    if rec["switch_choice"] is None and "switch" not in errs:
                        errs["switch"] = "parse_error:choice_missing"
                except Exception as e:  # noqa: BLE001
                    errs["switch"] = f"model_error:{type(e).__name__}: {e}"
                    rec["switch_choice"] = None
                gold_label = ("vulnerable" if row["gold"] else "secure") if task == "code" else (
                    "supported" if row["gold"] else "not_supported")
                rec["switch_correct"] = bool(rec.get("switch_choice") == gold_label)
                rec["errors"] = errs
                rows.append(rec)
                peak = max(peak, proc.memory_info().rss)
                print(f"{args.model} {task} {pos+1}/{len(manifest['rows'])} p={rec.get('noul_p')} sw={rec.get('switch_choice')}", flush=True)
            cell = f"{task}_{args.model}"
            report = {"created_utc": datetime.now(timezone.utc).isoformat(),
                      "task": "codexglue_defect" if task == "code" else "findver",
                      "model": args.model, "split": "official_test_locked",
                      "seed": manifest["seed"], "n": len(rows),
                      "device": "cpu", "threads": args.threads,
                      "model_revision": get_adapter(args.model).revision,
                      "project_git_sha": git_sha(ROOT),
                      "protocol_sha256": protocol["protocol_sha256"],
                      "manifest_sha256": manifest["sha256"],
                      "windows_sleep_prevention_active": sleep_on,
                      "peak_rss_mb": round(peak / 2**20, 1),
                      "real_model_calls": 2 * len(rows),
                      "rows": rows,
                      "status": "passed" if all(not r["errors"] for r in rows) else "partial_failure"}
            (OUT / f"cache_{cell}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print("wrote", OUT / f"cache_{cell}.json", report["status"], flush=True)
    finally:
        allow_sleep()
        agent.unload()


if __name__ == "__main__":
    raise SystemExit(main())
