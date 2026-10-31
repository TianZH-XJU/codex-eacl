#!/usr/bin/env python3
"""Vela multi-question batching measurement (single process, offline analysis later).

Same state text; compare 1 / 2 / 4 / 8 identical Choice questions in ONE
request vs separate requests. Records input_tokens, sequences (if reported),
wall time and RSS. No test data (uses a fixed dev-carve row).
"""
from __future__ import annotations

import argparse
import ctypes
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def prevent_sleep():
    try:
        return bool(ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED))
    except Exception:
        return False


def allow_sleep():
    try:
        ctypes.windll.SetThreadExecutionState(ES_CONTINUOUS)
    except Exception:
        pass


def git_sha(path):
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                              check=True, capture_output=True, text=True).stdout.strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", default="1,2,4,8")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default="results/v2/vela_batching.json")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_available():
        raise RuntimeError("CPU-only run refused: CUDA is available")

    from transformers import AutoModel
    t0 = time.perf_counter()
    agent = AutoModel.from_pretrained(str(ROOT / "models/vela"), trust_remote_code=True)
    load_s = time.perf_counter() - t0
    proc = psutil.Process()
    peak = proc.memory_info().rss

    manifest = json.loads((ROOT / "results/findver/controller_split_manifest.json").read_text(encoding="utf-8"))
    data = {r["example_id"]: r for r in
            json.loads((ROOT / "external/FinDVer/data/testmini.json").read_text(encoding="utf-8"))}
    eid = manifest["dev_example_ids"][0]
    r = data[eid]
    rep = json.loads((ROOT / f"external/FinDVer/financial_reports/{r['report']}").read_text(encoding="utf-8"))
    ev = "\n".join(rep["context"][i]["context"] for i in r["relevant_context"])
    state = {"message": f"Claim: {r['statement']}\nEvidence:\n{ev}"}
    q1 = {"type": "choice", "instructions": "Is the claim supported by the evidence?",
          "criteria": {"supported": None, "not supported": None}}

    sleep_on = prevent_sleep()
    rows = []
    try:
        for n in [int(x) for x in args.counts.split(",")]:
            qs = {f"q{i}": dict(q1) for i in range(n)}
            t1 = time.perf_counter()
            err, out = None, None
            try:
                out = agent.system_one(state, qs)
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {e}"
            ms = (time.perf_counter() - t1) * 1000.0
            peak = max(peak, proc.memory_info().rss)
            usage = (out.get("usage", {}) if out else {})
            ans = (out.get("answers", {}) if out else {})
            rows.append({"n_questions": n, "mode": "batched", "latency_ms": round(ms, 1),
                         "error": err, "usage": usage, "n_answers": len(ans),
                         "per_variant_ms": round(ms / n, 1) if n else None})
            print(f"batched n={n} ms={ms:.0f} usage={usage} err={err}", flush=True)
        # separate-call baseline for the largest n
        n = max(int(x) for x in args.counts.split(","))
        t1 = time.perf_counter()
        errs = 0
        for i in range(n):
            try:
                agent.system_one(state, {"q": dict(q1)})
            except Exception:  # noqa: BLE001
                errs += 1
        ms = (time.perf_counter() - t1) * 1000.0
        peak = max(peak, proc.memory_info().rss)
        rows.append({"n_questions": n, "mode": "separate", "latency_ms": round(ms, 1),
                     "error": None if errs == 0 else f"{errs} errors", "usage": {},
                     "n_answers": n, "per_variant_ms": round(ms / n, 1)})
        print(f"separate n={n} ms={ms:.0f}", flush=True)
    finally:
        allow_sleep()

    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "example_id": eid,
              "device": "cpu", "threads": args.threads, "model_revision": "d6f03aa9baca4f9017fa8f8e9c2764d3bed3f409",
              "project_git_sha": git_sha(ROOT), "load_seconds": round(load_s, 1),
              "peak_rss_mb": round(peak / 2**20, 1), "windows_sleep_prevention_active": sleep_on,
              "results": rows, "status": "passed" if all(r["error"] is None for r in rows) else "partial_failure"}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
