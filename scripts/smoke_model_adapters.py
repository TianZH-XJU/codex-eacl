#!/usr/bin/env python3
"""Stage 1 gate: one real row through the unified adapter (single model).

Loads one adapter, runs one fixed search_train intent row with a 3-option
Choice, validates answer shape + record fields, writes a small JSON.
Run once per model in SEPARATE processes (single-model residency).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import torch  # noqa: E402
from models.registry import get_adapter  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["laya", "vela"])
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    from build_menu_bank import load_pool  # noqa: E402
    row = load_pool("banking77", "search_train", 3, 17)[0]
    questions = {"q": {"type": "choice",
                       "instructions": "Which intent does this message express?",
                       "criteria": {"pay bill": None, "cancel flight": None,
                                    row["label"].replace("_", " "): None}}}
    t0 = time.perf_counter()
    adapter = get_adapter(args.model).load(device="cpu")
    load_s = time.perf_counter() - t0
    answers, usage, rec = adapter.predict("intent", {"message": row["text"]},
                                          questions, prompt_version="stage1-smoke-v1")
    adapter.unload()

    shape_ok = ("q" in answers and isinstance(answers["q"], dict)
                and answers["q"].get("type") == "choice"
                and "choice" in answers["q"])
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "model": args.model,
              "load_seconds": round(load_s, 1), "shape_ok": shape_ok,
              "choice": answers.get("q", {}).get("choice"),
              "record": {"model": rec.model, "revision": rec.revision, "task": rec.task,
                         "input_len": rec.input_len, "output_types": list(rec.output_types),
                         "ok": rec.ok, "parse_ok": rec.parse_ok, "latency_ms": rec.latency_ms,
                         "rss_mb": rec.rss_mb, "n_calls": rec.n_calls,
                         "adapter_version": rec.adapter_version,
                         "prompt_version": rec.prompt_version, "error": rec.error},
              "status": "passed" if shape_ok and rec.ok and rec.parse_ok else "partial_failure"}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "choice": report["choice"],
                      "rss_mb": rec.rss_mb}, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
