#!/usr/bin/env python3
"""Distinct-view Vela batching test (single process, offline analysis after).

Same state, up to 4 SEMANTICALLY DISTINCT questions in one request vs
separate requests: full Choice, top-10 menu Choice, Noul, Score.
Measures per-question latency, usage, and answer consistency vs separate calls.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def git_sha(path):
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                              check=True, capture_output=True, text=True).stdout.strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default="results/v2/vela_batching_distinct.json")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    from transformers import AutoModel
    t0 = time.perf_counter()
    agent = AutoModel.from_pretrained(str(ROOT / "models/vela"), trust_remote_code=True)
    load_s = time.perf_counter() - t0
    proc = psutil.Process()
    peak = proc.memory_info().rss

    manifest = json.loads((ROOT / "results/v2/data_manifest/findver_report_groups.json").read_text(encoding="utf-8"))
    data = {r["example_id"]: r for r in
            json.loads((ROOT / "external/FinDVer/data/testmini.json").read_text(encoding="utf-8"))}
    rep = manifest["splits"]["search_dev"][0]
    r = next(x for x in data.values() if x["report"] == rep)
    repj = json.loads((ROOT / f"external/FinDVer/financial_reports/{r['report']}").read_text(encoding="utf-8"))
    ev = "\n".join(repj["context"][i]["context"] for i in r["relevant_context"])
    state = f"Claim: {r['statement']}\nEvidence:\n{ev}"
    questions = {
        "full": {"type": "choice", "instructions": "Is the claim supported by the evidence?",
                 "criteria": {"supported": None, "not_supported": None}},
        "noul": {"type": "noul", "instructions": "Is the claim supported by the evidence?"},
        "menu": {"type": "choice", "instructions": "Which option best describes the evidence status?",
                 "criteria": {"supported": None, "not supported": None, "unclear": None, "partial": None}},
        "score": {"type": "score", "instructions": "How strongly is the claim supported (0-4)?",
                  "criteria": ["0", "1", "2", "3", "4"]},
    }
    out_qs = ["full", "noul", "menu", "score"]
    results = []
    # separate calls
    sep = {}
    t0 = time.perf_counter()
    for q in out_qs:
        a = agent.system_one(state, {q: questions[q]})
        sep[q] = a["answers"][q]
    sep_ms = (time.perf_counter() - t0) * 1000.0
    peak = max(peak, proc.memory_info().rss)
    results.append({"mode": "separate", "n_questions": len(out_qs), "latency_ms": round(sep_ms, 1),
                    "per_q_ms": round(sep_ms / len(out_qs), 1)})
    # batched call
    t0 = time.perf_counter()
    batched = agent.system_one(state, {q: questions[q] for q in out_qs})
    bat_ms = (time.perf_counter() - t0) * 1000.0
    peak = max(peak, proc.memory_info().rss)
    usage = batched.get("usage", {})
    answers = batched.get("answers", {})
    match = {q: (answers.get(q) == sep.get(q)) for q in out_qs}
    results.append({"mode": "batched", "n_questions": len(out_qs), "latency_ms": round(bat_ms, 1),
                    "per_q_ms": round(bat_ms / len(out_qs), 1), "usage": usage,
                    "answer_match_vs_separate": match, "all_match": all(match.values())})
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "model": "vella-2.0-0.3b", "device": "cpu", "threads": args.threads,
              "load_seconds": round(load_s, 1), "peak_rss_mb": round(peak / 2**20, 1),
              "project_git_sha": git_sha(ROOT), "model_revision": "d6f03aa9baca4f9017fa8f8e9c2764d3bed3f409",
              "results": results,
              "status": "passed" if all(match.values()) else "partial_match"}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"sep_ms": round(sep_ms, 1), "batched_ms": round(bat_ms, 1),
                      "all_match": all(match.values()), "status": report["status"]}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
