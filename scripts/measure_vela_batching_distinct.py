#!/usr/bin/env python3
"""Distinct-view Vela batching test (single process, offline analysis after).

Same state, up to 4 SEMANTICALLY DISTINCT questions in one request vs
separate requests: full Choice, Noul, menu Choice, Score.
Compares DECISIONS (not raw dicts with floats) and saves raw answers for
offline re-analysis. No test data (uses a fixed search_dev row).
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
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
    except Exception:
        pass


def git_sha(path):
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                              check=True, capture_output=True, text=True).stdout.strip()
    except Exception:
        return None


def decision(ans):
    """Categorical decision-level extraction (primary for batching equivalence).

    choice -> normalized key; noul -> pos/neg at 0.5; score -> argmax level key.
    Continuous values (probabilities, expected score) are NOT part of the
    decision; they are compared separately as deltas since fusion/calibration
    depends on them.
    """
    if not isinstance(ans, dict):
        return None
    t = ans.get("type")
    if t == "choice":
        return ["choice", str(ans.get("choice", "")).replace(" ", "_").lower()]
    if t == "noul":
        try:
            p = float(ans.get("noul", 0.0))
        except (TypeError, ValueError):
            return ["noul", None]
        return ["noul", "pos" if p >= 0.5 else "neg"]
    if t == "score":
        probs = ans.get("probabilities") or {}
        if isinstance(probs, dict) and probs:
            try:
                argmax = max(probs.items(), key=lambda kv: float(kv[1]))[0]
            except (TypeError, ValueError):
                argmax = None
            return ["score", str(argmax)]
        try:
            return ["score", str(int(round(float(ans.get("score", 0.0)))))]
        except (TypeError, ValueError):
            return ["score", None]
    return ["unknown", None]


def continuous_delta(sep_ans, bat_ans):
    """Magnitude of continuous shift under batching (same decision may hide it)."""
    if not isinstance(sep_ans, dict) or not isinstance(bat_ans, dict):
        return None
    t = sep_ans.get("type")
    try:
        if t == "choice":
            sp = sep_ans.get("probabilities") or {}
            bp = bat_ans.get("probabilities") or {}
            keys = set(sp) | set(bp)
            return {"l1_prob": round(sum(abs(float(sp.get(k, 0)) - float(bp.get(k, 0))) for k in keys), 4)}
        if t == "noul":
            return {"abs_p_diff": round(abs(float(sep_ans.get("noul", 0)) - float(bat_ans.get("noul", 0))), 4)}
        if t == "score":
            return {"abs_expected_diff": round(abs(float(sep_ans.get("score", 0)) - float(bat_ans.get("score", 0))), 4)}
    except (TypeError, ValueError):
        return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--out", default="results/v2/vela_batching_distinct.json")
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
    sleep_on = prevent_sleep()
    results = []
    try:
        # warm-up (not timed): one separate + one batched call to settle caches
        for _ in range(max(0, args.warmup)):
            try:
                agent.system_one(state, {"full": questions["full"]})
            except Exception:
                pass
        peak = max(peak, proc.memory_info().rss)

        # separate calls (store raw answers for decision-level comparison)
        sep = {}
        sep_usage = {}
        t0 = time.perf_counter()
        for q in out_qs:
            a = agent.system_one(state, {q: questions[q]})
            sep[q] = a["answers"][q]
            sep_usage[q] = a.get("usage", {})
        sep_ms = (time.perf_counter() - t0) * 1000.0
        rss_after_sep = proc.memory_info().rss
        peak = max(peak, rss_after_sep)
        results.append({"mode": "separate", "n_questions": len(out_qs),
                        "latency_ms": round(sep_ms, 1),
                        "per_q_ms": round(sep_ms / len(out_qs), 1),
                        "usage_per_question": sep_usage,
                        "rss_mb": round(rss_after_sep / 2**20, 1),
                        "raw_answers": sep,
                        "decisions": {q: decision(sep[q]) for q in out_qs}})

        # batched call
        t0 = time.perf_counter()
        batched = agent.system_one(state, {q: questions[q] for q in out_qs})
        bat_ms = (time.perf_counter() - t0) * 1000.0
        rss_after_bat = proc.memory_info().rss
        peak = max(peak, rss_after_bat)
        usage = batched.get("usage", {})
        answers = batched.get("answers", {})

        raw_match = {q: (answers.get(q) == sep.get(q)) for q in out_qs}
        dec_sep = {q: decision(sep.get(q)) for q in out_qs}
        dec_bat = {q: decision(answers.get(q)) for q in out_qs}
        dec_match = {q: (dec_sep[q] == dec_bat[q]) for q in out_qs}
        deltas = {q: continuous_delta(sep.get(q), answers.get(q)) for q in out_qs}
        results.append({"mode": "batched", "n_questions": len(out_qs),
                        "latency_ms": round(bat_ms, 1),
                        "per_q_ms": round(bat_ms / len(out_qs), 1), "usage": usage,
                        "rss_mb": round(rss_after_bat / 2**20, 1),
                        "raw_answers": {q: answers.get(q) for q in out_qs},
                        "decisions": dec_bat,
                        "raw_match_vs_separate": raw_match,
                        "decision_match_vs_separate": dec_match,
                        "continuous_deltas": deltas,
                        "all_raw_match": all(raw_match.values()),
                        "all_decision_match": all(dec_match.values())})
        print(json.dumps({"sep_ms": round(sep_ms, 1), "batched_ms": round(bat_ms, 1),
                          "raw_match": raw_match, "decision_match": dec_match,
                          "continuous_deltas": deltas,
                          "status": "passed" if all(dec_match.values()) else "partial_match"}, indent=2),
              flush=True)
        status = "passed" if all(dec_match.values()) else "partial_match"
    finally:
        allow_sleep()

    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "model": "vela-2.0-0.3B", "device": "cpu", "threads": args.threads,
              "warmup_calls": args.warmup, "order": "warmup -> separate -> batched (fixed; single-sample diagnostic)",
              "load_seconds": round(load_s, 1), "peak_rss_mb": round(peak / 2**20, 1),
              "project_git_sha": git_sha(ROOT), "model_revision": "d6f03aa9baca4f9017fa8f8e9c2764d3bed3f409",
              "results": results,
              "status": status,
              "note": "single-sample diagnostic; raw dict equality is float-sensitive, decision_match is primary"}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
