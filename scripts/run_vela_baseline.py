#!/usr/bin/env python3
"""Stage 3: Vela default baseline on the SAME rows as the Laya pilot caches.

Reads row ids from an existing Laya pilot cache (processbench/code/findver),
runs Vela noul (+ binary-choice switch where the Laya pilot has one) on those
exact rows, and writes a paired baseline JSON. No test rows. One process.
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
LAYA_ROOT = ROOT / "external" / "laya"
sys.path.insert(0, str(LAYA_ROOT))

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


def load_rows(task, pilot_cache):
    rows = [json.loads(x) for x in Path(pilot_cache).read_text(encoding="utf-8").splitlines() if x.strip()]
    by_pos = {}
    for r in rows:
        by_pos.setdefault((r["split"], r["position"]), []).append(r)
    return by_pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["processbench", "code_defect", "findver"])
    ap.add_argument("--pilot-cache", required=True)
    ap.add_argument("--model", default="vela")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_available():
        raise RuntimeError("CPU-only run refused: CUDA is available")

    from transformers import AutoModel
    snap = ROOT / "models" / "vela"
    proc = psutil.Process()
    t0 = time.perf_counter()
    agent = AutoModel.from_pretrained(str(snap), trust_remote_code=True)
    load_s = time.perf_counter() - t0
    peak = proc.memory_info().rss

    by_pos = load_rows(args.task, args.pilot_cache)
    sleep_on = prevent_sleep()
    results = []
    try:
        for key in sorted(by_pos, key=lambda k: (k[0], k[1])):
            srow = next(x for x in by_pos[key] if x["strategy"] == "single")
            text = srow.get("_text")
            if text is None:
                # recover text from source datasets
                text = recover_text(args.task, srow)
            reps = []
            if args.task == "processbench":
                n = srow.get("n_candidates", 0)
                cands = [f"step {k}" for k in range(max(0, n - 1))] + ["ALL STEPS CORRECT"]
                reps.append(("choice", {"type": "choice",
                    "instructions": "Which step contains the earliest error? Choose the sentinel if all steps are correct.",
                    "criteria": {c: None for c in cands}}, "choice"))
            else:
                single_q = {"code_defect": {"type": "noul", "instructions": "Is this function vulnerable?"},
                            "findver": {"type": "noul", "instructions": "Is the claim supported by the evidence?"}}[args.task]
                reps.append(("noul", single_q, "noul"))
                switch_q = {"code_defect": {"type": "choice", "instructions": "Is this function secure or vulnerable?",
                                            "criteria": {"secure": None, "vulnerable": None}},
                            "findver": {"type": "choice", "instructions": "Is the claim supported by the evidence?",
                                        "criteria": {"supported": None, "not supported": None}}}[args.task]
                reps.append(("switch", switch_q, "choice"))
            for rep_name, q, qtype in reps:
                t1 = time.perf_counter()
                err, out = None, None
                try:
                    out = agent.system_one({"message": text}, {"q": q})
                except Exception as e:  # noqa: BLE001
                    err = f"{type(e).__name__}: {e}"
                ms = (time.perf_counter() - t1) * 1000.0
                peak = max(peak, proc.memory_info().rss)
                ans = (out.get("answers", {}).get("q") if out else None) or {}
                usage = (out.get("usage", {}) if out else {})
                if err is None:
                    if qtype == "noul":
                        pred = "pos" if float(ans.get("noul", 0)) >= 0.5 else "neg"
                        gold = "pos" if bool(srow.get("gold_bool")) else "neg"
                    else:
                        pred = str(ans.get("choice", "")).replace(" ", "_").lower()
                        gold = gold_render(args.task, srow)
                else:
                    pred, gold = None, gold_render(args.task, srow)
                if qtype != "noul":
                    gold = gold_render(args.task, srow)
                results.append({"split": key[0], "position": key[1], "row_id": srow.get("row_id"),
                                "rep": rep_name, "pred": pred, "gold": gold,
                                "correct": bool(pred is not None and pred == gold),
                                "latency_ms": round(ms, 1), "error": err, "usage": usage})
            print(f"{key} done", flush=True)
    finally:
        allow_sleep()

    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "model": "vela-2.0-0.3B",
              "task": args.task, "device": "cpu", "threads": args.threads,
              "pilot_cache": args.pilot_cache, "n_rows": len(by_pos),
              "load_seconds": round(load_s, 1), "peak_rss_mb": round(peak / 2**20, 1),
              "windows_sleep_prevention_active": sleep_on,
              "project_git_sha": git_sha(ROOT), "model_revision": "d6f03aa9baca4f9017fa8f8e9c2764d3bed3f409",
              "results": results,
              "status": "passed" if all(r["error"] is None for r in results) else "partial_failure"}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    acc = sum(r["correct"] for r in results) / max(len(results), 1)
    print(json.dumps({"status": report["status"], "n": len(results),
                      "overall_acc": round(acc, 4)}, indent=2))


def recover_text(task, srow):
    if task == "processbench":
        rows = json.loads((ROOT / "data/raw/processbench" / srow["source_file"]).read_text(encoding="utf-8"))
        r = next(x for x in rows if x["id"] == srow["row_id"])
        # IDENTICAL state text to the Laya pilot cache (see build_processbench_cache.py).
        return r["problem"] + "\n\nSteps:\n" + "\n".join(f"[{k}] {s}" for k, s in enumerate(r["steps"]))
    if task == "code_defect":
        rows = json.loads((ROOT / "data/raw/codexglue_defect/function.json").read_text(encoding="utf-8"))
        return rows[int(srow["row_id"])]["func"]
    rows = json.loads((ROOT / "external/FinDVer/data/testmini.json").read_text(encoding="utf-8"))
    r = next(x for x in rows if x["example_id"] == srow["row_id"])
    rep = json.loads((ROOT / f"external/FinDVer/financial_reports/{r['report']}").read_text(encoding="utf-8"))
    ev = "\n".join(rep["context"][i]["context"] for i in r["relevant_context"])
    return f"Claim: {r['statement']}\n\nEvidence context:\n{ev}"


def gold_render(task, srow):
    g = srow.get("gold_rendered", "") or srow.get("gold_render", "") or ""
    return g.replace(" ", "_").lower()


if __name__ == "__main__":
    raise SystemExit(main())
