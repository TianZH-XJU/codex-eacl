#!/usr/bin/env python3
"""Unified native baseline runner (Stage 3): task adapter x model adapter.

One model per process. Rows come from group-safe search pools only.
Correctness rules mirror the proven per-task scripts:
- intent/processbench choice: normalized choice == gold label rendering.
- code/findver noul: P(true)>=0.5 vs gold bool; switch: normalized choice vs label.
Writes pilot-compatible JSONL (untracked) + small summary JSON (committed).
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
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from models.registry import get_adapter  # noqa: E402
from tasks.code import CodeAdapter  # noqa: E402
from tasks.findver import FindverAdapter  # noqa: E402
from tasks.intent import IntentAdapter  # noqa: E402
from tasks.math import ProcessbenchAdapter  # noqa: E402

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001

TASKS = {
    "banking77_intent": lambda: IntentAdapter("banking77"),
    "clinc150_intent": lambda: IntentAdapter("clinc150"),
    "codexglue_defect": CodeAdapter,
    "findver": FindverAdapter,
    "processbench": ProcessbenchAdapter,
}

REPS = {
    "banking77_intent": ("choice",),
    "clinc150_intent": ("choice",),
    "codexglue_defect": ("noul", "switch"),
    "findver": ("noul", "switch"),
    "processbench": ("choice",),
}

# task id -> model-adapter task namespace (must match supported_outputs())
MODEL_TASKS = {
    "banking77_intent": "intent",
    "clinc150_intent": "intent",
    "codexglue_defect": "code_defect",
    "findver": "findver",
    "processbench": "processbench",
}


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


def norm(s):
    return str(s or "").strip().lower().replace("_", " ")


def correctness(task, rep, row, ans):
    if not isinstance(ans, dict):
        return False
    if rep == "noul":
        try:
            pred = bool(float(ans.get("noul", 0.0)) >= 0.5)
        except (TypeError, ValueError):
            return False
        return pred == bool(row["gold"])
    # choice / switch
    choice = norm(ans.get("choice"))
    if task.endswith("intent"):
        return choice == norm(row["label"])
    if task == "codexglue_defect":
        gold = "vulnerable" if row["gold"] else "secure"
        return choice == gold
    if task == "findver":
        gold = "supported" if row["gold"] else "not supported"
        return choice == gold
    if task == "processbench":
        cands = [f"step {k}" for k in range(len(row["meta"]["steps"]))] + ["ALL_STEPS_CORRECT"]
        gold = cands[row["gold"]].lower().replace("_", " ")
        return choice == gold
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--model", required=True, choices=["laya", "vela"])
    ap.add_argument("--split", default="search_dev", choices=["search_train", "search_dev", "confirm"])
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_available():
        raise RuntimeError("CPU-only run refused: CUDA is available")

    adapter = TASKS[args.task]()
    rows = adapter.load_rows(args.split, args.limit, args.seed)
    model = get_adapter(args.model).load(device="cpu")
    proc = psutil.Process()
    peak = proc.memory_info().rss
    sleep_on = prevent_sleep()
    out_rows = []
    try:
        for pos, row in enumerate(rows):
            questions = adapter.build_question(row)
            for rep in REPS[args.task]:
                if isinstance(questions, dict) and questions.get("type"):
                    q = questions
                else:
                    q = questions.get(rep)
                t0 = time.perf_counter()
                err, ans, usage = None, None, {}
                try:
                    ans, usage, rec = model.predict(MODEL_TASKS[args.task],
                                                    {"message": row["text"]}, {"q": q},
                                                    prompt_version="unified-native-v1")
                except Exception as e:  # noqa: BLE001
                    err = f"{type(e).__name__}: {e}"
                    rec = None
                ms = (time.perf_counter() - t0) * 1000.0
                ok = False if err else correctness(args.task, rep, row, ans)
                out_rows.append({"task": args.task, "split": args.split, "seed": args.seed,
                                 "position": pos, "row_id": row["key"], "rep": rep,
                                 "gold": row["gold"], "choice": (ans or {}).get("choice"),
                                 "correct": bool(ok), "latency_ms": round(ms, 1),
                                 "usage": usage, "error": err,
                                 "record_rss_mb": round(rec.rss_mb, 1) if rec else None,
                                 "group_id": row["group_id"]})
                peak = max(peak, proc.memory_info().rss)
            print(f"{pos+1}/{len(rows)} done", flush=True)
    finally:
        allow_sleep()
        model.unload()

    by_rep = {}
    for rep in REPS[args.task]:
        sub = [r for r in out_rows if r["rep"] == rep and r["error"] is None]
        by_rep[rep] = {"n": len(sub),
                       "accuracy": round(sum(r["correct"] for r in sub) / len(sub), 4) if sub else None,
                       "mean_ms": round(sum(r["latency_ms"] for r in sub) / len(sub), 1) if sub else None}
    summary = {"created_utc": datetime.now(timezone.utc).isoformat(), "task": args.task,
               "model": args.model, "split": args.split, "seed": args.seed,
               "device": "cpu", "threads": args.threads,
               "model_revision": get_adapter(args.model).revision,
               "project_git_sha": git_sha(ROOT),
               "windows_sleep_prevention_active": sleep_on,
               "peak_rss_mb": round(peak / 2**20, 1), "by_rep": by_rep,
               "status": "passed" if all(r["error"] is None for r in out_rows) else "partial_failure"}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out_rows) + "\n", encoding="utf-8")
    out.with_name(out.stem + ".summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "by_rep": by_rep,
                      "peak_rss_mb": summary["peak_rss_mb"]}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
