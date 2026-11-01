#!/usr/bin/env python3
"""Phase 4 cache builder: real single-call outputs per row (noul p + switch).

One model per process. The cache makes all genome evaluations exact (policy is
a deterministic function of cached outputs; call counts exact). Construction
cost is recorded here and never hidden. No test rows.
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

from models.base import validate_prob_map  # noqa: E402
from models.registry import get_adapter  # noqa: E402
from tasks.code import CodeAdapter  # noqa: E402
from tasks.findver import FindverAdapter  # noqa: E402
from tasks.math import ProcessbenchAdapter  # noqa: E402

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001

TASKS = {"codexglue_defect": CodeAdapter, "findver": FindverAdapter,
         "processbench": ProcessbenchAdapter}
MODEL_TASKS = {"codexglue_defect": "code_defect", "findver": "findver",
               "processbench": "processbench"}


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
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--model", required=True, choices=["laya", "vela"])
    ap.add_argument("--split", default="search_dev")
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
    import math as _math
    if args.model == "vela":
        _w = getattr(model, "_weights", {}) or {}
        provenance = {"weights_verified": bool(_w.get("verified", False)),
                      "weights_sha256": _w.get("sha256"),
                      "weights_skipped": bool(_w.get("skipped", False)),
                      "revision": model.revision}
    else:
        provenance = {"weights_verified": None, "weights_sha256": None,
                      "weights_skipped": False,
                      "laya_pkg": getattr(model, "_pkg_version", None),
                      "revision": model.revision,
                      "note": "Laya weights resolve via HF cache; package version pinned, snapshot hash N/A"}
    proc = psutil.Process()
    peak = proc.memory_info().rss
    sleep_on = prevent_sleep()
    out, calls = [], 0
    try:
        for pos, row in enumerate(rows):
            qs = adapter.build_question(row)
            rec = {}
            reps = ("choice",) if args.task == "processbench" else ("noul", "switch")
            for rep in reps:
                t0 = time.perf_counter()
                err, ans = None, None
                try:
                    q = qs if isinstance(qs, dict) and qs.get("type") else qs[rep]
                    ans, _, rec_one = model.predict(MODEL_TASKS[args.task], {"message": row["text"]},
                                                    {"q": q}, prompt_version="policy-cache-v1")
                    inner = ans.get("q", ans) if isinstance(ans, dict) else {}
                    if not rec_one.ok:
                        err = f"model_error:{rec_one.error}"
                        inner = {}
                    elif not rec_one.parse_ok:
                        err = f"parse_error:{rec_one.error}"
                        inner = {}
                except Exception as e:  # noqa: BLE001
                    err, inner = f"model_error:{type(e).__name__}: {e}", {}
                ms = (time.perf_counter() - t0) * 1000.0
                calls += 1
                rec[rep] = {"ms": round(ms, 1), "error": err, "raw": inner}
                peak = max(peak, proc.memory_info().rss)
            if args.task == "processbench":
                raw = rec["choice"]["raw"] if rec["choice"]["error"] is None else {}
                probs = raw.get("probabilities", {}) if isinstance(raw, dict) else {}
                probs, probs_err = validate_prob_map(probs)
                if probs_err is not None:
                    rec["choice"]["error"] = f"parse_error:probs_{probs_err}"
                order = sorted(probs, key=lambda k: probs[k], reverse=True)
                top1 = probs[order[0]] if order else None
                margin = probs[order[0]] - probs[order[1]] if len(order) > 1 else None
                cands = [f"step {k}" for k in range(len(row["meta"]["steps"]))] + ["ALL_STEPS_CORRECT"]
                gold_render = cands[row["gold"]].lower().replace("_", " ")
                pred = str(raw.get("choice", "")).strip().lower().replace("_", " ") if isinstance(raw, dict) else None
                out.append({"key": row["key"], "gold": row["gold"], "group_id": row["group_id"],
                            "n_cands": len(cands), "gold_render": gold_render,
                            "pred": pred, "correct": bool(pred == gold_render) if pred else False,
                            "top1": top1, "margin": margin, "probs": probs,
                            "errors": {k: v["error"] for k, v in rec.items()}})
                print(f"{pos+1}/{len(rows)} pred={pred} top1={top1} err={[v['error'] for v in rec.values()]}",
                      flush=True)
                continue
            try:
                noul_p = float(rec["noul"]["raw"].get("noul")) if rec["noul"]["error"] is None else None
            except (TypeError, ValueError):
                noul_p = None
            sw = rec["switch"]["raw"] if rec["switch"]["error"] is None else {}
            sw_choice = str(sw.get("choice", "")).replace(" ", "_").lower() if isinstance(sw, dict) else None
            if args.task == "codexglue_defect":
                gold_label = "vulnerable" if row["gold"] else "secure"
            else:
                gold_label = "supported" if row["gold"] else "not_supported"
            out.append({"key": row["key"], "gold": bool(row["gold"]), "group_id": row["group_id"],
                        "noul_p": noul_p, "switch_choice": sw_choice,
                        "switch_correct": bool(sw_choice == gold_label) if sw_choice else False,
                        "errors": {k: v["error"] for k, v in rec.items()}})
            print(f"{pos+1}/{len(rows)} p={noul_p} sw={sw_choice} err={[v['error'] for v in rec.values()]}",
                  flush=True)
    finally:
        allow_sleep()
        model.unload()

    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "task": args.task,
              "model": args.model, "split": args.split, "seed": args.seed, "n": len(out),
              "device": "cpu", "threads": args.threads,
              "model_revision": get_adapter(args.model).revision,
              "model_provenance": provenance,
              "project_git_sha": git_sha(ROOT),
              "run_manifest": {"argv": sys.argv[1:], "cpu_threads": args.threads,
                               "project_git_sha": git_sha(ROOT), "protocol": "v4.1",
                               "prompt_version": "policy-cache-v1",
                               "adapter_versions": "src-v1"},
              "windows_sleep_prevention_active": sleep_on,
              "peak_rss_mb": round(peak / 2**20, 1),
              "real_model_calls": calls, "rows": out,
              "status": "passed" if all(all(v is None for v in r["errors"].values()) for r in out) else "partial_failure"}
    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    ok = (sum(1 for r in out if r.get("top1") is not None) if args.task == "processbench"
          else sum(1 for r in out if r.get("noul_p") is not None))
    print(json.dumps({"status": report["status"], "n": len(out), "usable_ok": ok,
                      "peak_rss_mb": report["peak_rss_mb"]}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
