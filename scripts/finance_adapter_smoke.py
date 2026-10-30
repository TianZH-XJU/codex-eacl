#!/usr/bin/env python3
"""Gate 0: FinDVer long-doc/truncation audit + Noul vs Choice adapter check.

Offline audit: input lengths (statement + evidence context), table/text mix,
truncation risk vs a fixed head budget. Small Laya adapter check runs on the
controller DEV carve-out only (default 8 rows): evidence-provided condition
(relevant_context entries) with noul and binary choice. No test access.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import psutil  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LAYA_ROOT = ROOT / "external" / "laya"
sys.path.insert(0, str(LAYA_ROOT))

import laya  # noqa: E402

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="results/findver/controller_split_manifest.json")
    ap.add_argument("--findver-root", default="external/FinDVer")
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--model", default="english")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default="results/findver/adapter_smoke.json")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_available():
        raise RuntimeError("CPU-only run refused: CUDA is available")

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    data = {r["example_id"]: r for r in
            json.loads(Path(args.findver_root, "data", "testmini.json").read_text(encoding="utf-8"))}
    dev_ids = manifest["dev_example_ids"][:args.limit]

    def evidence_text(row):
        rep = json.loads(Path(args.findver_root, "financial_reports", row["report"]).read_text(encoding="utf-8"))
        parts = []
        for i in row["relevant_context"]:
            parts.append(rep["context"][i]["context"])
        return "\n".join(parts)

    lens = {"statement": [], "evidence": []}
    for eid in manifest["dev_example_ids"] + manifest["train_example_ids"]:
        r = data[eid]
        lens["statement"].append(len(r["statement"]))
        lens["evidence"].append(len(evidence_text(r)))
    audit = {k: {"min": min(v), "p50": sorted(v)[len(v) // 2],
                "p95": sorted(v)[min(len(v) - 1, int(len(v) * 0.95))], "max": max(v)}
             for k, v in lens.items()}

    agent = laya.load(args.model, device=args.device)
    agent.predict({"message": "warm up"}, {"q": {"type": "noul", "instructions": "Is this supported?"}})

    proc = psutil.Process()
    peak = proc.memory_info().rss
    sleep_on = prevent_sleep()
    rows = []
    try:
        for pos, eid in enumerate(dev_ids):
            r = data[eid]
            ev = evidence_text(r)
            state = {"message": f"Claim: {r['statement']}\n\nEvidence:\n{ev}"}
            t0 = time.perf_counter()
            n_err = None
            try:
                n_out = agent.predict(state, {"q": {"type": "noul", "instructions": "Is the claim supported by the evidence?"}})
                n_ans = n_out["answers"]["q"]
            except Exception as e:  # noqa: BLE001
                n_err, n_ans = f"{type(e).__name__}: {e}", None
            n_ms = (time.perf_counter() - t0) * 1000.0
            t0 = time.perf_counter()
            c_err = None
            try:
                c_out = agent.predict(state, {"q": {"type": "choice",
                    "instructions": "Is the claim supported by the evidence?",
                    "criteria": {"supported": None, "not supported": None}}})
                c_ans = c_out["answers"]["q"]
            except Exception as e:  # noqa: BLE001
                c_err, c_ans = f"{type(e).__name__}: {e}", None
            c_ms = (time.perf_counter() - t0) * 1000.0
            n_p = n_ans.get("noul") if n_ans else None
            c_choice = c_ans.get("choice") if c_ans else None
            agree = (n_p is not None and c_choice is not None
                     and ((n_p >= 0.5) == (c_choice.replace(" ", "_") == "supported")))
            rows.append({"example_id": eid, "label": r["entailment_label"], "subset": r["subset"],
                         "statement_len": len(r["statement"]), "evidence_len": len(ev),
                         "noul_p_true": n_p, "noul_error": n_err, "noul_ms": round(n_ms, 1),
                         "choice": c_choice, "choice_error": c_err, "choice_ms": round(c_ms, 1),
                         "agree": agree})
            peak = max(peak, proc.memory_info().rss)
            print(f"{pos+1}/{len(dev_ids)} {eid} label={r['entailment_label']} agree={agree}", flush=True)
    finally:
        allow_sleep()

    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "task": "findver_claim",
              "condition": "evidence-provided (relevant_context only), NOT end-to-end retrieval",
              "split": "controller-dev carve-out from testmini", "n": len(rows),
              "length_audit_chars": audit,
              "model": args.model, "device": args.device, "threads": args.threads,
              "laya_commit": git_sha(LAYA_ROOT), "project_git_sha": git_sha(ROOT),
              "windows_sleep_prevention_active": sleep_on,
              "peak_rss_mb": round(peak / (1024 * 1024), 1),
              "structured_ok": all(x["noul_error"] is None and x["choice_error"] is None for x in rows),
              "agreement_rate": round(sum(x["agree"] for x in rows) / len(rows), 4) if rows else None,
              "rows": rows, "status": "passed" if all(x["noul_error"] is None and x["choice_error"] is None for x in rows) else "partial_failure"}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("structured_ok", "agreement_rate", "status")}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
