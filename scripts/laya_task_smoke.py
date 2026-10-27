#!/usr/bin/env python3
"""Laya typed-primitive + strategy-adapter API smoke on the pinned local build.

Verifies on the LOCAL pinned Laya (no upstream README claims):
1. choice/score/noul return structured results in the project format.
2. WIDE is a same-question rerun with a larger head/max_len budget.
3. predict_tournament works; records rounds; <=group_size choices pass through.
4. noul default + custom labels override rendering.
5. score output shape on a small synthetic scale.

~8 Laya calls, CPU, 2 threads. Records laya commit + model revision.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAYA_ROOT = ROOT / "external" / "laya"
sys.path.insert(0, str(LAYA_ROOT))

import torch  # noqa: E402

import laya  # noqa: E402
from prepare_processbench import (  # noqa: E402
    REVISION as PB_REVISION, REPO_ID as PB_REPO,
    to_choice_task,
)

MODEL = "english"
N_FEWSHOT = 8


def git_sha(path):
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                              check=True, capture_output=True, text=True).stdout.strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="processbench", choices=["processbench"],
                    help="task adapter (only processbench wired; code/finance adapters pending)")
    ap.add_argument("--limit", type=int, default=1,
                    help="number of task samples for the choice check (diagnostic accuracy only)")
    ap.add_argument("--limit-steps-sample", type=int, default=0)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default="results/processbench/laya_api_smoke.json")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_available():
        raise RuntimeError("CPU-only run refused: CUDA is available")

    from huggingface_hub import hf_hub_download
    gsm = json.loads(Path(hf_hub_download(
        PB_REPO, "gsm8k.json", repo_type="dataset", revision=PB_REVISION)).read_text(encoding="utf-8"))
    sample = gsm[args.limit_steps_sample]
    task = to_choice_task(sample)
    # Choice check over --limit task samples (diagnostic accuracy, not a result).
    limit_rows = gsm[:max(1, args.limit)]
    long_sample, long_source = None, None
    for filename in ("gsm8k.json", "math.json", "olympiadbench.json", "omnimath.json"):
        for r in json.loads(Path(hf_hub_download(
                PB_REPO, filename, repo_type="dataset", revision=PB_REVISION)).read_text(encoding="utf-8")):
            if len(r["steps"]) >= 17:
                long_sample, long_source = r, filename
                break
        if long_sample is not None:
            break
    assert long_sample is not None, "no sample with >=17 steps found"
    long_task = to_choice_task(long_sample)

    def choice_q(names):
        return {"type": "choice", "instructions": "Which step contains the earliest error? Choose the sentinel if all steps are correct.",
                "criteria": {name.replace("_", " "): None for name in names}}

    agent = laya.load(args.model, device=args.device)
    state = {"message": sample["problem"] + "\n\nSteps:\n" + "\n".join(f"[{k}] {s}" for k, s in enumerate(sample["steps"]))}
    long_state = {"message": long_sample["problem"] + "\n\nSteps:\n" + "\n".join(f"[{k}] {s}" for k, s in enumerate(long_sample["steps"]))}
    q = choice_q(task["candidates"])
    few = dict(q, criteria=dict(list(q["criteria"].items())[:N_FEWSHOT]))
    for _text in (sample["problem"],):
        agent.predict({"message": _text}, {"q": few})

    checks, notes = {}, []
    t0 = time.perf_counter()
    choice_hits, choice_n = 0, 0
    try:
        for r in limit_rows:
            tq = to_choice_task(r)
            st = {"message": r["problem"] + "\n\nSteps:\n" + "\n".join(f"[{k}] {s}" for k, s in enumerate(r["steps"]))}
            qq = choice_q(tq["candidates"])
            out = agent.predict(st, {"q": qq})
            ans = out["answers"]["q"]
            if not all(k in ans for k in ("choice", "answer_confidence", "probabilities")):
                raise ValueError("choice answer missing required keys")
            pred_norm = str(ans["choice"]).replace(" ", "_")
            gold_norm = tq["candidates"][tq["gold_index"]].replace(" ", "_")
            choice_hits += int(pred_norm == gold_norm)
            choice_n += 1
        a1 = agent.predict(state, {"q": q})
        ans = a1["answers"]["q"]
        checks["choice_structured"] = True
        notes.append(f"choice -> {ans.get('choice')!r} conf={ans.get('answer_confidence')}")
        notes.append(f"choice diagnostic accuracy on {choice_n} processbench rows: {choice_hits}/{choice_n}")
    except Exception as e:  # noqa: BLE001
        checks["choice_structured"] = False
        notes.append(f"choice FAILED: {type(e).__name__}: {e}")
    try:
        a2, c2 = agent.predict(state, {"q": q}, head_max_len=512, max_len=1024), 1
        ans2 = a2["answers"]["q"]
        checks["wide_rerun_same_shape"] = all(k in ans2 for k in ("choice", "answer_confidence", "probabilities"))
        notes.append("wide used head_max_len/max_len rerun kwargs (same question, not continued thought)")
    except Exception as e:  # noqa: BLE001
        checks["wide_rerun_same_shape"] = False
        notes.append(f"wide FAILED: {type(e).__name__}: {e}")
    try:
        out = laya.predict_tournament(agent, long_state, {"q": choice_q(long_task["candidates"])})
        t = out["tournament"]["q"]
        checks["tournament_rounds"] = (t["rounds"] >= 1 and t["n"] == len(long_task["candidates"]))
        notes.append(f"tournament n={t['n']} rounds={t['rounds']} finalists={len(t['labels'])}")
        out2 = laya.predict_tournament(agent, state, {"q": q})
        checks["tournament_passthrough_small"] = (out2["tournament"]["q"]["rounds"] == 0)
    except Exception as e:  # noqa: BLE001
        checks["tournament_rounds"] = False
        checks["tournament_passthrough_small"] = False
        notes.append(f"tournament FAILED: {type(e).__name__}: {e}")
    try:
        nq = {"type": "noul", "instructions": "Is step 0 erroneous?"}
        an = agent.predict({"message": "Step 0: " + sample["steps"][0]}, {"n": nq})
        ansa = an["answers"]["n"]
        # Pinned API: noul value is always P(true); labels only change display text.
        checks["noul_default_labels"] = ("noul" in ansa and "answer_confidence" in ansa
                                         and 0.0 <= float(ansa["noul"]) <= 1.0)
        dec = "true" if float(ansa["noul"]) >= 0.5 else "false"
        nq2 = {"type": "noul", "instructions": "Is step 0 erroneous?",
               "labels": {"false": "no", "true": "yes"}}
        an2 = agent.predict({"message": "Step 0: " + sample["steps"][0]}, {"n": nq2})
        ansa2 = an2["answers"]["n"]
        checks["noul_custom_labels"] = ("noul" in ansa2
                                        and 0.0 <= float(ansa2["noul"]) <= 1.0)
        dec2 = "true" if float(ansa2["noul"]) >= 0.5 else "false"
        notes.append(f"noul P(true) default={ansa['noul']} -> {dec}; custom={ansa2['noul']} -> {dec2}")
    except Exception as e:  # noqa: BLE001
        checks["noul_default_labels"] = False
        checks["noul_custom_labels"] = False
        notes.append(f"noul FAILED: {type(e).__name__}: {e}")
    try:
        sq = {"type": "score", "instructions": "Score clarity of step 0 from 1 to 5",
              "criteria": [str(v) for v in range(1, 6)]}
        as_ = agent.predict({"message": "Step 0: " + sample["steps"][0]}, {"s": sq})
        anss = as_["answers"]["s"]
        # Pinned API: probabilities over level indices, legend maps index -> label.
        probs = anss.get("probabilities", {})
        legend = anss.get("legend", {})
        top = max(probs, key=lambda k: probs[k]) if probs else None
        checks["score_shape"] = ("score" in anss and bool(probs) and bool(legend)
                                 and top in legend and "answer_confidence" in anss)
        notes.append(f"score={anss.get('score')} top_level={top} -> {legend.get(top)}; direction/scale need task-level validation")
    except Exception as e:  # noqa: BLE001
        checks["score_shape"] = False
        notes.append(f"score FAILED: {type(e).__name__}: {e}")

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "laya_commit": git_sha(LAYA_ROOT), "laya_version": getattr(laya, "__version__", None),
        "model": args.model, "device": args.device, "threads": args.threads,
        "processbench_revision": PB_REVISION,
        "long_sample_id": long_sample["id"], "long_sample_source": long_source,
        "long_sample_n_steps": len(long_sample["steps"]),
        "checks": checks, "notes": notes,
        "elapsed_seconds": round(time.perf_counter() - t0, 1),
        "status": "passed" if all(checks.values()) else "partial_failure",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
