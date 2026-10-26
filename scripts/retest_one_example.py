#!/usr/bin/env python3
"""Re-run Laya for one cached example to check latency reproducibility."""
import argparse, json, random, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external" / "laya"))
sys.path.insert(0, str(ROOT / "external" / "laya" / "research" / "benchmarks" / "tournament"))
import torch
import benchmark as bench
import laya

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="massive")
ap.add_argument("--split", default="validation")
ap.add_argument("--limit", type=int, default=30)
ap.add_argument("--seed", type=int, default=17)
ap.add_argument("--index", type=int, required=True)
ap.add_argument("--strategies", default="single,tournament")
ap.add_argument("--model", default="english")
ap.add_argument("--device", default="cpu")
ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--out", default=None)
args = ap.parse_args()

torch.set_num_threads(args.threads)
torch.set_num_interop_threads(1)
load, _ = bench.DATASETS[args.dataset]
names, rows = load(args.split)
if args.limit and args.limit < len(rows):
    rows = random.Random(args.seed).sample(rows, args.limit)
text, gold = rows[args.index]
question = {"type": "choice", "instructions": bench.DATASETS[args.dataset][1],
            "criteria": {n.replace("_", " "): None for n in names}}
agent = laya.load(args.model, device=args.device)
fns = {s: bench.strategy(s, agent) for s in args.strategies.split(",")}
import subprocess, datetime
laya_commit = None
try:
    laya_commit = subprocess.run(["git", "-C", str(ROOT / "external" / "laya"), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
except Exception:
    pass
out = {"dataset": args.dataset, "split": args.split, "seed": args.seed, "limit": args.limit,
       "index": args.index, "text": text, "gold": gold, "model": args.model, "device": args.device,
       "cpu_threads": args.threads, "laya_commit": laya_commit,
       "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "results": {}}
for s in args.strategies.split(","):
    t0 = time.perf_counter()
    try:
        answer, calls = fns[s]({"message": text}, question)
        out["results"][s] = {"choice": answer["choice"], "answer_confidence": answer["answer_confidence"],
                             "entropy_confidence": answer.get("confidence"),
                             "correct": answer["choice"] == gold.replace("_", " "),
                             "calls": calls, "latency_ms": round((time.perf_counter()-t0)*1000,1), "error": None}
    except Exception as e:
        out["results"][s] = {"error": f"{type(e).__name__}: {e}", "latency_ms": round((time.perf_counter()-t0)*1000,1)}
text_out = json.dumps(out, ensure_ascii=False, indent=2)
print(text_out, flush=True)
if args.out:
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text_out + "\n", encoding="utf-8")