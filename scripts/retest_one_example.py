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
out = {"dataset": args.dataset, "split": args.split, "index": args.index, "text": text, "gold": gold, "results": {}}
for s in args.strategies.split(","):
    t0 = time.perf_counter()
    try:
        answer, calls = fns[s]({"message": text}, question)
        out["results"][s] = {"choice": answer["choice"], "confidence": answer["answer_confidence"],
                             "correct": answer["choice"] == gold.replace("_", " "),
                             "calls": calls, "latency_ms": round((time.perf_counter()-t0)*1000,1), "error": None}
    except Exception as e:
        out["results"][s] = {"error": f"{type(e).__name__}: {e}", "latency_ms": round((time.perf_counter()-t0)*1000,1)}
print(json.dumps(out, ensure_ascii=False), flush=True)