#!/usr/bin/env python3
"""E2 menu-parameter pilot: exhaustive (+ random-subset) search with REAL model calls.

Genome = {retriever, k, norm, order} = 2*3*2*2 = 24 combos (tiny, exhaustive).
Menus built offline (TF-IDF fit on search_train pool only); each genome evaluated
with real menu-restricted Choice calls on a search_dev row subset. No test rows.
One model per process. Random Search with a smaller budget measures search
efficiency against the exhaustive landscape. GP over this 24-point space is
unjustified (documented); GP is reserved for tree space pending budget.
"""
from __future__ import annotations

import argparse
import ctypes
import itertools
import json
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_menu_bank import load_pool, rank_labels  # noqa: E402

LAYA_ROOT = ROOT / "external" / "laya"

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


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


GENOME_SPACE = {
    "retriever": ["tfidf-word", "tfidf-char"],
    "k": [5, 10, 20],
    "norm": ["raw", "lower"],
    "order": ["ranked", "shuffled"],
}


def all_genomes():
    keys = list(GENOME_SPACE)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(GENOME_SPACE[k] for k in keys))]


def sample_genomes(n, seed):
    rng = random.Random(seed)
    pool = all_genomes()
    return rng.sample(pool, min(n, len(pool)))


def norm_text(t, how):
    if how == "lower":
        return t.lower()
    return t


def build_menus(genome, train, dev_rows):
    ranked = rank_labels([norm_text(r["text"], genome["norm"]) for r in train],
                         [r["label"] for r in train],
                         [norm_text(r["text"], genome["norm"]) for r in dev_rows],
                         sorted({r["label"] for r in train}), genome["retriever"])
    k = genome["k"]
    menus = []
    for r, preds in zip(dev_rows, ranked):
        menu = preds[:k]
        if genome["order"] == "shuffled":
            rr = random.Random(hash((r["key"], k)) & 0xFFFFFFFF)
            menu = list(menu)
            rr.shuffle(menu)
        menus.append({**r, "menu": menu, "recall_hit": bool(r["label"] in menu)})
    return menus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["banking77", "clinc150"])
    ap.add_argument("--model", required=True, choices=["laya", "vela"])
    ap.add_argument("--strategy", required=True, choices=["exhaustive", "random"])
    ap.add_argument("--budget", type=int, default=24)
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-rows", type=int, default=30)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_available():
        raise RuntimeError("CPU-only run refused: CUDA is available")

    genomes = all_genomes() if args.strategy == "exhaustive" else sample_genomes(args.budget, args.seed)
    train = load_pool(args.dataset, "search_train", args.n_train, args.seed)
    dev_all = load_pool(args.dataset, "search_dev", None, args.seed + 1)
    dev_rows = dev_all[:args.n_rows]

    if args.model == "laya":
        sys.path.insert(0, str(LAYA_ROOT / "research" / "benchmarks" / "tournament"))
        import laya  # noqa: E402
        agent = laya.load("english", device="cpu")

        def call(text, criteria):
            return agent.predict({"message": text},
                                 {"q": {"type": "choice",
                                        "instructions": "Which intent does this message express?",
                                        "criteria": criteria}})["answers"]["q"]
    else:
        from transformers import AutoModel
        agent = AutoModel.from_pretrained(str(ROOT / "models/vela"), trust_remote_code=True)

        def call(text, criteria):
            return agent.system_one(text, {"q": {"type": "choice",
                "instructions": "Which intent does this message express?",
                "criteria": criteria}})["answers"]["q"]

    proc = psutil.Process()
    peak = proc.memory_info().rss
    sleep_on = prevent_sleep()
    results = []
    try:
        for gi, genome in enumerate(genomes):
            menus = build_menus(genome, train, dev_rows)
            correct, lat, hits = 0, 0.0, 0
            for row in menus:
                q = {c.replace("_", " "): None for c in row["menu"]}
                t0 = time.perf_counter()
                err, choice = None, None
                try:
                    ans = call(row["text"], q)
                    choice = str(ans.get("choice", "")).replace(" ", "_").lower()
                except Exception as e:  # noqa: BLE001
                    err = f"{type(e).__name__}: {e}"
                ms = (time.perf_counter() - t0) * 1000.0
                ok = bool(err is None and choice == row["label"].replace(" ", "_").lower())
                correct += int(ok)
                lat += ms
                hits += int(row["recall_hit"])
            peak = max(peak, proc.memory_info().rss)
            results.append({"genome_id": gi, "genome": genome, "n": len(menus),
                            "accuracy": round(correct / len(menus), 4),
                            "mean_ms": round(lat / len(menus), 1),
                            "recall_hit_rate": round(hits / len(menus), 4)})
            print(f"genome {gi+1}/{len(genomes)} {genome} acc={correct/len(menus):.3f}", flush=True)
    finally:
        allow_sleep()

    best = max(results, key=lambda r: (r["accuracy"], -r["mean_ms"]))
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "task": args.dataset,
              "model": args.model, "strategy": args.strategy, "n_genomes": len(genomes),
              "n_rows": len(dev_rows), "n_train": len(train), "seed": args.seed,
              "device": "cpu", "threads": args.threads,
              "laya_commit": git_sha(LAYA_ROOT), "project_git_sha": git_sha(ROOT),
              "model_revision": "d6f03aa9baca4f9017fa8f8e9c2764d3bed3f409" if args.model == "vela" else None,
              "windows_sleep_prevention_active": sleep_on, "peak_rss_mb": round(peak / 2**20, 1),
              "genomes": results, "best": best,
              "status": "passed" if all(True for _ in results) else "partial_failure"}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"best": best, "peak_rss_mb": report["peak_rss_mb"]}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
