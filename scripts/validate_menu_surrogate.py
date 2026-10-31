#!/usr/bin/env python3
"""Validate the restricted-menu surrogate (offline analysis + small paired run).

For each row: (a) full-menu real call with FULL probabilities saved;
(b) restricted-menu real calls for K in {10, 20} and a random-K=10 control;
compare restricted-real vs cached-restriction (argmax over full probs).
Writes results/v2/menu/surrogate_validation.json. No test rows.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil  # noqa: E402
import torch  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
LAYA_ROOT = ROOT / "external" / "laya"
sys.path.insert(0, str(LAYA_ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

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


def tfidf_menus(train_texts, train_labels, dev_texts, labels, k_list, seed=17):
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    Xtr = vec.fit_transform(train_texts)
    Xte = vec.transform(dev_texts)
    uniq = sorted(set(train_labels))
    C = np.asarray([np.asarray(Xtr[[i for i, l in enumerate(train_labels) if l == u]]).mean(axis=0).ravel() for u in uniq])
    S = cosine_similarity(Xte, C)
    order = np.argsort(-S, axis=1)
    rng = random.Random(seed)
    out = {}
    for idx in range(len(dev_texts)):
        ranked = [uniq[j] for j in order[idx]]
        menus = {f"tfidf_top{k}": ranked[:k] for k in k_list}
        # random menu: sample K=10 from full label set
        menus["random_top10"] = rng.sample(labels, 10)
        out[idx] = menus
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["banking77", "clinc150"])
    ap.add_argument("--model", required=True, choices=["laya", "vela"])
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_available():
        raise RuntimeError("CPU-only run refused: CUDA is available")

    from group_oof import normalize_text  # noqa: E402
    import pyarrow.parquet as pq
    if args.dataset == "banking77":
        raw = pq.read_table(ROOT / "data/raw/banking77/data/train-00000-of-00001.parquet").to_pylist()
        rows_all = [{"text": r["text"], "label": r["label_text"]} for r in raw]
        labels = sorted({r["label_text"] for r in raw})
    else:
        base = ROOT / "data/raw/clinc150/plus/train-00000-of-00001.parquet"
        names = json.loads(pq.read_schema(base).metadata[b"huggingface"])["info"]["features"]["names"]
        raw = pq.read_table(base).to_pylist()
        rows_all = [{"text": r["text"], "label": names[r["intent"]]} for r in raw]
        labels = sorted(set(names))

    manifest = json.loads((ROOT / "results/v4/data_manifest" / f"{args.dataset}_groups.json").read_text(encoding="utf-8"))
    # train pool for TF-IDF fit: search_train groups (cap 1500 rows); dev rows: search_dev pool
    def rows_of(groups):
        out = []
        for g in groups:
            for i in manifest["members"].get(g, []):
                out.append({**rows_all[i], "group": g, "idx": i})
        return out
    train_pool = rows_of(manifest["splits"]["search_train"])
    rng = random.Random(args.seed)
    rng.shuffle(train_pool)
    train_pool = train_pool[:1500]
    dev_pool = rows_of(manifest["splits"]["search_dev"])
    rng.shuffle(dev_pool)
    dev_rows = dev_pool[:args.limit]

    menus = tfidf_menus([r["text"] for r in train_pool], [r["label"] for r in train_pool],
                        [r["text"] for r in dev_rows], labels, [10, 20], args.seed)

    if args.model == "laya":
        sys.path.insert(0, str(LAYA_ROOT / "research" / "benchmarks" / "tournament"))
        import laya  # noqa: E402
        agent = laya.load("english", device="cpu")
        def call(text, criteria):
            return agent.predict({"message": text},
                                 {"q": {"type": "choice", "instructions": "Which intent does this message express?",
                                        "criteria": criteria}})["answers"]["q"]
    else:
        from transformers import AutoModel
        agent = AutoModel.from_pretrained(str(ROOT / "models/vela"), trust_remote_code=True)
        def call(text, criteria):
            return agent.system_one(text, {"q": {"type": "choice",
                "instructions": "Which intent does this message express?", "criteria": criteria}})["answers"]["q"]

    def norm(c):
        return str(c or "").replace(" ", "_").lower()

    proc = psutil.Process()
    import os
    peak = proc.memory_info().rss
    sleep_on = prevent_sleep()
    results = []
    try:
        for pos, row in enumerate(dev_rows):
            # (a) full menu, save FULL probabilities
            t0 = time.perf_counter()
            full = call(row["text"], {c.replace("_", " "): None for c in labels})
            full_ms = (time.perf_counter() - t0) * 1000.0
            full_probs = dict(full.get("probabilities", {}) or {})
            full_choice = norm(full.get("choice"))
            # (b) restricted real calls
            variants = {"tfidf_top10": menus[pos]["tfidf_top10"], "tfidf_top20": menus[pos]["tfidf_top20"],
                        "random_top10": menus[pos]["random_top10"]}
            for vname, menu in variants.items():
                t0 = time.perf_counter()
                err, ans = None, None
                try:
                    ans = call(row["text"], {str(c).replace("_", " "): None for c in menu})
                except Exception as e:  # noqa: BLE001
                    err = f"{type(e).__name__}: {e}"
                ms = (time.perf_counter() - t0) * 1000.0
                choice = norm(ans.get("choice")) if ans else None
                # (c) cached restriction: argmax over full probs restricted to menu
                keymap = {k.replace(" ", "_").lower(): k for k in (full_probs or {})}
                cands = [(full_probs.get(k, 0.0), k) for k in keymap for m in [menu]
                         if k in {str(x).replace(" ", "_").lower() for x in m}]
                # map menu labels to probability keys
                scored = []
                for m in menu:
                    mk = str(m).replace(" ", "_").lower()
                    hit = [kk for kk in (full_probs or {}) if kk.replace(" ", "_").lower() == mk]
                    scored.append((float(full_probs.get(hit[0], 0.0)) if hit else 0.0, m))
                cached = max(scored, key=lambda t: t[0])[1] if scored else None
                gold = row["label"].replace(" ", "_").lower()
                results.append({"position": pos, "gold": gold, "variant": vname, "menu_size": len(menu),
                                "restricted_real_choice": choice, "restricted_real_correct": bool(choice == gold),
                                "cached_restricted_choice": str(cached).replace(" ", "_").lower() if cached else None,
                                "restricted_ms": round(ms, 1), "full_ms": round(full_ms, 1),
                                "error": err,
                                "agree_real_vs_cached": bool(choice and cached and norm(choice) == str(cached).replace(" ", "_").lower())})
                peak = max(peak, proc.memory_info().rss)
            # full answer record
            results.append({"position": pos, "gold": row["label"].replace(" ", "_").lower(), "variant": "full",
                            "menu_size": len(labels), "restricted_real_choice": full_choice,
                            "restricted_real_correct": bool(full_choice == row["label"].replace(" ", "_").lower()),
                            "cached_restricted_choice": full_choice, "restricted_ms": round(full_ms, 1),
                            "error": None, "agree_real_vs_cached": True})
            print(f"{pos+1}/{len(dev_rows)} done", flush=True)
    finally:
        allow_sleep()

    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "task": args.dataset, "model": args.model,
              "n": len(dev_rows), "variants": ["full", "tfidf_top10", "tfidf_top20", "random_top10"],
              "device": "cpu", "threads": args.threads, "laya_commit": git_sha(LAYA_ROOT),
              "project_git_sha": git_sha(ROOT), "windows_sleep_prevention_active": sleep_on,
              "peak_rss_mb": round(peak / 2**20, 1),
              "results": results, "status": "passed" if all(r["error"] is None for r in results) else "partial_failure"}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for v in ("full", "tfidf_top10", "tfidf_top20", "random_top10"):
        sub = [r for r in results if r["variant"] == v and r["error"] is None]
        print(v, "acc:", round(sum(r["restricted_real_correct"] for r in sub) / len(sub), 4) if sub else None)
    agree = [r["agree_real_vs_cached"] for r in results if r["variant"].startswith("tfidf") and r["error"] is None]
    print("restricted-real vs cached-restricted agreement:", round(sum(agree) / len(agree), 4) if agree else None)


if __name__ == "__main__":
    raise SystemExit(main())
