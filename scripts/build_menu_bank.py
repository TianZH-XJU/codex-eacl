#!/usr/bin/env python3
"""E2 menu bank prerequisite: cheap retrieval recall@K on v2 search pools (offline).

TF-IDF word, char n-gram, and random baselines rank the full label set per
row; report Recall@K (K=1/5/10/20/40/full) + macro recall. If cheap methods
already reach high recall, a GP menu has headroom worth evolving; if not,
EvoMenu needs stronger features first. No model calls.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]


def load_pool(dataset, split="search_train", limit=None, seed=17):
    manifest = json.loads((ROOT / "results/v4/data_manifest" / f"{dataset}_groups.json").read_text(encoding="utf-8"))
    import pyarrow.parquet as pq
    if dataset == "clinc150":
        base = ROOT / "data/raw/clinc150/plus/train-00000-of-00001.parquet"
        names = json.loads(pq.read_schema(base).metadata[b"huggingface"])["info"]["features"]["intent"]["names"]
        rows_all = pq.read_table(base).to_pylist()
        pool = []
        for g in manifest["splits"][split]:
            for i in manifest["members"].get(g, []):
                r = rows_all[i]
                pool.append({"key": f"train:{i}", "text": r["text"], "label": names[r["intent"]], "group": g})
    else:
        import pyarrow.parquet as pq2
        rows_all = pq2.read_table(ROOT / "data/raw/banking77/data/train-00000-of-00001.parquet").to_pylist()
        pool = []
        for g in manifest["splits"][split]:
            for i in manifest["members"].get(g, []):
                r = rows_all[i]
                pool.append({"key": f"train:{i}", "text": r["text"], "label": r["label_text"], "group": g})
    rng = random.Random(seed)
    rng.shuffle(pool)
    if limit:
        # stratified-ish: round-robin over labels to keep coverage
        by_label = defaultdict(list)
        for r in pool:
            by_label[r["label"]].append(r)
        labs = sorted(by_label)
        out = []
        i = 0
        while len(out) < min(limit, len(pool)):
            for lab in labs:
                if by_label[lab] and len(out) < limit:
                    out.append(by_label[lab].pop(0))
            i += 1
            if i > limit + 5:
                break
        return out
    return pool


def rank_labels(train_texts, train_labels, query_texts, labels, kind="tfidf-word"):
    if kind == "tfidf-word":
        vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    elif kind == "tfidf-char":
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=30000)
    else:
        raise ValueError(kind)
    Xtr = vec.fit_transform(train_texts)
    Xte = vec.transform(query_texts)
    # centroid per label
    uniq = sorted(set(train_labels))
    centroids = {}
    for lab in uniq:
        idx = [i for i, l in enumerate(train_labels) if l == lab]
        centroids[lab] = np.asarray(Xtr[idx].mean(axis=0)).ravel()
    C = np.stack([centroids[l] for l in uniq])
    S = cosine_similarity(Xte, C)
    order = np.argsort(-S, axis=1)
    return [[uniq[j] for j in row] for row in order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k-list", default="1,5,10,20,40")
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-test", type=int, default=300)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--dump-menus", type=int, default=0,
                    help="if >0, write per-row top-K menus for the dev pool to --menu-out")
    ap.add_argument("--menu-out", default="results/v2/menu/menus.json")
    ap.add_argument("--out", default="results/v2/menu/recall_bank.json")
    args = ap.parse_args()
    ks = [int(x) for x in args.k_list.split(",")]

    out = {"created_utc": datetime.now(timezone.utc).isoformat(), "tasks": {}, "menus": {}}
    for ds in ("clinc150", "banking77"):
        train = load_pool(ds, "search_train", args.n_train, args.seed)
        dev = load_pool(ds, "search_dev", args.n_test, args.seed + 1)
        # fit on train pool only; evaluate on dev pool (group-disjoint by construction)
        labels = sorted({r["label"] for r in train})
        res = {}
        for kind in ("tfidf-word", "tfidf-char"):
            ranked = rank_labels([r["text"] for r in train], [r["label"] for r in train],
                                 [r["text"] for r in dev], labels, kind)
            rec = {}
            for k in ks + ["full"]:
                kk = len(labels) if k == "full" else k
                hits = sum(1 for preds, r in zip(ranked, dev) if r["label"] in preds[:kk])
                rec[str(k)] = round(hits / len(dev), 4) if dev else None
            # macro recall at K=10
            per_lab = {}
            for lab in sorted({r["label"] for r in dev}):
                idx = [i for i, r in enumerate(dev) if r["label"] == lab]
                per_lab[lab] = round(sum(1 for i in idx if dev[i]["label"] in ranked[i][:10]) / max(len(idx), 1), 4)
            res[kind] = {"recall_at_k": rec, "macro_recall_at_10": round(float(np.mean(list(per_lab.values()))), 4) if per_lab else None}
        # random baseline
        rng = random.Random(args.seed)
        rr = []
        for r in dev:
            rr.append(r["label"] in rng.sample(labels, min(10, len(labels))))
        res["random"] = {"recall_at_10": round(sum(rr) / len(rr), 4) if rr else None}
        # majority-class retrieval (always predict train-majority label)
        from collections import Counter
        maj = Counter(r["label"] for r in train).most_common(1)[0][0]
        res["majority_label"] = {"label": maj, "dev_hit_rate": round(sum(1 for r in dev if r["label"] == maj) / len(dev), 4) if dev else None}
        out["tasks"][ds] = {"n_train": len(train), "n_dev": len(dev), "n_labels": len(labels), "methods": res}
        print(ds, json.dumps(res, indent=1)[:800])
        if args.dump_menus:
            ranked = rank_labels([r["text"] for r in train], [r["label"] for r in train],
                                 [r["text"] for r in dev], labels, "tfidf-word")
            out["menus"][ds] = [{"key": r["key"], "gold": r["label"], "group_id": r["group"], "text": r["text"],
                                 f"top{args.dump_menus}": preds[:args.dump_menus],
                                 "recall_hit": bool(r["label"] in preds[:args.dump_menus])}
                                for r, preds in zip(dev, ranked)]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
