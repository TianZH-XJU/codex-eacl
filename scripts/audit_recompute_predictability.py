#!/usr/bin/env python3
"""Audit appendix: recompute Gate 1 upgrade-gain predictability with FIXED stats.

- GroupKFold by true group_id (problem/function/report), NOT position shuffle.
- Majority baseline from TRAIN fold labels only.
- Cluster bootstrap CI over groups.
Reads frozen pilot caches; writes results/v2/audit/predictability_recompute.json.
No model calls. Old verdicts are not overwritten.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from group_oof import (attach_code_groups, attach_findver_groups,  # noqa: E402
                       attach_processbench_groups, cluster_bootstrap_ci,
                       group_folds, train_only_majority)

FEATS = ["initial_answer_confidence", "initial_entropy_confidence", "initial_margin",
         "initial_entropy", "initial_top1_prob", "initial_text_len", "initial_n_options"]


def load(p):
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def correctness(items, strategy):
    return int(next(x for x in items if x["strategy"] == strategy)["correct"])


def single_row(items):
    return next(x for x in items if x["strategy"] == "single")


def feats(r):
    return [float(r.get(k) or 0.0) for k in FEATS]


def run(by_idx, upgrades, seed=17):
    idxs = sorted(by_idx, key=str)
    groups = [single_row(by_idx[i])["group_id"] for i in idxs]
    folds, _ = group_folds(groups, n_splits=3, seed=seed)
    out = {}
    for action in upgrades:
        accs, majs, diffs = [], [], []
        for tr, te in folds:
            tri, tei = [idxs[i] for i in tr], [idxs[i] for i in te]
            Xtr = np.array([feats(single_row(by_idx[i])) for i in tri], dtype=float)
            ytr = np.array([int(correctness(by_idx[i], action)) - int(correctness(by_idx[i], "single")) + 1 for i in tri])
            sc = StandardScaler().fit(Xtr)
            maj = train_only_majority(ytr.tolist())
            Xte = sc.transform(np.array([feats(single_row(by_idx[i])) for i in tei], dtype=float))
            yte = np.array([int(correctness(by_idx[i], action)) - int(correctness(by_idx[i], "single")) + 1 for i in tei])
            if len(set(ytr.tolist())) < 2:
                pred = np.full_like(yte, maj)
            else:
                clf = LogisticRegression(max_iter=1000, random_state=seed)
                clf.fit(sc.transform(Xtr), ytr)
                pred = clf.predict(Xte)
            accs.append(float((pred == yte).mean()))
            majs.append(float((np.array(yte) == maj).mean()))
            diffs.extend((pred == yte).astype(float).tolist())
        te_groups, te_vals = [], []
        for k, (tr, te) in enumerate(folds):
            for i in te:
                te_groups.append(single_row(by_idx[idxs[i]])["group_id"])
        # per-fold correctness already in accs; cluster CI over all OOF diffs:
        ci = cluster_bootstrap_ci(te_groups, diffs)
        out[action] = {"fold_oof_accuracy": [round(a, 4) for a in accs],
                       "fold_majority_train_only": [round(m, 4) for m in majs],
                       "mean_oof": round(float(np.mean(accs)), 4),
                       "mean_majority": round(float(np.mean(majs)), 4),
                       "cluster_bootstrap_oof": ci}
    return out


def group(by):
    d = {}
    for r in by:
        d.setdefault((r.get("split"), r.get("position")), []).append(r)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/v2/audit/predictability_recompute.json")
    args = ap.parse_args()

    pb = group(load(ROOT / "results/processbench/pilot_train_cache.jsonl"))
    pb = {k: v for k, v in pb.items()}
    # attach groups: need flat row list per example
    flat_pb = [x for items in pb.values() for x in items if x["strategy"] == "single"]
    # rebuild per-example structure with group ids via attach on single rows is insufficient;
    # attach on full row list instead:
    full_pb = [x for items in pb.values() for x in items]
    full_pb = attach_processbench_groups(full_pb)
    by_pb = {}
    for r in full_pb:
        by_pb.setdefault((r["split"], r["position"]), []).append(r)

    code = [x for x in load(ROOT / "results/codexglue_defect/pilot_valid_cache.jsonl")]
    code = attach_code_groups(code)
    by_code = {}
    for r in code:
        by_code.setdefault((r["split"], r["position"]), []).append(r)

    fv = [x for x in load(ROOT / "results/findver/pilot_dev_cache.jsonl")]
    fv = attach_findver_groups(fv)
    by_fv = {}
    for r in fv:
        by_fv.setdefault((r["split"], r["position"]), []).append(r)

    out = {"created_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
           "method": "GroupKFold(3) by true group + train-only majority + cluster bootstrap",
           "processbench_train": run(by_pb, ["wide", "tournament"]),
           "codexglue_valid": run(by_code, ["switch", "wide"]),
           "findver_dev": run(by_fv, ["switch", "wide"])}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    for task, res in (("processbench", out["processbench_train"]), ("code", out["codexglue_valid"]), ("findver", out["findver_dev"])):
        for action, m in res.items():
            print(task, action, "oof", m["mean_oof"], "maj", m["mean_majority"], "CI", m["cluster_bootstrap_oof"]["ci95_lo"], m["cluster_bootstrap_oof"]["ci95_hi"])


if __name__ == "__main__":
    raise SystemExit(main())
