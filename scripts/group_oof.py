#!/usr/bin/env python3
"""Group-aware OOF utilities for the post-gate diagnostic track.

- group_id derivation (offline, from row ids; no model calls)
- GroupKFold splitting with zero group overlap guarantee
- train-only majority baseline (never reads held-out labels)
- cluster bootstrap CIs (whole groups resampled together)
"""
from __future__ import annotations

import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def normalize_text(s):
    s = unicodedata.normalize("NFKC", str(s or ""))
    return "\n".join(" ".join(line.split()) for line in s.replace("\r\n", "\n").split("\n"))


def sha256_hex(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def problem_group_id(problem_text):
    return "pb:" + sha256_hex(normalize_text(problem_text))[:16]


def function_group_id(func_text):
    lines = [line.rstrip() for line in str(func_text or "").replace("\r\n", "\n").split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return "fn:" + sha256_hex("\n".join(lines))[:16]


def report_group_id(report_id):
    return "rp:" + str(report_id)


def attach_processbench_groups(rows, raw_dir="data/raw/processbench"):
    """rows: pilot-cache style dicts with row_id + source_file. Returns new list with group_id."""
    cache = {}
    for fn in ("gsm8k.json", "math.json", "olympiadbench.json", "omnimath.json"):
        for r in json.loads(Path(raw_dir, fn).read_text(encoding="utf-8")):
            cache[(fn, r["id"])] = r["problem"]
    out = []
    for r in rows:
        key = (r.get("source_file"), r.get("row_id"))
        if key not in cache:
            raise ValueError(f"row not found in raw data: {key}")
        rr = dict(r)
        rr["group_id"] = problem_group_id(cache[key])
        out.append(rr)
    return out


def attach_code_groups(rows, raw_path="data/raw/codexglue_defect/function.json"):
    js_all = json.loads(Path(raw_path).read_text(encoding="utf-8"))
    out = []
    for r in rows:
        idx = r.get("meta", {}).get("idx", r.get("row_id"))
        rr = dict(r)
        rr["group_id"] = function_group_id(js_all[int(idx)]["func"])
        out.append(rr)
    return out


def attach_findver_groups(rows, manifest_path="results/findver/controller_split_manifest.json",
                          data_path="external/FinDVer/data/testmini.json"):
    by_id = {r["example_id"]: r for r in json.loads(Path(data_path).read_text(encoding="utf-8"))}
    out = []
    for r in rows:
        eid = r.get("row_id")
        if eid not in by_id:
            raise ValueError(f"example id not found: {eid}")
        rr = dict(r)
        rr["group_id"] = report_group_id(by_id[eid]["report"])
        out.append(rr)
    return out


def group_folds(group_ids, n_splits=3, seed=17):
    """Deterministic group-disjoint folds. Returns list of (train_idx, test_idx)."""
    from sklearn.model_selection import GroupKFold
    import numpy as np  # noqa: F811
    groups = np.array(list(group_ids))
    gkf = GroupKFold(n_splits=n_splits)
    rng_state = int(np.random.RandomState(seed).randint(0, 2**31 - 1))
    # GroupKFold is deterministic; shuffle group order deterministically first.
    uniq = sorted(set(group_ids.tolist() if hasattr(group_ids, "tolist") else group_ids))
    rng = np.random.RandomState(seed)
    order = list(uniq)
    rng.shuffle(order)
    remap = {g: i for i, g in enumerate(order)}
    recoded = np.array([remap[g] for g in groups])
    folds = []
    for tr, te in gkf.split(np.zeros(len(groups)), np.zeros(len(groups)), recoded):
        assert not (set(groups[tr]) & set(groups[te])), "group leakage across fold!"
        folds.append((tr, te))
    return folds, {"shuffle_seed": seed, "gkf_random_state": rng_state}


def train_only_majority(y_train):
    """Majority class estimated from TRAIN labels only. Never sees held-out labels."""
    y = list(y_train)
    if not y:
        raise ValueError("empty train labels")
    counts = Counter(y)
    top = counts.most_common()
    best = max(c[1] for c in top)
    candidates = sorted(c for c, n in top if n == best)
    return candidates[0]


def cluster_bootstrap_ci(group_ids, values, reps=5000, seed=17):
    """Mean CI by resampling whole groups. Returns dict with group stats."""
    import numpy as np  # noqa: F811
    groups = {}
    for g, v in zip(group_ids, values):
        groups.setdefault(g, []).append(float(v))
    glist = sorted(groups)
    arr = {g: np.array(groups[g]) for g in glist}
    sizes = [len(arr[g]) for g in glist]
    total_n = sum(sizes)
    rng = np.random.RandomState(seed)
    means = []
    for _ in range(reps):
        picked = rng.choice(glist, size=len(glist), replace=True)
        vals = np.concatenate([arr[g] for g in picked])
        means.append(float(vals.mean()))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean": round(float(np.mean(means)), 4), "ci95_lo": round(float(lo), 4),
            "ci95_hi": round(float(hi), 4), "n_groups": len(glist), "n_rows": total_n,
            "group_size_min": min(sizes), "group_size_max": max(sizes), "reps": reps, "seed": seed}


def validate_paired_join(left, right, key="row_id"):
    """Join two row lists on stable IDs and verify state/gold agreement."""
    rmap = {r[key]: r for r in right}
    joined, mismatches = [], []
    for r in left:
        o = rmap.get(r[key])
        if o is None:
            mismatches.append({"row_id": r[key], "reason": "missing counterpart"})
            continue
        for field in ("state_text_sha256", "gold_sha256"):
            if field in r or field in o:
                if r.get(field) != o.get(field):
                    mismatches.append({"row_id": r[key], "reason": f"{field} mismatch"})
        joined.append((r, o))
    if mismatches:
        raise ValueError(f"paired join failed on {len(mismatches)} rows: {mismatches[:3]}")
    return joined
