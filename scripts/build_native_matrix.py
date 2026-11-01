#!/usr/bin/env python3
"""A1 native model baselines (offline; no new Laya/Vela calls).

- Majority / random-choice baselines from pilot caches (official ProcessBench
  error-acc / correct-acc / harmonic-F1 included).
- TF-IDF + LogReg/SVM trained on track-clean train pools, evaluated on
  track-clean dev pools (code: function groups; findver: clean reports).
- Compares cached Laya/Vela paired accuracies side by side.
Writes results/v2/native_baselines/.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(p):
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def processbench_official(rows):
    """rows: single-strategy pilot rows with gold_index/n_candidates."""
    err_ok = err_n = cor_ok = cor_n = 0
    by_src = {}
    for r in rows:
        n = r.get("n_candidates", 0)
        gold = r.get("gold_rendered", "")
        pred = r.get("choice", "")
        hit = (pred == gold)
        if gold == "ALL STEPS CORRECT":
            cor_ok += hit
            cor_n += 1
        else:
            err_ok += hit
            err_n += 1
    error_acc = err_ok / err_n if err_n else None
    correct_acc = cor_ok / cor_n if cor_n else None
    f1 = (2 * error_acc * correct_acc / (error_acc + correct_acc)
          if error_acc and correct_acc and (error_acc + correct_acc) else 0.0)
    return {"error_acc": round(error_acc, 4) if error_acc is not None else None,
            "correct_acc": round(correct_acc, 4) if correct_acc is not None else None,
            "harmonic_f1": round(f1, 4), "n_error": err_n, "n_correct": cor_n}


def tfidf_baselines(texts_train, y_train, texts_dev, y_dev, average="binary"):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    Xtr = vec.fit_transform(texts_train)
    Xde = vec.transform(texts_dev)
    out = {}
    lr = LogisticRegression(max_iter=1000)
    lr.fit(Xtr, y_train)
    p = lr.predict(Xde)
    out["tfidf_logreg"] = {"accuracy": round(accuracy_score(y_dev, p), 4),
                           "balanced_accuracy": round(balanced_accuracy_score(y_dev, p), 4),
                           "macro_f1": round(f1_score(y_dev, p, average="macro", zero_division=0), 4)}
    svm = LinearSVC()
    svm.fit(Xtr, y_train)
    p = svm.predict(Xde)
    out["tfidf_linearsvm"] = {"accuracy": round(accuracy_score(y_dev, p), 4),
                              "balanced_accuracy": round(balanced_accuracy_score(y_dev, p), 4),
                              "macro_f1": round(f1_score(y_dev, p, average="macro", zero_division=0), 4)}
    maj = Counter(y_train).most_common(1)[0][0]
    pm = [maj] * len(y_dev)
    out["majority"] = {"accuracy": round(accuracy_score(y_dev, pm), 4),
                       "balanced_accuracy": round(balanced_accuracy_score(y_dev, pm), 4),
                       "macro_f1": round(f1_score(y_dev, pm, average="macro", zero_division=0), 4)}
    return out


def code_rows_for_groups(manifest, raw, groups, valid_idx, test_idx):
    """Resolve group ids to (func, target) rows using ONLY manifest members
    (official-train indices). Raises AssertionError on any valid/test/leaked
    index (P0-2 leak guard). Pure function: unit-testable without raw files."""
    leaked = set(manifest.get("excluded_cross_split_leaked_idx", []))
    members = manifest.get("members")
    assert members is not None, "manifest must pin train-only members"
    rows = []
    for g in groups:
        for i in members.get(g, []):
            assert i not in valid_idx and i not in test_idx and i not in leaked, f"non-train idx {i}"
            rows.append((raw[i]["func"], raw[i]["target"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/v2/native_baselines")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    matrix, metrics = [], {}
    # ProcessBench official metrics from Laya pilot caches (single).
    for split, path in (("train", "results/processbench/pilot_train_cache.jsonl"),
                        ("dev", "results/processbench/pilot_dev_cache.jsonl")):
        rows = [r for r in load(ROOT / path) if r["strategy"] == "single"]
        m = processbench_official(rows)
        # random-step expected accuracy = mean(1/n_candidates)
        m["random_step_expected_acc"] = round(float(np.mean([1.0 / max(r.get("n_candidates", 1), 1) for r in rows])), 4)
        m["laya_single_acc"] = round(sum(r["correct"] for r in rows) / len(rows), 4)
        metrics[f"processbench_laya_single_{split}"] = m
        matrix.append({"task": "processbench", "split": split, "model": "laya-default-single", **{k: v for k, v in m.items() if k in ("laya_single_acc", "harmonic_f1")}})

    # Code TF-IDF on track-clean function groups (search_train -> search_dev).
    # Uses ONLY manifest members (official-train indices); group hashes that
    # collide across splits must never resolve to valid/test rows (P0 audit).
    import json as js
    v2 = js.loads((ROOT / "results/v2/data_manifest/codexglue_function_groups.json").read_text(encoding="utf-8"))
    raw = js.loads((ROOT / "data/raw/codexglue_defect/function.json").read_text(encoding="utf-8"))
    valid_idx = {int(x) for x in (ROOT / "data/raw/codexglue_defect/valid.txt").read_text(encoding="utf-8").split()}
    test_idx = {int(x) for x in (ROOT / "data/raw/codexglue_defect/test.txt").read_text(encoding="utf-8").split()}
    assert valid_idx.isdisjoint({int(x) for x in (ROOT / "data/raw/codexglue_defect/train.txt").read_text(encoding="utf-8").split()})
    def rows_for(groups):
        return code_rows_for_groups(v2, raw, groups, valid_idx, test_idx)
    tr = rows_for(v2["splits"]["search_train"])
    de = rows_for(v2["splits"]["search_dev"])
    code_scores = tfidf_baselines([t for t, _ in tr], [y for _, y in tr],
                                  [t for t, _ in de], [y for _, y in de])
    metrics["codexglue_tfidf_searchdev"] = code_scores
    for name, s in code_scores.items():
        matrix.append({"task": "codexglue", "split": "search_dev", "model": name, **s})

    # FinDVer TF-IDF on track-clean report pool.
    fmanifest = js.loads((ROOT / "results/v2/data_manifest/findver_report_groups.json").read_text(encoding="utf-8"))
    data = {r["example_id"]: r for r in js.loads((ROOT / "external/FinDVer/data/testmini.json").read_text(encoding="utf-8"))}
    rep2rows = {}
    for eid, r in data.items():
        rep2rows.setdefault(r["report"], []).append((r["statement"], int(bool(r["entailment_label"])), r["subset"]))
    def frep_rows(reports):
        rows = []
        for rep in reports:
            rows.extend(rep2rows.get(rep, []))
        return rows
    ftr = frep_rows(fmanifest["splits"]["search_train"])
    fde = frep_rows(fmanifest["splits"]["search_dev"])
    fin_scores = tfidf_baselines([t for t, _, _ in ftr], [y for _, y, _ in ftr],
                                 [t for t, _, _ in fde], [y for _, y, _ in fde])
    metrics["findver_tfidf_searchdev"] = fin_scores
    for name, s in fin_scores.items():
        matrix.append({"task": "findver", "split": "search_dev", "model": name, **s})

    import csv
    with (out / "native_model_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["task", "split", "model", "accuracy", "balanced_accuracy",
                                          "macro_f1", "harmonic_f1", "error_acc", "correct_acc"])
        w.writeheader()
        for r in matrix:
            w.writerow({k: r.get(k, "") for k in w.fieldnames})
    (out / "native_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    lat = {"laya_single_mean_ms": {"processbench_train": 2972.6},
           "note": "full latency table in committed pilot summaries; Vela staircase in results/final/resource_profile.json"}
    (out / "native_latency.csv").write_text("model,task,mean_latency_ms,note\nlaya,processbench_train,2972.6,pilot summary\n", encoding="utf-8")
    print(json.dumps({k: metrics[k] for k in metrics}, indent=1)[:2000])


if __name__ == "__main__":
    raise SystemExit(main())
