import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_cv_stability import build_folds, stratified_bootstrap_ci


def _by(n_classes=4, per_class=3):
    by = {}
    k = 0
    for c in range(n_classes):
        for j in range(per_class):
            gold = f"class_{c}"
            by[k] = [{"index": k, "strategy": "single", "gold_normalized": gold,
                      "correct": True, "cumulative_latency_ms": 1000.0, "cumulative_calls": 1}]
            k += 1
    return by


def test_folds_cover_each_sample_once():
    by = _by()
    folds = build_folds(by)
    assert len(folds) == 3
    seen = []
    for f in folds:
        assert len(f["test"]) == 4  # one per class
        assert len(f["train"]) == 8
        assert not set(f["train"]) & set(f["test"])
        seen.extend(f["test"])
    assert sorted(seen) == sorted(by.keys())


def test_bootstrap_ci_covers_zero_for_null_diff():
    golds = ["a"] * 10 + ["b"] * 10
    diff = [0] * 20
    ci = stratified_bootstrap_ci(golds, diff, reps=200, seed=0)
    assert ci["ci95_lo"] == 0.0 and ci["ci95_hi"] == 0.0


def test_bootstrap_ci_positive_for_clear_gain():
    golds = ["a"] * 10 + ["b"] * 10
    diff = [1] * 20
    ci = stratified_bootstrap_ci(golds, diff, reps=200, seed=0)
    assert ci["ci95_lo"] > 0.9
