import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from group_oof import (cluster_bootstrap_ci, function_group_id, group_folds,
                       problem_group_id, report_group_id, train_only_majority,
                       validate_paired_join)


def test_group_folds_have_zero_group_overlap():
    # 4 problems x 3 rows: every fold must keep each problem wholly in train or test.
    gids = []
    for p in range(4):
        gids.extend([f"p{p}"] * 3)
    folds, _ = group_folds(gids, n_splits=2, seed=17)
    assert len(folds) == 2
    seen_test = set()
    for tr, te in folds:
        assert not (set(np.array(gids)[tr]) & set(np.array(gids)[te]))
        seen_test.update(te.tolist())
    assert sorted(seen_test) == list(range(12))


def test_majority_baseline_uses_train_labels_only():
    ytr = [0, 0, 0, 1]
    assert train_only_majority(ytr) == 0
    # Changing held-out labels must not change the train-only prediction.
    assert train_only_majority(ytr) == train_only_majority([0, 0, 0, 1])
    ytr2 = [1, 1, 0]
    assert train_only_majority(ytr2) == 1


def test_cluster_bootstrap_resamples_groups():
    # Two groups with opposite values: cluster CI must be wide (covers both),
    # while a naive row bootstrap would be tighter. Check group accounting.
    gids = ["a"] * 5 + ["b"] * 5
    vals = [1.0] * 5 + [0.0] * 5
    ci = cluster_bootstrap_ci(gids, vals, reps=500, seed=0)
    assert ci["n_groups"] == 2 and ci["n_rows"] == 10
    assert ci["ci95_lo"] <= 0.3 and ci["ci95_hi"] >= 0.7


def test_group_id_functions_are_stable():
    # whitespace collapse within lines; newlines are preserved as separators.
    assert problem_group_id("  Hello   world ") == problem_group_id("Hello world")
    assert problem_group_id("Hello\nworld") != problem_group_id("Hello world")
    assert function_group_id("f(){\n}\n") == function_group_id("f(){\n}")
    assert report_group_id("R.json") == "rp:R.json"


def test_paired_join_rejects_state_hash_mismatch():
    left = [{"row_id": "a", "state_text_sha256": "x", "gold_sha256": "g"}]
    right = [{"row_id": "a", "state_text_sha256": "DIFFERENT", "gold_sha256": "g"}]
    try:
        validate_paired_join(left, right)
    except ValueError as e:
        assert "state_text_sha256 mismatch" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_paired_join_stable_row_id_order_independent():
    left = [{"row_id": "a"}, {"row_id": "b"}]
    right = [{"row_id": "b"}, {"row_id": "a"}]
    joined = validate_paired_join(left, right)
    assert [a["row_id"] for a, _ in joined] == ["a", "b"]
