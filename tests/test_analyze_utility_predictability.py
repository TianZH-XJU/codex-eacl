import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_utility_predictability as UP  # noqa: E402


def _row(p, gold, sw, gid="g0"):
    return {"gold": gold, "group_id": gid, "key": "k",
            "noul_p": p, "switch_correct": sw}


def test_features_use_no_switch_no_gold():
    rows = [_row(0.9, True, True), _row(0.1, False, False)]
    X = UP.features(rows)
    assert X.shape == (2, 3)
    assert abs(X[0, 0] - 0.4) < 1e-9  # |0.9-0.5|
    assert X[1, 0] == X[0, 0]  # symmetric confidence
    # mutation invariance: flipping gold/switch/group leaves X bit-identical
    import copy
    mut = copy.deepcopy(rows)
    for r in mut:
        r["gold"] = not r["gold"]
        r["switch_correct"] = not r["switch_correct"]
        r["group_id"] = "other"
    assert UP.features(mut).tolist() == X.tolist()


def test_oof_singleton_class_fallback():
    # all-negative train fold content -> guarded zero scores, no crash
    X = np.array([[0.4, 0.9, 0.3]] * 6 + [[0.1, 0.6, 0.5]] * 6)
    y = np.zeros(12, dtype=int)
    groups = np.array([f"g{i}" for i in range(12)])
    sc = UP.oof_scores(X, y, groups, "logreg")
    assert (sc == 0.0).all()
    sc = UP.oof_scores(X, y, groups, "tree")
    assert (sc == 0.0).all()


def test_champion_tie_break_documented():
    d = json.loads((ROOT / "results/v4_1/evolution/utility_predictability.json").read_text(encoding="utf-8"))
    assert "tie" in d["selection_rule"] and "logreg>tree>threshold" in d["selection_rule"]
    for k, c in d["cells"].items():
        assert "first-max" in c["champion_tie_break"] and "logreg>tree>threshold" in c["champion_tie_break"]


def test_labels_mark_only_repairs():
    rows = [_row(0.1, True, True),    # noul wrong, switch right -> 1
            _row(0.9, True, True),    # noul right -> 0
            _row(0.9, True, False),   # noul right, switch wrong -> 0
            _row(0.1, True, False)]   # both wrong -> 0
    assert UP.labels(rows).tolist() == [1, 0, 0, 0]


def test_net_utility_counts_and_cost():
    rows = [_row(0.1, True, True),
            _row(0.9, True, False),
            _row(0.9, True, True)]
    out = UP.net_utility(rows, np.array([True, True, False]))
    assert out["repair"] == 1 and out["regression"] == 1
    assert out["net"] == 0.0 and out["n_switched"] == 2
    assert out["cost_adjusted"] == round(0.0 - 0.05 * (2 / 3), 4)


def test_committed_predictability_artifact():
    d = json.loads((ROOT / "results/v4_1/evolution/utility_predictability.json").read_text(encoding="utf-8"))
    assert d["experiment_id"] == "EXP-02" and d["repo_commit"]
    assert set(d["cells"]) == {"code_laya", "code_vela", "findver_laya", "findver_vela"}
    for k, c in d["cells"].items():
        assert c["splits"] == {"train": "search_train", "dev": "search_dev", "confirm": "confirm"}
        assert c["champion"] in c["candidates"]
        for m in ("logreg", "tree", "threshold"):
            assert m in c["candidates"] or m in c["models"]
        # OOF auroc is a probability in [0,1]
        for m in ("logreg", "tree"):
            assert 0.0 <= c["models"][m]["oof_auroc"] <= 1.0
