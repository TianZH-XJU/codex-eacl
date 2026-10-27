import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from calibrate_temperature import dev_metrics, nll_of_rows, tempered


def _row(probs, gold, correct=True):
    return {"initial_probabilities": probs, "gold_normalized": gold, "correct": correct}


def test_tempered_is_distribution_and_T1_identity():
    probs = {"a": 0.7, "b": 0.2, "c": 0.1}
    pt = tempered(probs, 1.0)
    assert abs(sum(pt.values()) - 1.0) < 1e-9
    for k in probs:
        assert abs(pt[k] - probs[k]) < 1e-9
    # high T flattens toward uniform
    flat = tempered(probs, 10.0)
    assert max(flat.values()) < max(probs.values())


def test_nll_prefers_true_label_mass():
    good = [_row({"a": 0.9, "b": 0.1}, "a")]
    bad = [_row({"a": 0.1, "b": 0.9}, "a")]
    assert nll_of_rows(good, 1.0) < nll_of_rows(bad, 1.0)


def test_ece_zero_for_perfect_confident_predictions():
    rows = [_row({"a": 1.0, "b": 0.0}, "a", True) for _ in range(10)]
    m = dev_metrics(rows, 1.0)
    assert m["ece_15bin"] == 0.0
    assert m["nll"] == 0.0
