import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from measure_vela_batching_distinct import continuous_delta, decision


def test_choice_decision_ignores_probability_shift():
    sep = {"type": "choice", "choice": "supported",
           "probabilities": {"supported": 0.924, "not_supported": 0.076}}
    bat = {"type": "choice", "choice": "supported",
           "probabilities": {"supported": 0.747, "not_supported": 0.253}}
    assert decision(sep) == decision(bat) == ["choice", "supported"]
    assert sep != bat  # raw dicts differ: float-sensitive equality must not be used
    d = continuous_delta(sep, bat)
    assert abs(d["l1_prob"] - 0.354) < 0.01


def test_noul_decision_is_threshold_only():
    sep = {"type": "noul", "noul": 0.799}
    bat = {"type": "noul", "noul": 0.666}
    assert decision(sep) == decision(bat) == ["noul", "pos"]
    assert sep != bat
    assert continuous_delta(sep, bat)["abs_p_diff"] > 0.1


def test_score_decision_is_argmax_level():
    sep = {"type": "score", "score": 2.275,
           "probabilities": {"0": 0.068, "1": 0.249, "2": 0.230, "3": 0.246, "4": 0.207}}
    bat = {"type": "score", "score": 2.114,
           "probabilities": {"0": 0.112, "1": 0.270, "2": 0.213, "3": 0.206, "4": 0.201}}
    assert decision(sep) == decision(bat) == ["score", "1"]
    assert continuous_delta(sep, bat)["abs_expected_diff"] > 0.1


def test_unknown_and_non_dict():
    assert decision(None) is None
    assert decision({"type": "mystery"}) == ["unknown", None]
    assert continuous_delta(None, {}) is None
