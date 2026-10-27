import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_controller_baselines import FEATURE_NAMES, features, upgrade_needed


def _row(strategy, correct, **kw):
    r = {
        "strategy": strategy,
        "correct": correct,
        "cumulative_calls": 1,
        "cumulative_latency_ms": 1000.0,
        "text": "hello world",
        "initial_choice": "some choice",
        "initial_answer_confidence": 0.9,
        "initial_margin": 0.5,
        "initial_entropy_confidence": 0.3,
        "initial_act_probability": 1.0,
    }
    r.update(kw)
    return r


def test_upgrade_label_matches_concrete_target():
    # SINGLE wrong, WIDE right, TOURNAMENT wrong -> upgrade to TOURNAMENT is 0.
    items = [_row("single", False), _row("wide", True), _row("tournament", False)]
    assert upgrade_needed(items, "tournament") == 0
    assert upgrade_needed(items, "wide") == 1
    # SINGLE right -> never upgrade regardless of target.
    items2 = [_row("single", True), _row("wide", True), _row("tournament", True)]
    assert upgrade_needed(items2, "tournament") == 0
    assert upgrade_needed(items2, "wide") == 0
    # SINGLE wrong, target right -> upgrade.
    items3 = [_row("single", False), _row("wide", False), _row("tournament", True)]
    assert upgrade_needed(items3, "tournament") == 1
    assert upgrade_needed(items3, "wide") == 0


def test_features_are_locked_and_finite():
    r = _row("single", True)
    vals = features(r)
    assert len(vals) == len(FEATURE_NAMES) == 6
    assert FEATURE_NAMES == [
        "answer_confidence",
        "margin",
        "entropy_confidence",
        "act_probability",
        "text_len",
        "choice_name_len",
    ]
    assert all(v == v and abs(v) != float("inf") for v in vals)  # no NaN/inf
    # text_len and choice_name_len reflect input metadata.
    assert vals[4] == len("hello world")
    assert vals[5] == len("some choice")
