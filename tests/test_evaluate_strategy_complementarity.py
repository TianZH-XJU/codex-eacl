import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_strategy_complementarity import delta_correct, group


def _row(split, pos, strategy, correct, **kw):
    r = {"split": split, "position": pos, "strategy": strategy, "correct": correct,
         "cumulative_latency_ms": 100.0, "choice": "c",
         "initial_answer_confidence": 0.5, "initial_entropy_confidence": 0.5,
         "initial_margin": 0.1, "initial_entropy": 0.5, "initial_top1_prob": 0.6,
         "initial_text_len": 10, "initial_n_options": 5}
    r.update(kw)
    return r


def test_delta_correct_values():
    rows = [_row("pilot_train", 0, "single", True), _row("pilot_train", 0, "wide", False),
            _row("pilot_train", 0, "tournament", True)]
    by = group(rows)
    assert delta_correct(by, ("pilot_train", 0), "wide") == -1
    assert delta_correct(by, ("pilot_train", 0), "tournament") == 0
    rows2 = [_row("pilot_train", 1, "single", False), _row("pilot_train", 1, "wide", True),
             _row("pilot_train", 1, "tournament", False)]
    by2 = group(rows2)
    assert delta_correct(by2, ("pilot_train", 1), "wide") == +1
