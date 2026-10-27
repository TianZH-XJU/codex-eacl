import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_cached_errors import analyze_split


def _row(idx, strategy, correct, gold="g1", choice="c1"):
    return {"index": idx, "strategy": strategy, "correct": correct,
            "gold": gold, "gold_normalized": gold,
            "text": "t", "initial_choice": choice,
            "initial_answer_confidence": 0.5, "initial_margin": 0.1,
            "initial_entropy_confidence": 0.5, "initial_act_probability": 1.0,
            "initial_probabilities": {}, "cumulative_latency_ms": 1000.0, "cumulative_calls": 1}


def test_avoidable_errors_and_all_wrong():
    by = {0: [_row(0, "single", True), _row(0, "wide", False), _row(0, "tournament", False)],
          1: [_row(1, "single", False), _row(1, "wide", True), _row(1, "tournament", False)],
          2: [_row(2, "single", False), _row(2, "wide", False), _row(2, "tournament", False)]}
    costs = {"single": {"mean_cumulative_ms": 1.0}, "wide": {"mean_cumulative_ms": 2.0},
             "tournament": {"mean_cumulative_ms": 3.0}}
    p_num = lambda r: {"single": 0.9, "wide": 0.1, "tournament": 0.1}  # noqa: E731
    p_ch = lambda r: {"single": 0.1, "wide": 0.1, "tournament": 0.9}  # noqa: E731
    p_thr = lambda r: "single"  # noqa: E731
    out = analyze_split(by, p_num, p_ch, p_thr, costs)
    assert len(out["samples"]) == 3
    s1 = next(s for s in out["samples"] if s["index"] == 1)
    assert s1["correct_alternatives"] == ["wide"]
    assert s1["chosen"]["numeric_only"] == "single"  # wrong choice, avoidable
    pc = out["per_class"]["g1"]
    assert pc["n"] == 3 and pc["all_wrong"] == 1 and pc["oracle_correct"] == 2
    assert pc["methods"]["numeric_only"]["avoidable_errors"] == 1
    assert pc["methods"]["numeric_choice"]["avoidable_errors"] == 2  # idx0 (single right) + idx1 (wide right)
