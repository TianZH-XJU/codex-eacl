import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_processbench import ALL_CORRECT_SENTINEL, to_choice_task, validate_row


def _row(label, n_steps=4):
    return {"id": "x", "problem": "p", "steps": [f"s{k}" for k in range(n_steps)],
            "label": label, "generator": "g", "final_answer_correct": True}


def test_all_correct_maps_to_sentinel_last():
    t = to_choice_task(_row(-1, 4))
    assert t["candidates"][-1] == ALL_CORRECT_SENTINEL
    assert t["gold_index"] == 4


def test_error_label_maps_to_zero_based_step():
    t = to_choice_task(_row(0, 4))
    assert t["gold_index"] == 0
    assert to_choice_task(_row(3, 4))["gold_index"] == 3


def test_validate_row_rejects_bad_labels():
    assert validate_row(_row(-1))[0] is True
    assert validate_row(_row(2, 4))[0] is True
    assert validate_row(_row(4, 4))[0] is False  # out of range
    assert validate_row(_row("1", 4))[0] is False  # not int
    bad = _row(0, 0)
    bad["steps"] = []
    assert validate_row(bad)[0] is False
