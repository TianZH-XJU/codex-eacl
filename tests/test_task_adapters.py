import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tasks.base import TaskAdapter  # noqa: E402
from tasks.intent import IntentAdapter  # noqa: E402


def test_intent_adapter_rejects_test_splits():
    a = IntentAdapter("banking77")
    with pytest.raises(ValueError):
        a.load_rows("test", 5, 17)
    with pytest.raises(ValueError):
        IntentAdapter("massive")


def test_intent_full_label_sets():
    assert len(IntentAdapter("banking77").labels()) == 77
    assert len(IntentAdapter("clinc150").labels()) == 151


def test_intent_rows_deterministic_and_grouped():
    a = IntentAdapter("banking77")
    r1 = a.load_rows("search_train", 10, 17)
    r2 = a.load_rows("search_train", 10, 17)
    assert [r["key"] for r in r1] == [r["key"] for r in r2]
    assert all(r["group"] for r in r1)
    assert a.primary_metric() == "accuracy"


def test_menu_question_always_contains_gold():
    a = IntentAdapter("banking77")
    row = a.load_rows("search_train", 1, 17)[0]
    q = a.build_question(row, menu=["pay_bill", "cancel"])
    assert row["label"].replace("_", " ") in q["criteria"] or row["label"] in [
        c.replace(" ", "_") for c in q["criteria"]]
    qfull = a.build_question(row)
    assert len(qfull["criteria"]) == 77
