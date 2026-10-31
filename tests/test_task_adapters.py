import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tasks.base import TaskAdapter  # noqa: E402
from tasks.code import CodeAdapter  # noqa: E402
from tasks.findver import FindverAdapter  # noqa: E402
from tasks.intent import IntentAdapter  # noqa: E402
from tasks.math import SENTINEL, ProcessbenchAdapter  # noqa: E402


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


def test_code_rows_match_proven_shape_and_split_rules():
    a = CodeAdapter()
    r1 = a.load_rows("search_dev", 10, 17)
    r2 = a.load_rows("search_dev", 10, 17)
    assert [r["key"] for r in r1] == [r["key"] for r in r2]
    assert len(r1) == 10 and all(r["group_id"] for r in r1)
    assert all(isinstance(r["gold"], bool) for r in r1)
    assert a.labels() == ["secure", "vulnerable"]
    q = a.build_question(r1[0])
    assert set(q) == {"noul", "switch"}
    assert q["switch"]["criteria"] == {"secure": None, "vulnerable": None}
    with pytest.raises(ValueError):
        a.load_rows("test", 5, 17)


def test_findver_state_text_and_grouping():
    a = FindverAdapter()
    rows = a.load_rows("search_dev", 5, 17)
    assert len(rows) == 5
    assert all(r["text"].startswith("Claim: ") and "\n\nEvidence:\n" in r["text"] for r in rows)
    assert all(r["group_id"].startswith("rp:") for r in rows)
    assert all(isinstance(r["gold"], bool) for r in rows)
    # same group rows share the report prefix
    assert a.primary_metric() == "macro_f1"


def test_processbench_sentinel_last_and_gold_mapping():
    a = ProcessbenchAdapter()
    rows = a.load_rows("search_dev", 8, 17)
    assert len(rows) == 8 and all(r["group_id"] for r in rows)
    assert SENTINEL == "ALL_STEPS_CORRECT"
    for r in rows:
        cands = a.candidates(r)
        assert cands[-1] == SENTINEL
        assert r["gold"] == (r["meta"]["label"] if r["meta"]["label"] != -1 else len(r["meta"]["steps"]))
        q = a.build_question(r)
        assert list(q["criteria"])[-1] == SENTINEL.replace("_", " ")
    assert a.primary_metric() == "step_accuracy"


def test_all_adapters_reject_test_split():
    for a in (CodeAdapter(), FindverAdapter(), ProcessbenchAdapter()):
        with pytest.raises(ValueError):
            a.load_rows("test", 3, 17)
