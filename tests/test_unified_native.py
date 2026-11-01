import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_unified_native as U  # noqa: E402
from models.registry import get_adapter  # noqa: E402


def test_task_maps_cover_everything():
    assert set(U.TASKS) == set(U.REPS) == set(U.MODEL_TASKS)
    for t, mt in U.MODEL_TASKS.items():
        for m in ("laya", "vela"):
            allowed = get_adapter(m).supported_outputs(mt)
            for rep in U.REPS[t]:
                want = "choice" if rep in ("choice", "switch") else rep
                assert want in allowed, (m, t, rep)


def test_correctness_rules():
    nest = lambda d: {"q": d}  # noqa: E731 - single-question API envelope
    assert U.unwrap(nest({"type": "choice", "choice": "a"})) == {"type": "choice", "choice": "a"}
    assert U.unwrap({"type": "noul", "noul": 0.1}) == {"type": "noul", "noul": 0.1}
    assert U.unwrap(None) == {} and U.unwrap({}) == {}
    # runner sequence: correctness(task, rep, row, unwrap(ans))
    scor = lambda t, r, row, a: U.correctness(t, r, row, U.unwrap(a))  # noqa: E731
    assert scor("banking77_intent", "choice",
                {"label": "pay_bill", "gold": None, "meta": {}},
                nest({"choice": "pay bill"}))
    assert not scor("banking77_intent", "choice",
                     {"label": "pay_bill", "gold": None, "meta": {}},
                     nest({"choice": "cancel"}))
    assert U.correctness("codexglue_defect", "noul", {"gold": True}, {"noul": 0.9})
    assert not U.correctness("codexglue_defect", "noul", {"gold": True}, {"noul": 0.1})
    assert U.correctness("codexglue_defect", "switch", {"gold": False}, {"choice": "secure"})
    assert U.correctness("findver", "switch", {"gold": True}, {"choice": "supported"})
    assert U.correctness("processbench", "choice",
                         {"gold": 2, "meta": {"steps": ["a", "b", "c"]}}, {"choice": "step 2"})
    assert U.correctness("processbench", "choice",
                         {"gold": 1, "meta": {"steps": ["a"]}},
                         {"choice": "ALL STEPS CORRECT"})
    assert not U.correctness("findver", "noul", {"gold": True}, {})


def test_findver_choice_normalization_variants():
    n = U.norm
    assert n("not_supported") == n("not supported") == n("Not Supported") == "not supported"
    assert n("SUPPORTED") == "supported"
    row_neg = {"gold": False}
    row_pos = {"gold": True}
    assert U.correctness("findver", "switch", row_neg, {"choice": "not_supported"}) is True
    assert U.correctness("findver", "switch", row_neg, {"choice": "not supported"}) is True
    assert U.correctness("findver", "switch", row_neg, {"choice": "Not Supported"}) is True
    assert U.correctness("findver", "switch", row_neg, {"choice": "supported"}) is False
    assert U.correctness("findver", "switch", row_pos, {"choice": "supported"}) is True
    assert U.correctness("findver", "switch", row_pos, {"choice": "unclear"}) is False
    assert U.correctness("findver", "switch", row_neg, {"choice": None}) is False
    assert U.correctness("findver", "noul", {"gold": True}, {"noul": 0.5}) is True
    assert U.correctness("findver", "noul", {"gold": False}, {"noul": 0.4999}) is True
    assert U.correctness("findver", "noul", {"gold": True}, {"noul": "bad"}) is False


def test_noul_rejects_nonfinite_and_out_of_range():
    # NaN/Inf/OOR model outputs must fail closed (False), never silently label.
    assert U.correctness("findver", "noul", {"gold": True}, {"noul": float("nan")}) is False
    assert U.correctness("findver", "noul", {"gold": False}, {"noul": float("nan")}) is False
    assert U.correctness("findver", "noul", {"gold": True}, {"noul": float("inf")}) is False
    assert U.correctness("findver", "noul", {"gold": True}, {"noul": 1.5}) is False
    assert U.correctness("findver", "noul", {"gold": False}, {"noul": -0.1}) is False
    assert U.correctness("codexglue_defect", "noul", {"gold": True}, {"noul": float("nan")}) is False
    # boundaries still work
    assert U.correctness("findver", "noul", {"gold": True}, {"noul": 1.0}) is True
    assert U.correctness("findver", "noul", {"gold": False}, {"noul": 0.0}) is True
