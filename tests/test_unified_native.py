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
    assert U.correctness("banking77_intent", "choice",
                         {"label": "pay_bill", "gold": None, "meta": {}},
                         {"choice": "pay bill"})
    assert not U.correctness("banking77_intent", "choice",
                             {"label": "pay_bill", "gold": None, "meta": {}},
                             {"choice": "cancel"})
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
