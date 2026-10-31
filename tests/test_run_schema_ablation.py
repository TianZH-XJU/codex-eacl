import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_schema_ablation as m


def test_predict_kwargs_accepts_in_range():
    assert m.parse_predict_kwargs("head_max_len=96", "laya") == {"head_max_len": 96}
    assert m.parse_predict_kwargs("head_max_len=96,max_len=384", "laya") == {"head_max_len": 96, "max_len": 384}
    assert m.parse_predict_kwargs('{"max_len": 384}', "laya") == {"max_len": 384}


def test_predict_kwargs_rejects_ood_unknown_nonint_and_vela():
    for bad in ("head_max_len=512", "max_len=1024", "foo=1", "head_max_len=1.5", "head_max_len"):
        with pytest.raises(SystemExit):
            m.parse_predict_kwargs(bad, "laya")
    with pytest.raises(SystemExit):
        m.parse_predict_kwargs("head_max_len=96", "vela")


def test_row_resolution_deterministic():
    a = m.resolve_rows("findver", 10, 17)
    b = m.resolve_rows("findver", 10, 17)
    assert [r["key"] for r in a] == [r["key"] for r in b]
    assert len(a) == 10
    assert all(r["group_id"].startswith("rp:") for r in a)


def test_code_row_resolution_labels():
    rows = m.resolve_rows("code_defect", 6, 17)
    assert len(rows) == 6
    assert {r["gold"] for r in rows} <= {True, False}
    assert all(r["group_id"].startswith("fn:") for r in rows)


def test_question_builder_shapes():
    q = m.build_question("code_defect", "legacy_noul")
    assert q["type"] == "noul" and "criteria" not in q
    q2 = m.build_question("code_defect", "semantic_noul")
    assert set(q2["criteria"]) == {"false", "true"}
    q3 = m.build_question("findver", "choice")
    assert q3["type"] == "choice" and len(q3["criteria"]) == 2


def _check_budget_kw(kw):
    allowed = {"head_max_len": (1, 192), "max_len": (1, 512)}
    for k, v in kw.items():
        if k not in allowed:
            return False
        lo, hi = allowed[k]
        if not isinstance(v, int) or isinstance(v, bool) or not (lo <= v <= hi):
            return False
    return True


def test_predict_kwargs_validation():
    import json as _json
    assert _check_budget_kw(_json.loads('{"head_max_len": 96}'))
    assert _check_budget_kw(_json.loads('{"max_len": 384}'))
    for bad in ('{"head_max_len": 512}', '{"max_len": 1024}', '{"foo": 1}',
                '{"head_max_len": 1.5}', '{"head_max_len": true}'):
        assert not _check_budget_kw(_json.loads(bad)), bad
