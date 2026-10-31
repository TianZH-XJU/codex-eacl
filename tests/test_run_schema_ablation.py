import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_schema_ablation as m


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
