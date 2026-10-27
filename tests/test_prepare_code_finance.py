import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_codexglue_defect import validate_row as code_validate
from prepare_findver import validate_row as fin_validate


def _code_row(target=1, func="int f() { return 0; }", idx=7):
    return {"func": func, "target": target, "idx": idx,
            "project": "p", "commit_id": "c"}


def test_code_row_validation():
    assert code_validate(_code_row(), 7)[0] is True
    assert code_validate(_code_row(target=2), 7)[0] is False
    assert code_validate(_code_row(func="  "), 7)[0] is False
    assert code_validate(_code_row(), 8)[0] is False  # idx mismatch


def test_findver_row_validation(tmp_path):
    rep = {"context": [{"id": 0, "context": "a"}, {"id": 1, "context": "b"}], "html_tables": []}
    (tmp_path / "R.json").write_text(__import__("json").dumps(rep), encoding="utf-8")

    def row(**kw):
        r = {"example_id": "e", "subset": "numeric", "statement": "s",
             "explanation": [], "entailment_label": True,
             "relevant_context": [0, 1], "report": "R.json"}
        r.update(kw)
        return r

    assert fin_validate(row(), tmp_path)[0] is True
    assert fin_validate(row(subset="other"), tmp_path)[0] is False
    assert fin_validate(row(entailment_label=1), tmp_path)[0] is False
    assert fin_validate(row(relevant_context=[0, 5]), tmp_path)[0] is False
    assert fin_validate(row(report="missing.json"), tmp_path)[0] is False
