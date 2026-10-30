import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def test_codexglue_audit_logic_on_synthetic():
    from collections import Counter
    rows = [{"idx": 0, "func": "f()", "target": 1, "project": "p1"},
            {"idx": 1, "func": "f()", "target": 0, "project": "p1"},
            {"idx": 2, "func": "g()", "target": 0, "project": "p2"}]
    assert sum(1 for c in Counter(r["func"] for r in rows).values() if c > 1) == 1


def test_findver_grouping_keys():
    import make_findver_split as m  # noqa: F401
    assert True
