import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_policy_cache import clean_noul_p  # noqa: E402


def test_clean_noul_p_accepts_unit_interval():
    assert clean_noul_p(0.0) == (0.0, None)
    assert clean_noul_p(1.0) == (1.0, None)
    assert clean_noul_p(0.5) == (0.5, None)
    assert clean_noul_p("0.7") == (0.7, None)


def test_clean_noul_p_rejects_garbage_with_error():
    v, e = clean_noul_p(float("nan"))
    assert v is None and e is not None and e.startswith("parse_error:")
    v, e = clean_noul_p(float("inf"))
    assert v is None and e is not None
    v, e = clean_noul_p(1.5)
    assert v is None and e is not None
    v, e = clean_noul_p(-0.1)
    assert v is None and e is not None
    v, e = clean_noul_p(None)
    assert v is None and e is not None
    v, e = clean_noul_p("bad")
    assert v is None and e is not None
