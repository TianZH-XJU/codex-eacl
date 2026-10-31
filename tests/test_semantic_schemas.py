import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from semantic_schemas import SCHEMAS, schema_hash, validate_noul_schema


def test_semantic_noul_schema_validates():
    for name in ("code_noul", "findver_noul"):
        ok, why = validate_noul_schema(SCHEMAS[name])
        assert ok, (name, why)


def test_legacy_criterialess_noul_fails_validation():
    assert validate_noul_schema({"type": "noul", "instructions": "Is it?"})[0] is False
    assert validate_noul_schema({"type": "noul", "instructions": "x",
                                 "criteria": {"false": "a"}})[0] is False
    assert validate_noul_schema({"type": "noul", "instructions": "x",
                                 "criteria": {"false": "same", "true": "same"}})[0] is False
    assert validate_noul_schema({"type": "choice"})[0] is False


def test_schema_hashes_stable_and_distinct():
    hashes = {schema_hash(n) for n in SCHEMAS}
    assert len(hashes) == len(SCHEMAS)
    assert all(len(h) == 12 for h in hashes)
