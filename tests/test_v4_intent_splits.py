import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
V4 = ROOT / "results/v4/data_manifest"
for f in ("clinc150_groups.json", "banking77_groups.json"):
    if not (V4 / f).exists():
        pytest.skip(f"missing {f}", allow_module_level=True)


def _load(name):
    return json.loads((V4 / name).read_text(encoding="utf-8"))


def test_intent_splits_disjoint_and_cover_search_pool():
    for name in ("clinc150_groups.json", "banking77_groups.json"):
        m = _load(name)
        sets = [set(m["splits"][k]) for k in ("search_train", "search_dev", "confirm")]
        assert sets[0].isdisjoint(sets[1]) and sets[0].isdisjoint(sets[2]) and sets[1].isdisjoint(sets[2])
        assert sum(len(s) for s in sets) == sum(m["split_sizes_groups"].values())


def test_intent_leakage_audit_present():
    m = _load("clinc150_groups.json")
    assert m["n_cross_split_groups"] >= 0 and m["n_exact_duplicate_groups"] >= 0
    assert m["oos_rows_total"] > 0, "clinc150 must report OOS rows separately"
    m = _load("banking77_groups.json")
    assert m["oos_rows_total"] == 0
    assert "revision" in m and "repo" in m
