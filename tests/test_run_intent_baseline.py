import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_intent_baseline as m


def test_search_rows_deterministic_and_train_only():
    a = m.load_search_rows("banking77", 10, 17)
    b = m.load_search_rows("banking77", 10, 17)
    assert [r["key"] for r in a] == [r["key"] for r in b]
    assert len(a) == 10
    assert all(r["group_id"] for r in a)


def test_search_rows_come_from_manifest_search_train():
    import json
    rows = m.load_search_rows("banking77", 20, 17)
    manifest = json.loads((ROOT / "results/v2/data_manifest/banking77_groups.json").read_text(encoding="utf-8"))
    allowed = set()
    for g in manifest["splits"]["search_train"]:
        allowed.update(manifest["members"].get(g, []))
    # row keys are "train:<idx>"
    for r in rows:
        assert int(r["key"].split(":")[1]) in allowed
