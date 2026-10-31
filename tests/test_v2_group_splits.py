import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "results" / "v2" / "data_manifest"
for p in (V2 / "codexglue_function_groups.json", V2 / "findver_report_groups.json",
          V2 / "processbench_problem_groups.json"):
    if not p.exists():
        pytest.skip(f"v2 manifests not generated yet: {p}", allow_module_level=True)


def test_v2_splits_have_zero_group_overlap():
    for name in ("codexglue_function_groups.json", "findver_report_groups.json",
                 "processbench_problem_groups.json"):
        m = json.loads((V2 / name).read_text(encoding="utf-8"))
        sets = [set(m["splits"][k]) for k in ("search_train", "search_dev", "confirm")]
        assert sets[0].isdisjoint(sets[1]) and sets[0].isdisjoint(sets[2]) and sets[1].isdisjoint(sets[2]), name
        assert sum(len(s) for s in sets) == sum(m["split_sizes_groups"].values()) if "split_sizes_groups" in m else True


def test_codexglue_excludes_cross_split_leaks():
    m = json.loads((V2 / "codexglue_function_groups.json").read_text(encoding="utf-8"))
    assert len(m["excluded_cross_split_leaked_idx"]) >= 1
    leaked = set(m["excluded_cross_split_leaked_idx"])
    for k in ("search_train", "search_dev", "confirm"):
        assert leaked.isdisjoint(set(m["splits"][k]))


def test_findver_pool_excludes_official_test_reports():
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "scripts"))
    test_reports = {r["report"] for r in json.loads(
        (ROOT / "external/FinDVer/data/test.json").read_text(encoding="utf-8"))}
    m = json.loads((V2 / "findver_report_groups.json").read_text(encoding="utf-8"))
    used = set()
    for k in ("search_train", "search_dev", "confirm"):
        used.update(m["splits"][k])
    assert used.isdisjoint(test_reports)
