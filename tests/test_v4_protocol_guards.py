import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_schema_ablation as m


def _search_manifests():
    return [ROOT / "results/v2/data_manifest/codexglue_function_groups.json",
            ROOT / "results/v2/data_manifest/findver_report_groups.json",
            ROOT / "results/v2/data_manifest/processbench_problem_groups.json",
            ROOT / "results/findver/controller_split_manifest.json"]


def test_search_manifests_never_reference_official_test_files():
    # v2 processbench search pools must not contain any official-test problem group
    from group_oof import problem_group_id
    split = json.loads((ROOT / "results/processbench/split_manifest.json").read_text(encoding="utf-8"))
    test_ids = {r["id"] for r in split["rows"]["test"]}
    test_groups = set()
    for fn in ("gsm8k.json", "math.json", "olympiadbench.json", "omnimath.json"):
        for r in json.loads((ROOT / "data/raw/processbench" / fn).read_text(encoding="utf-8")):
            if r["id"] in test_ids:
                test_groups.add(problem_group_id(r["problem"]))
    manifest = json.loads((ROOT / "results/v2/data_manifest/processbench_problem_groups.json").read_text(encoding="utf-8"))
    for key in ("search_train", "search_dev", "confirm"):
        assert set(manifest["splits"][key]).isdisjoint(test_groups), key
    assert manifest["n_history_touched_groups"] > 0


def test_search_rows_exclude_official_test_ids():
    # codexglue v2 search pools must reference official-TRAIN members only
    manifest = json.loads((ROOT / "results/v2/data_manifest/codexglue_function_groups.json").read_text(encoding="utf-8"))
    assert "members" in manifest, "manifest must pin per-group train-only member idx"
    raw = ROOT / "data/raw/codexglue_defect"
    train_idx = {int(x) for x in (raw / "train.txt").read_text(encoding="utf-8").split()}
    leaked = set(manifest.get("excluded_cross_split_leaked_idx", []))
    assert len(leaked) > 0, "expected known cross-split duplicates to be listed"
    for key in ("search_train", "search_dev", "confirm"):
        for g in manifest["splits"][key]:
            for i in manifest["members"].get(g, []):
                assert i in train_idx, f"non-train idx {i} in code search pool"
                assert i not in leaked, f"leaked idx {i} in code search pool"


def test_findver_search_pool_disjoint_from_official_test():
    manifest = json.loads((ROOT / "results/v2/data_manifest/findver_report_groups.json").read_text(encoding="utf-8"))
    test_reports = {r["report"] for r in json.loads(
        (ROOT / "external/FinDVer/data/test.json").read_text(encoding="utf-8"))}
    for key in ("search_train", "search_dev", "confirm"):
        assert set(manifest["splits"][key]).isdisjoint(test_reports)


def test_choice_normalization_is_canonical():
    assert m.normalize_choice("not_supported", {"supported", "not supported"}) == ("not supported", "ok")
    assert m.normalize_choice("Not Supported", {"supported", "not supported"}) == ("not supported", "ok")
    assert m.normalize_choice("SECURE", {"secure", "vulnerable"}) == ("secure", "ok")
