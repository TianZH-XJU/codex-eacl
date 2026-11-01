import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tasks.code import CodeAdapter  # noqa: E402
from tasks.findver import FindverAdapter  # noqa: E402
from tasks.intent import IntentAdapter  # noqa: E402
from tasks.math import ProcessbenchAdapter  # noqa: E402


def test_group_units_are_honest():
    assert CodeAdapter.group_unit == "function_text_hash"
    assert "project" not in CodeAdapter.group_unit and "holdout" not in CodeAdapter.group_unit
    assert FindverAdapter.group_unit == "report_id"
    assert ProcessbenchAdapter.group_unit == "problem_text"
    assert IntentAdapter("banking77").group_unit == "normalized_utterance"


def test_v4_1_evo_files_carry_train_cache_paths():
    for name in ("evo_train_code_laya", "evo_train_code_vela",
                 "evo_train_findver_laya", "evo_train_findver_vela"):
        p = ROOT / "results/v4_1/evolution" / f"{name}.json"
        if not p.exists():
            pytest.skip(f"not generated yet: {name}")
        d = json.loads(p.read_text(encoding="utf-8"))
        assert "train_cache" in d and Path(ROOT / d["train_cache"]["path"]).exists()
        assert d.get("run_manifest", {}).get("metric") == "macro_f1"


def test_processbench_mapping_matches_official_adapter():
    sys.path.insert(0, str(ROOT / "scripts"))
    from prepare_processbench import to_choice_task
    a = ProcessbenchAdapter()
    rows = a.load_rows("search_dev", 8, 17)
    assert len(rows) == 8
    by_id = {}
    for fn in ("gsm8k.json", "math.json", "olympiadbench.json", "omnimath.json"):
        for r in json.loads((ROOT / "data/raw/processbench" / fn).read_text(encoding="utf-8")):
            by_id[r["id"]] = r
    for r in rows:
        raw = by_id[r["key"]]
        task = to_choice_task(raw)
        assert r["gold"] == task["gold_index"], r["key"]
        assert a.candidates(r)[-1] == "ALL_STEPS_CORRECT"
        assert (r["gold"] == len(raw["steps"])) == (raw["label"] == -1)


def test_significance_accepts_v4_1_train_cache_files(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import test_evo_significance as S
    out = tmp_path / "sig.json"
    sys.argv = ["x", "--evo",
                str(ROOT / "results/v4_1/evolution/evo_train_code_laya.json"),
                "--out", str(out)]
    S.main()
    d = json.loads(out.read_text(encoding="utf-8"))
    assert "evo_train_code_laya" in d
    assert set(d["evo_train_code_laya"]) == {"17", "23", "41"}


def test_findver_normalization_and_report_grouping():
    a = FindverAdapter()
    assert a.labels() == ["not_supported", "supported"]
    rows = a.load_rows("search_dev", 20, 17)
    assert len(rows) == 20
    manifest = json.loads((ROOT / "results/v2/data_manifest/findver_report_groups.json").read_text(encoding="utf-8"))
    dev_reports = set(manifest["splits"]["search_dev"])
    assert {r["group_id"][3:] for r in rows} <= dev_reports
    assert all(isinstance(r["gold"], bool) for r in rows)
