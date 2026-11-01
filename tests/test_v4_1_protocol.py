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
