import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_binary_cache as b


def test_task_configs_have_matching_labels():
    for task in ("code_defect", "findver"):
        cfg = b.TASKS[task]
        assert set(cfg["switch_q"]["criteria"].keys()) == {cfg["pos_label"]} | (
            {"secure"} if task == "code_defect" else {"not supported"})
        assert cfg["single_q"]["type"] == "noul"
        assert cfg["render"]("not supported") == "not_supported"


def test_load_rows_code_valid_indices(tmp_path):
    import json
    (tmp_path / "function.json").write_text(json.dumps([
        {"func": "f0", "target": 1, "project": "p", "commit_id": "c"},
        {"func": "f1", "target": 0, "project": "p", "commit_id": "c"}]), encoding="utf-8")
    (tmp_path / "valid.txt").write_text("1\n0\n", encoding="utf-8")
    rows = b.load_rows("code_defect", str(tmp_path), "unused", "valid", 2, 17)
    assert len(rows) == 2
    assert {r["gold_render"] for r in rows} == {"vulnerable", "secure"}
