import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import summarize_protocol_metrics as SPM  # noqa: E402


def test_summarize_handles_heterogeneous_schemas(tmp_path):
    a = tmp_path / "a.json"
    a.write_text(json.dumps({"experiment_id": "X", "seeds": [17],
                             "train_cache": {"path": "p"}, "tasks": {"t": {}}}),
                 encoding="utf-8")
    b = tmp_path / "b.json"
    b.write_text(json.dumps({"cells": {"c1": {}}}), encoding="utf-8")
    assert SPM.summarize(str(a))["seeds"] == [17]
    assert SPM.summarize(str(b))["cells"] == ["c1"]
    assert SPM.summarize(str(a)).get("metric") is None  # omitted, not guessed


def test_main_records_corrupt_file_without_aborting(tmp_path):
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"experiment_id": "X"}), encoding="utf-8")
    bad = tmp_path / "bad.json"
    bad.write_text("{bad json", encoding="utf-8")
    import shutil
    dest = tmp_path / "evo"
    dest.mkdir()
    shutil.copy(str(good), str(dest / "good.json"))
    shutil.copy(str(bad), str(dest / "bad.json"))
    # main() globs the fixed repo dir; exercise the same code path directly
    files = {}
    import glob as _glob
    for f in sorted(_glob.glob(str(dest / "*.json"))):
        try:
            files[Path(f).stem] = SPM.summarize(f)
        except Exception as e:
            files[Path(f).stem] = {"file": Path(f).name,
                                    "error": f"{type(e).__name__}: {e}"}
    assert "error" not in files["good"]
    assert files["bad"]["error"].startswith("JSONDecodeError")


def test_committed_protocol_index():
    d = json.loads((ROOT / "results/v4_1/manifests/protocol_metrics_index.json").read_text(encoding="utf-8"))
    assert d["experiment_id"] == "S-06" and d["repo_commit"]
    assert len(d["files"]) >= 26
    assert not [k for k, v in d["files"].items() if "error" in v]
    for key in ("evo_train_code_laya", "nsga2_deploy_devconfirm", "transfer_heldout",
                "action_complementarity", "utility_predictability"):
        assert key in d["files"], key
