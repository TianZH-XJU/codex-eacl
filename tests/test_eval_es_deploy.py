import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import eval_es_deploy as ED  # noqa: E402


def test_smoke_frozen_eval_both_tasks(tmp_path):
    out = tmp_path / "es_deploy.json"
    old = sys.argv
    sys.argv = ["x", "--out", str(out)]
    try:
        ED.main()
    finally:
        sys.argv = old
    d = json.loads(out.read_text(encoding="utf-8"))
    assert set(d["tasks"]) == {"code_laya", "code_vela"}
    for k, v in d["tasks"].items():
        for s in ("17", "23", "41"):
            ps = v["per_seed"][s]
            assert ps["splits"] == {"train": "search_train", "dev": "search_dev", "confirm": "confirm"}
            for tag in ("es_minus_switch_dev", "es_minus_switch_confirm"):
                assert ps[tag]["comparison"] == "es_minus_switchonly"
                assert "ci95" in ps[tag] and "est" in ps[tag]
            # genome is the frozen train-selected ES best, byte-identical
            src = json.loads((ROOT / f"results/v4_1/evolution/es_train_{k}.json").read_text(encoding="utf-8"))
            assert ps["es_genome"] == src["per_seed"][s]["es"]["best"]


def test_committed_es_deploy_matches_eval_convention():
    d = json.loads((ROOT / "results/v4_1/evolution/es_deploy_devconfirm.json").read_text(encoding="utf-8"))
    assert d["protocol"].startswith("v4.1 E-02")
    assert "paired_bootstrap" in d and d["paired_bootstrap"]["reps"] == 5000
    # code_vela deploy == switch-only behaviorally: all-zero diffs
    for s in ("17", "23", "41"):
        ps = d["tasks"]["code_vela"]["per_seed"][s]
        assert ps["es_minus_switch_dev"]["est"] == 0.0
        assert ps["es_minus_switch_confirm"]["est"] == 0.0
    # direction label is live, not vacuous: code_laya s17 dev is SIG negative
    assert d["tasks"]["code_laya"]["per_seed"]["17"]["es_minus_switch_dev"]["est"] < 0
    assert d["tasks"]["code_laya"]["per_seed"]["17"]["es_minus_switch_dev"]["excludes_zero"] is True
