import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_action_complementarity as AC  # noqa: E402


def _row(noul_ok, sw_ok, gid="g0", gold=True, key="k"):
    return {"gold": gold, "group_id": gid, "key": key,
            "noul_p": 0.9 if noul_ok else 0.1,
            "switch_correct": sw_ok}


def test_cell_stats_decomposition():
    # base = noul (3/5) vs switch (1/5); repair 1, harmful 3, neutral 1
    rows = [_row(True, False, "g1", True, "k1"),    # noul ok, switch wrong
            _row(False, True, "g1", True, "k2"),    # noul wrong, switch ok -> repairable
            _row(True, False, "g2", True, "k3"),
            _row(True, False, "g2", True, "k4"),    # harmful x3 (noul ok, switch wrong)
            _row(False, False, "g3", True, "k5")]   # both wrong -> neutral
    st = AC.cell_stats(rows)
    assert st["base_action"] == "noul" and st["n"] == 5
    assert (st["repairable"], st["harmful"], st["neutral"]) == (1, 3, 1)
    assert abs(st["oracle_acc"] - 4 / 5) < 1e-9
    assert abs(st["headroom"] - 1 / 5) < 1e-9
    assert st["disagreement"] == 0.8


def test_cell_stats_base_is_best_fixed():    # switch strictly better -> base flips, repair/harmful swap sides
    rows = [_row(False, True, "g1", True, "k1"),
            _row(False, True, "g1", True, "k2"),
            _row(True, True, "g2", True, "k3")]
    st = AC.cell_stats(rows)
    assert st["base_action"] == "switch"
    assert (st["repairable"], st["harmful"], st["neutral"]) == (0, 2, 1)
    assert st["headroom"] == 0.0


def test_cell_stats_tie_breaks_to_noul():
    # 1/2 vs 1/2 tie -> deterministic noul base
    rows = [_row(True, False, "g1", True, "k1"),
            _row(False, True, "g1", True, "k2")]
    st = AC.cell_stats(rows)
    assert st["base_action"] == "noul"
    assert (st["repairable"], st["harmful"], st["neutral"]) == (1, 1, 0)


def test_cell_stats_empty_raises():
    import pytest
    with pytest.raises(ValueError):
        AC.cell_stats([])


def test_committed_complementarity_artifact():
    d = json.loads((ROOT / "results/v4_1/evolution/action_complementarity.json").read_text(encoding="utf-8"))
    assert d["experiment_id"] == "EXP-01" and d["repo_commit"]
    assert set(d["cells"]) == {"code_laya", "code_vela", "findver_laya", "findver_vela", "math_laya", "math_vela"}
    c = d["cells"]["code_vela"]
    assert (c["repairable"], c["harmful"]) == (4, 10) and c["base_action"] == "switch"
    assert c["headroom_ci"]["est"] == c["headroom"]
    m = d["cells"]["math_vela"]
    assert m["headroom"] == 0.0 and "single representation" in m["note"]
    assert d["cells"]["code_laya"]["n_groups"] == 60
    assert d["cells"]["findver_laya"]["n_groups"] == 41
    assert "singleton" in d["ci_method"]
