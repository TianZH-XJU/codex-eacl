import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_group_faultmap as GF  # noqa: E402
import analyze_search_landscape as SL  # noqa: E402


def test_best_so_far_plateau():
    log = [{"best_scalar": 0.5}, {"best_scalar": 0.7}, {"best_scalar": 0.7}, {"best_scalar": 0.6}]
    out = SL.best_so_far(log)
    assert out["series"] == [0.5, 0.7, 0.7, 0.7]
    assert out["plateau_gen"] == 1 and out["first"] == 0.5 and out["last"] == 0.6
    out1 = SL.best_so_far([{"best_scalar": 0.3}])
    assert out1["plateau_gen"] == 0 and out1["series"] == [0.3]
    import pytest
    with pytest.raises(ValueError):
        SL.best_so_far([])


def test_group_severity_ci_degenerate_and_empty():
    rows = [{"group_id": "g", "gold": True, "noul_p": 0.9, "switch_correct": True}]
    c = GF.group_severity_ci(rows, n=100)
    assert c["est"] == 0.0 and c["ci95"] == [0.0, 0.0] and c["n_groups"] == 1
    c0 = GF.group_severity_ci([], n=100)
    assert c0 == {"est": 0.0, "ci95": [0.0, 0.0], "n_groups": 0, "group_sizes": []}


def test_committed_landscape_unavailable_marked():
    d = json.loads((ROOT / "results/v4_1/evolution/search_landscape.json").read_text(encoding="utf-8"))
    assert d["experiment_id"] == "EXP-03"
    assert set(d["unavailable_from_logs"]) == {"parent_child_correlation", "neutral_mutation_rate", "train_dev_rank_correlation", "random_trajectories", "nsga2_per_generation_trajectories"}
    f = d["families"]["evo_train_code_vela_b48"]["per_seed"]["17"]["ga"]
    assert f["unique"] == 22 and f["generated"] == 43
    assert f["best_so_far"]["plateau_gen"] == 1


def test_committed_group_faultmap_math_cell():
    d = json.loads((ROOT / "results/v4_1/evolution/group_faultmap.json").read_text(encoding="utf-8"))
    assert d["experiment_id"] == "EXP-04"
    cells = {tuple(c["cell"]): c for c in d["cells"]}
    c = cells[("math", "vela", "mid")]
    assert c["n"] == 14 and c["discover"]["n_groups"] == 12
    assert c["discover"]["est"] == 0.9286
    assert c["discover"]["ci95"][0] >= 0.7  # lower bound stays high
