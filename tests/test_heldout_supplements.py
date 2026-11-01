import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_complementarity_heldout as CH  # noqa: E402


def _rows():
    return json.loads((ROOT / "results/v4/evolution/cache_train_code_laya.json").read_text(encoding="utf-8"))["rows"]


def test_cell_stats_matches_train_screen():
    rows = _rows()
    st = CH.cell_stats(rows, "noul")
    # matches the committed EXP-01 screen: repair 14, harmful 15, net -1
    assert st["repair"] == 14 and st["harmful"] == 15 and st["net"] == -1
    assert st["net_ci95"] == [-0.2, 0.1667]
    # switch-baseline mirrors: repair/harmful swap
    st2 = CH.cell_stats(rows, "switch")
    assert st2["repair"] == 15 and st2["harmful"] == 14 and st2["net"] == 1


def test_macro_f1_matches_fitness_evaluate():
    from evolution import CascadeGenome, evaluate
    rows = _rows()
    g = CascadeGenome(lo=0.0, hi=1.0, use_second=1)
    assert abs(CH.macro_f1(rows, True) - evaluate(g, rows)["macro_f1"]) < 5e-5


def test_committed_heldout_complementarity():
    d = json.loads((ROOT / "results/v4_1/evolution/complementarity_heldout.json").read_text(encoding="utf-8"))
    assert d["design"].startswith("frozen train-selected")
    assert set(d["cells"]) == {"code_laya", "code_vela", "findver_laya", "findver_vela"}
    for k, c in d["cells"].items():
        assert c["base_action_frozen"] in ("noul", "switch")
        # direction frozen from train: dev/confirm use the same base
        assert set(c["splits"]) == {"train", "dev", "confirm"}
    # code_laya dev net is the only significantly positive held-out cell
    c = d["cells"]["code_laya"]["dev"]
    assert c["net"] == 12 and c["net_ci95"][0] > 0
    # confirm: no cell excludes zero except none (code_laya conf ns)
    # confirm: the only cell whose cluster-bootstrap CI excludes zero is
    # findver_laya (net +6). The exact McNemar p is 0.0703 (ns), so the
    # percentile cluster bootstrap is anti-conservative at G=16 groups — this
    # is reported as MARGINAL, not as a confirmed positive signal.
    sig = sorted(k for k, v in d["cells"].items()
                 if v["confirm"]["net_ci95"][0] > 0 or v["confirm"]["net_ci95"][1] < 0)
    assert sig == ["findver_laya"], sig
    c = d["cells"]["findver_laya"]["confirm"]
    assert c["net"] == 6 and c["net_ci95"][0] > 0
    assert c["mcnemar_exact_p"] == 0.0703  # exact test does not exclude zero
    # primary metric: fixed-action macro-F1 delta on the same pool includes zero
    assert c["macro_f1_delta_switch_minus_noul"] == 0.1113


def test_committed_routing_net_cis():
    d = json.loads((ROOT / "results/v4_1/evolution/routing_net_cis.json").read_text(encoding="utf-8"))
    assert set(d["cells"]) == {"code_laya", "code_vela", "findver_laya", "findver_vela"}
    for k, v in d["cells"].items():
        assert v["champion"] in ("logreg", "tree", "threshold")
        # router confirm CIs must not exclude zero (no confirmed positive utility)
        ci = v["confirm"]["router"]["net_ci95"]
        assert not (ci[0] > 0), (k, ci)
