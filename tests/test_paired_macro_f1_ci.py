import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import paired_macro_f1_ci as MF  # noqa: E402
from evolution import CascadeGenome, evaluate  # noqa: E402


def _rows():
    return json.loads((ROOT / "results/v4/evolution/cache_train_code_laya.json").read_text(encoding="utf-8"))["rows"]


def test_macro_f1_matches_fitness_evaluate():
    g = CascadeGenome(lo=0.0, hi=1.0, use_second=1)
    rows = _rows()
    assert abs(MF.macro_f1(MF.correctness(g, rows)) - evaluate(g, rows)["macro_f1"]) < 5e-5


def test_paired_join_guards():
    g = CascadeGenome(lo=0.0, hi=1.0, use_second=1)
    rows = _rows()
    a = MF.correctness(g, rows)
    with pytest.raises(SystemExit):
        MF.paired_macro_f1_ci(a[:-1], a, n=100)
    b = [list(t) for t in a]
    b[0] = (b[0][0], b[0][1], "other_key", b[0][3])
    with pytest.raises(SystemExit):
        MF.paired_macro_f1_ci(a, [tuple(t) for t in b], n=100)
    c = [list(t) for t in a]
    c[0] = (c[0][0], c[0][1], c[0][2], not c[0][3])
    with pytest.raises(SystemExit):
        MF.paired_macro_f1_ci(a, [tuple(t) for t in c], n=100)


def test_committed_macro_f1_cis():
    d = json.loads((ROOT / "results/v4_1/evolution/macro_f1_deploy_cis.json").read_text(encoding="utf-8"))
    assert d["method"].startswith("paired group-bootstrap")
    # in-sample train exception: findver/Vela GA-best beats switch-only (selection bias demo)
    for s in ("17", "23", "41"):
        x = d["tasks"]["findver_vela"]["per_seed"][s]["train"]["ga_minus_switchonly"]
        assert x["est"] == 0.1202 and x["excludes_zero"] is True
    # ...which vanishes out-of-pool: dev and confirm are ns
    for s in ("17", "23", "41"):
        for tag in ("dev", "confirm"):
            x = d["tasks"]["findver_vela"]["per_seed"][s][tag]["ga_minus_switchonly"]
            assert x["excludes_zero"] is False
    # code/Laya dev SIG-negative under accuracy is ns under macro-F1
    for s in ("17", "23"):
        x = d["tasks"]["code_laya"]["per_seed"][s]["dev"]["ga_minus_switchonly"]
        assert x["excludes_zero"] is False
    # GA-vs-random: all ns under macro-F1 (matches accuracy verdict)
    n_sig = 0
    for k, v in d["tasks"].items():
        for s, ps in v["per_seed"].items():
            for tag in ("train", "dev", "confirm"):
                if ps[tag]["ga_minus_random"]["excludes_zero"]:
                    n_sig += 1
    assert n_sig == 0
