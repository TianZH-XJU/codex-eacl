import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import complementarity_screen as CS  # noqa: E402


def test_screen_cell_code_vela_shape():
    # repair 10 / regression 4 / disagreement 14 / n 60, rest agree
    noul = [False] * 10 + [True] * 4 + [True] * 20 + [False] * 26
    sw = [True] * 10 + [False] * 4 + [True] * 20 + [False] * 26
    c = CS.screen_cell(noul, sw)
    assert c == {"n": 60, "repair": 10, "regression": 4, "net": 6,
                 "net_pp": 10.0, "disagreement": round(14 / 60, 4),
                 "oracle_acc": round(34 / 60, 4), "binomial_p": 0.0898}


def test_screen_cell_empty_and_degenerate():
    c = CS.screen_cell([], [])
    assert c["n"] == 0 and c["binomial_p"] == 1.0
    c = CS.screen_cell([True, False], [True, False])  # full agreement
    assert c["repair"] == 0 and c["regression"] == 0 and c["binomial_p"] == 1.0


def test_binom_upper_tail_exact():
    # P(X>=10 | n=14, p=0.5) = 1471/16384
    assert abs(CS.binom_upper_tail(10, 14) - 1471 / 16384) < 1e-12
    assert CS.binom_upper_tail(0, 5) == 1.0


def test_committed_screen_artifact():
    d = json.loads((ROOT / "results/v4_1/evolution/complementarity_screen.json").read_text(encoding="utf-8"))
    assert d["verdict"] == "KEEP_BLOCKED"
    assert len(d["cells"]) == 6
    assert d["pools"].startswith("search_train")
    assert d["git_sha"] and d["run_manifest"]
    best = max((c for c in d["cells"] if "net_pp" in c), key=lambda c: c["net_pp"])
    assert best["net"] == 6 and best["binomial_p"] == 0.0898
