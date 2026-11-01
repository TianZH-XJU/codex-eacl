import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_policy_evolution as RPE  # noqa: E402


def _tiny_cache(path, n=40, seed=0, prefix="r"):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        gold = rng.random() < 0.6
        rows.append({"key": f"{prefix}{i}", "gold": gold, "group_id": f"g{i % 5}",
                     "noul_p": rng.random(),
                     "switch_correct": rng.random() < 0.6})
    path.write_text(json.dumps({"task": "codexglue_defect", "model": "laya",
                                "split": "search_train", "rows": rows}),
                    encoding="utf-8")


def test_ga_gens_effective_formula():
    assert RPE.ga_gens_effective(24, 8, 1, 3) == 2
    assert RPE.ga_gens_effective(48, 8, 1, 5) == 5
    assert RPE.ga_gens_effective(10, 8, 1, 5) == 0  # budget < pop: no full gen
    assert RPE.ga_gens_effective(24, 8, 1, 99) == 2  # capped by budget


def test_significance_tags_carry_algorithm(tmp_path):
    import test_evo_significance as S
    for stem, tag in (("evo_train_code_laya", "ga_minus_random"),
                      ("es_train_code_laya", "es_minus_random")):
        out = tmp_path / f"{stem}.json"
        old = sys.argv
        sys.argv = ["x", "--evo",
                    str(ROOT / f"results/v4_1/evolution/{stem}.json"),
                    "--out", str(out)]
        try:
            S.main()
        finally:
            sys.argv = old
        d = json.loads(out.read_text(encoding="utf-8"))
        assert all(v["comparison"] == tag for v in d[stem].values()), stem


def _run_main(tmp_path, algo, extra=()):
    tr = tmp_path / "cache_train_t.json"
    dv = tmp_path / "cache_dev_t.json"
    _tiny_cache(tr, n=40, seed=1, prefix="tr")
    _tiny_cache(dv, n=20, seed=2, prefix="dv")
    out = tmp_path / f"out_{algo}.json"
    argv = ["x", "--train-cache", str(tr), "--dev-cache", str(dv),
            "--algo", algo, "--budget", "10", "--seeds", "17",
            "--pop", "4", "--gens", "2", "--mut", "0.2", "--cross", "0.8",
            "--out", str(out), *extra]
    old = sys.argv
    sys.argv = argv
    try:
        RPE.main()
    finally:
        sys.argv = old
    return json.loads(out.read_text(encoding="utf-8"))


def test_entry_keys_follow_algorithm(tmp_path):
    d = _run_main(tmp_path, "es", ("--mu", "2", "--lams", "2"))
    e = d["per_seed"]["17"]
    assert set(e) == {"es", "random", "dev"}
    assert "ga" not in e and "ga" not in e["dev"]
    assert e["dev"]["es"]["accuracy"] is not None
    # ES generations come from the actual log, not the GA formula
    assert d["gens_rule"].startswith("es-actual")
    assert d["gens_effective"] == len(e["es"]["log"]) == e["es"]["generations_actual"]
    assert e["es"]["unique"] <= 10


def test_ga_entry_keys_unchanged(tmp_path):
    d = _run_main(tmp_path, "ga")
    e = d["per_seed"]["17"]
    assert set(e) == {"ga", "random", "dev"}
    assert d["gens_rule"].startswith("ga-budget")
    assert d["gens_effective"] == RPE.ga_gens_effective(10, 4, 1, 2) == 2


def test_overlapping_train_dev_keys_fail_closed(tmp_path):
    # Fail-closed pool hygiene: shared row keys between train and dev refuse.
    tr = tmp_path / "cache_train_t.json"
    dv = tmp_path / "cache_dev_t.json"
    _tiny_cache(tr, n=20, seed=1, prefix="shared")
    _tiny_cache(dv, n=20, seed=2, prefix="shared")  # same key space -> overlap
    out = tmp_path / "out_leak.json"
    old = sys.argv
    sys.argv = ["x", "--train-cache", str(tr), "--dev-cache", str(dv),
                "--algo", "ga", "--budget", "10", "--seeds", "17",
                "--pop", "4", "--gens", "2", "--out", str(out)]
    try:
        with pytest.raises(SystemExit, match="pool leak"):
            RPE.main()
    finally:
        sys.argv = old
