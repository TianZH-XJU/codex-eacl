import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_faultmap as FM  # noqa: E402


def _bin_rows():
    return [
        {"gold": True, "noul_p": 0.9, "switch_correct": True},
        {"gold": True, "noul_p": 0.05, "switch_correct": False},
        {"gold": False, "noul_p": 0.95, "switch_correct": False},
        {"gold": False, "noul_p": 0.1, "switch_correct": True},
        {"gold": True, "noul_p": 0.45, "switch_correct": False},
        {"gold": False, "noul_p": 0.55, "switch_correct": True},
        {"gold": True, "noul_p": 0.42, "switch_correct": True},
        {"gold": False, "noul_p": 0.58, "switch_correct": False},
    ]


def _math_rows():
    return [{"correct": i % 3 != 0, "top1": 0.9 if i % 3 else 0.2,
             "margin": 0.5 if i % 3 else 0.01, "group_id": f"g{i % 3}", "gold": i,
             "key": f"m{i}", "pred": "step 1" if i % 3 else "step 0",
             "gold_render": "step 1", "n_cands": 4} for i in range(9)]


def test_conf_of_binary_bands():
    assert FM.conf_of_binary({"noul_p": 0.5}) == "low"
    assert FM.conf_of_binary({"noul_p": 0.44}) == "low"
    assert FM.conf_of_binary({"noul_p": 0.3}) == "mid"
    assert FM.conf_of_binary({"noul_p": 0.0}) == "high"
    assert FM.conf_of_binary({"noul_p": 1.0}) == "high"
    assert FM.conf_of_binary({"noul_p": "bad"}) is None
    assert FM.conf_of_binary({}) is None


def test_conf_of_math_bands():
    assert FM.conf_of_math({"margin": 0.01}) == "low"
    assert FM.conf_of_math({"margin": 0.1}) == "mid"
    assert FM.conf_of_math({"margin": 0.9}) == "high"
    assert FM.conf_of_math({"margin": None}) is None
    assert FM.conf_of_math({}) is None


def test_severity_binary_fail_closed():
    # rows: 0 ok/ok; 1 wrong/wrong; 2 wrong/wrong; 3 ok/ok;
    #       4 wrong/wrong; 5 wrong/ok; 6 wrong/ok; 7 wrong/wrong
    rows = _bin_rows()
    s, n, det = FM.severity(rows)
    assert n == 8
    assert det["both_wrong"] == 4
    assert det["only_noul"] == 0 and det["only_switch"] == 2 and det["both_ok"] == 2
    assert s == 4 / 8


def test_severity_math_and_min_n_gate():
    rows = _math_rows()
    s, n, det = FM.severity(rows)
    assert n == 9 and abs(s - 1 / 3) < 1e-9
    assert det["acc"] == round(2 / 3, 4)
    s2, n2, _ = FM.severity(rows[:2])
    assert (s2, n2) == (0.5, 2)
    # empty input is safe
    assert FM.severity([]) == (0.0, 0, {})


def test_wilson_lb_bounds():
    assert FM.wilson_lb(0, 0) == 0.0
    lb = FM.wilson_lb(5, 10)
    assert 0.0 <= lb <= 0.5
    assert FM.wilson_lb(10, 10) > 0.7


def test_end_to_end_on_synthetic_caches(tmp_path):
    disc = tmp_path / "cache_train_synth.json"
    disc.write_text(json.dumps({"task": "codexglue_defect", "model": "laya",
                                "split": "search_train",
                                "rows": _bin_rows()}), encoding="utf-8")
    dev = tmp_path / "cache_synth_dev.json"
    dev.write_text(json.dumps({"task": "codexglue_defect", "model": "laya",
                               "split": "search_dev",
                               "rows": _bin_rows()}), encoding="utf-8")
    conf = tmp_path / "cache_confirm_synth.json"
    conf.write_text(json.dumps({"task": "codexglue_defect", "model": "laya",
                                "split": "confirm",
                                "rows": _bin_rows()}), encoding="utf-8")
    out = tmp_path / "out.json"
    import sys
    old = sys.argv
    sys.argv = ["run_faultmap.py", "--train-glob", str(tmp_path / "cache_train_*.json"),
                "--dev-glob", str(tmp_path / "cache_*.json"),
                "--confirm-glob", str(tmp_path / "cache_confirm_*.json"),
                "--min-n", "2", "--out", str(out)]
    try:
        FM.main()
    finally:
        sys.argv = old
    d = json.loads(out.read_text(encoding="utf-8"))
    assert d["status"] == "passed"
    assert d["n_cells_eligible"] >= 1
    assert all(c["n_discover"] >= 2 for c in d["cells"])
    # identical discover/validate pools: every eligible cell must confirm
    assert all(c["confirmed_dev"] is True for c in d["cells"])
    assert all(c["confirmed_confirm"] is True for c in d["cells"])


def test_is_binary_routing():
    assert FM.is_binary({"noul_p": 0.5}) is True
    assert FM.is_binary({"top1": 0.9}) is False


def _write_cache(path, task, model, split, rows):
    path.write_text(json.dumps({"task": task, "model": model,
                                "split": split, "rows": rows}),
                    encoding="utf-8")


def test_select_files_excludes_train_confirm_and_test_from_dev(tmp_path):
    _write_cache(tmp_path / "cache_train_a.json", "codexglue_defect", "laya",
                 "search_train", _bin_rows())
    _write_cache(tmp_path / "cache_a.json", "codexglue_defect", "laya",
                 "search_dev", _bin_rows())
    _write_cache(tmp_path / "cache_confirm_a.json", "codexglue_defect", "laya",
                 "confirm", _bin_rows())
    _write_cache(tmp_path / "cache_test_a.json", "codexglue_defect", "laya",
                 "test", _bin_rows())
    tr, dev, conf = FM.select_files(str(tmp_path / "cache_train_*.json"),
                                    str(tmp_path / "cache_*.json"),
                                    str(tmp_path / "cache_confirm_*.json"))
    assert [Path(f).name for f in tr] == ["cache_train_a.json"]
    assert [Path(f).name for f in dev] == ["cache_a.json"]
    assert [Path(f).name for f in conf] == ["cache_confirm_a.json"]


def test_select_files_excludes_test_from_confirm(tmp_path):
    _write_cache(tmp_path / "cache_confirm_a.json", "codexglue_defect", "laya",
                 "confirm", _bin_rows())
    _write_cache(tmp_path / "cache_test_b.json", "codexglue_defect", "laya",
                 "test", _bin_rows())
    tr, dev, conf = FM.select_files(str(tmp_path / "NOMATCH_train*.json"),
                                    str(tmp_path / "NOMATCH*.json"),
                                    str(tmp_path / "cache_*.json"))
    assert all("cache_test_" not in f.replace("\\", "/") for f in conf)
    assert any(f.endswith("cache_confirm_a.json") for f in conf)


def test_bucketize_missing_conf_band_counts_in_denominator():
    # Same-task missing rows share schema (noul_p key present, value None);
    # severity routes on rows[0], consistent within a (task, model) cell.
    rows = [{"gold": True, "noul_p": None, "switch_correct": False},
            {"gold": True, "noul_p": None, "switch_correct": False},
            {"gold": True, "noul_p": 0.9, "switch_correct": True}]
    rooms = {("code", "laya", "missing"): rows}
    s, n, det = FM.severity(rooms[("code", "laya", "missing")])
    assert n == 3
    # both None rows are both-wrong (fail-closed); clean row is both-ok
    assert det["both_wrong"] == 2 and det["both_ok"] == 1
    assert s == 2 / 3


def test_bucketize_routes_none_conf_to_missing_band(tmp_path):
    rows = [{"gold": True, "noul_p": None, "switch_correct": False,
             "group_id": "g1", "key": "k1"},
            {"gold": False, "noul_p": 0.9, "switch_correct": True,
             "group_id": "g1", "key": "k2"}]
    f = tmp_path / "cache_train_m.json"
    _write_cache(f, "codexglue_defect", "laya", "search_train", rows)
    rooms = FM.bucketize([str(f)])
    assert ("code", "laya", "missing") in rooms
    assert len(rooms[("code", "laya", "missing")]) == 1
    s, n, det = FM.severity(rooms[("code", "laya", "missing")])
    assert (n, det["both_wrong"]) == (1, 1)
