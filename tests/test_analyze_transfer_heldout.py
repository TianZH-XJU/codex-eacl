import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_transfer_heldout as TH  # noqa: E402
from evolution import CascadeGenome  # noqa: E402


def test_frozen_genomes_come_from_train_files():
    d = json.loads((ROOT / "results/v4_1/evolution/transfer_heldout.json").read_text(encoding="utf-8"))
    assert d["experiment_id"] == "EXP-05"
    assert "no re-search, no confirm tuning" in d["pools"]
    assert len(d["matrix"]) == 16
    for k, v in d["matrix"].items():
        st = k.split("_on_")[0]
        for tag in ("dev", "confirm"):
            c = v[tag]
            assert c["pool"].endswith(".json") and c["source"]["evo_file"].endswith(f"evo_train_{st}.json")
            assert "cache_train_" in c["source"]["train_cache"]
            for s, x in c["vs_switch_paired"]["per_seed"].items():
                assert set(x) >= {"est", "ci95", "excludes_zero", "comparison"}
                assert x["comparison"] == "transfer_minus_switch"


def test_frozen_genome_recompute_matches_artifact():
    # True frozen-proof: reload the train-selected best from the evo file and
    # re-evaluate on the target dev pool; the mean must equal the artifact's.
    from evolution import evaluate as cascade_evaluate
    d = json.loads((ROOT / "results/v4_1/evolution/transfer_heldout.json").read_text(encoding="utf-8"))
    evo = json.loads((ROOT / "results/v4_1/evolution/evo_train_code_laya.json").read_text(encoding="utf-8"))
    dev = json.loads((ROOT / "results/v4/evolution/cache_code_vela.json").read_text(encoding="utf-8"))["rows"]
    fits = []
    for s in ("17", "23", "41"):
        g = CascadeGenome(**evo["per_seed"][s]["ga"]["best"])
        fits.append(cascade_evaluate(g, dev)["macro_f1"])
    expect = round(sum(fits) / len(fits), 4)
    assert d["matrix"]["code_laya_on_code_vela"]["dev"]["mean_macro_f1"] == expect


def test_heldout_verdict_no_positive_sig():
    d = json.loads((ROOT / "results/v4_1/evolution/transfer_heldout.json").read_text(encoding="utf-8"))
    pos = [(k, tag, s) for k, v in d["matrix"].items() for tag in ("dev", "confirm")
           for s, x in v[tag]["vs_switch_paired"]["per_seed"].items()
           if x["excludes_zero"] and x["est"] > 0]
    assert pos == []


def test_diagonal_confirm_matches_eval_file():
    # diagonal confirm transfer == train-selected GA evaluated on confirm:
    # est must match exactly; CIs use a module-global RNG consumed in call
    # order, so across scripts only overlap + same verdict is asserted.
    d = json.loads((ROOT / "results/v4_1/evolution/transfer_heldout.json").read_text(encoding="utf-8"))
    e = json.loads((ROOT / "results/v4_1/evolution/train_search_confirm_eval.json").read_text(encoding="utf-8"))
    for t in ("code_laya", "code_vela", "findver_laya", "findver_vela"):
        for s in ("17", "23", "41"):
            a = d["matrix"][f"{t}_on_{t}"]["confirm"]["vs_switch_paired"]["per_seed"][s]
            b = e["tasks"][t]["per_seed"][s]["ga_minus_switchonly_report"]
            assert a["est"] == b["est"], (t, s)
            assert a["excludes_zero"] == b["excludes_zero"], (t, s)
            lo, hi = max(a["ci95"][0], b["ci95"][0]), min(a["ci95"][1], b["ci95"][1])
            assert lo <= hi, (t, s, a["ci95"], b["ci95"])
