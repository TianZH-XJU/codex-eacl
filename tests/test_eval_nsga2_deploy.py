import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import eval_nsga2_deploy as D  # noqa: E402
from evolution import AbstainGenome, CascadeGenome  # noqa: E402


def test_select_binary_rule_and_tiebreaks():
    g1 = CascadeGenome(lo=0.1, hi=0.2, use_second=1)
    g2 = CascadeGenome(lo=0.3, hi=0.4, use_second=1)
    # accuracy dominates
    assert D.select_binary([(0.6, -2.0, 0.4, (1,), g1), (0.5, -1.0, 0.4, (2,), g2)]) is g1
    # accuracy tie -> fewer calls wins
    assert D.select_binary([(0.5, -2.0, 0.4, (1,), g1), (0.5, -1.0, 0.4, (2,), g2)]) is g2
    # full tie -> larger genome tuple wins (deterministic under reverse sort)
    assert D.select_binary([(0.5, -1.0, 0.4, (2,), g2), (0.5, -1.0, 0.4, (1,), g1)]) is g2


def test_select_abstain_rule_and_tiebreaks():
    a1 = AbstainGenome(t=0.1, use_margin=0, m=0.0)
    a2 = AbstainGenome(t=0.2, use_margin=0, m=0.0)
    assert D.select_abstain([(0.4, 0.1, (1,), a1), (0.3, 0.9, (2,), a2)]) is a1
    assert D.select_abstain([(0.3, 0.5, (1,), a1), (0.3, 0.8, (2,), a2)]) is a2


def test_smoke_one_task_reports_frozen_selection(tmp_path):
    out = tmp_path / "deploy.json"
    old = sys.argv
    sys.argv = ["x", "--tasks", "code_vela", "--out", str(out)]
    try:
        D.main()
    finally:
        sys.argv = old
    d = json.loads(out.read_text(encoding="utf-8"))
    assert set(d["tasks"]) == {"code_vela"}
    assert "deploy_rule" in d and "cascade" in d["deploy_rule"]
    for s in ("17", "23", "41"):
        ps = d["tasks"]["code_vela"]["per_seed"][s]
        assert ps["n_candidates"] >= 1
        assert ps["deploy_genome"] in ps["candidate_genomes"]
        assert ps["splits"] == {"train": "search_train", "dev": "search_dev", "confirm": "confirm"}
        for k in ("deploy_minus_base_dev", "deploy_minus_base_confirm"):
            assert ps[k]["comparison"] == "deploy_minus_switchonly"
            assert "ci95" in ps[k] and "est" in ps[k]
    # frozen candidates match the committed train pareto genomes, in order
    ns = json.loads((ROOT / "results/v4_1/evolution/nsga2_train_code_vela.json").read_text(encoding="utf-8"))
    dev = json.loads((ROOT / "results/v4/evolution/cache_code_vela.json").read_text(encoding="utf-8"))["rows"]
    from evolution import evaluate as cascade_evaluate  # noqa: E402
    for s in ("17", "23", "41"):
        ps = d["tasks"]["code_vela"]["per_seed"][s]
        train_pareto = ns["per_seed"][s]["pareto"]
        assert ps["candidate_genomes"] == [e["genome"] for e in train_pareto]
        assert ps["candidate_train_objectives"] == [e["objectives"] for e in train_pareto]
        # deploy must be a dev-accuracy winner (tie-breaks may vary, max may not)
        scored = []
        for gvec in ps["candidate_genomes"]:
            gg = CascadeGenome(lo=gvec[0], hi=gvec[1], use_second=int(round(gvec[2])))
            f = cascade_evaluate(gg, dev)
            scored.append((f["accuracy"], tuple(gvec)))
        best_acc = max(a for a, _ in scored)
        assert tuple(ps["deploy_genome"]) in {g for a, g in scored if a == best_acc}
