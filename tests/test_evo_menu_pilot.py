import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_evo_menu_pilot import all_genomes, build_menus, sample_genomes  # noqa: E402


def test_genome_space_is_24_and_deterministic():
    g = all_genomes()
    assert len(g) == 24
    assert g == all_genomes()
    assert {x["retriever"] for x in g} == {"tfidf-word", "tfidf-char"}
    assert {x["k"] for x in g} == {5, 10, 20}


def test_random_subset_is_reproducible_and_within_space():
    a = sample_genomes(12, 17)
    b = sample_genomes(12, 17)
    assert a == b and len(a) == 12
    full = all_genomes()
    assert all(x in full for x in a)


def test_build_menus_respects_k_and_recall_flag():
    train = [{"text": f"pay bill {i}", "label": "pay_bill", "key": f"t{i}", "group": "g"}
             for i in range(6)]
    train += [{"text": f"cancel flight {i}", "label": "cancel", "key": f"c{i}", "group": "h"}
              for i in range(6)]
    dev = [{"text": "pay bill now", "label": "pay_bill", "key": "d0", "group": "x"}]
    for k in (5, 10, 20):
        menus = build_menus({"retriever": "tfidf-word", "k": k, "norm": "raw", "order": "ranked"},
                            train, dev)
        assert len(menus[0]["menu"]) == min(k, 2)
        assert menus[0]["recall_hit"] == ("pay_bill" in menus[0]["menu"])
    m1 = build_menus({"retriever": "tfidf-word", "k": 2, "norm": "raw", "order": "ranked"}, train, dev)
    m2 = build_menus({"retriever": "tfidf-word", "k": 2, "norm": "raw", "order": "shuffled"}, train, dev)
    assert sorted(m1[0]["menu"]) == sorted(m2[0]["menu"])
