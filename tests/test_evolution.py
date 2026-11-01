import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evolution import (BudgetExceeded, BudgetTracker, CascadeGenome, crossover,  # noqa: E402
                       decide, diversity, evaluate, ga_run, mutate, random_run,
                       sample_genome)


def synth(n=60, seed=0):
    import random
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        p = rng.random()
        gold = rng.random() < 0.7
        # switch correct iff p in middle band; noul follows p>=0.5 vs gold with noise
        switch_ok = 0.35 <= p <= 0.65
        rows.append({"gold": gold, "noul_p": p, "switch_correct": switch_ok})
    return rows


def test_repair_and_vector_roundtrip():
    g = CascadeGenome(lo=0.8, hi=0.2, use_second=2).repaired()
    assert (g.lo, g.hi) == (0.2, 0.8) and g.use_second == 1
    assert CascadeGenome.from_vector(g.to_vector()) == g
    g2 = CascadeGenome(lo=-1, hi=5, use_second=0).repaired()
    assert (g2.lo, g2.hi) == (0.0, 1.0)


def test_decide_and_evaluate_exact():
    rows = [{"gold": True, "noul_p": 0.9, "switch_correct": False},
            {"gold": False, "noul_p": 0.5, "switch_correct": True}]
    g = CascadeGenome(lo=0.4, hi=0.6, use_second=1)
    assert decide(g, 0.9, False, True) == (True, 1)
    assert decide(g, 0.5, True, False) == (True, 2)
    fit = evaluate(g, rows)
    assert fit["accuracy"] == 1.0 and fit["calls_per_row"] == 1.5
    g0 = CascadeGenome(lo=0.0, hi=1.0, use_second=0)
    assert evaluate(g0, rows)["calls_per_row"] == 1.0


def test_budget_enforced():
    t = BudgetTracker(max_unique_evals=3, max_model_calls=10)
    for i in range(3):
        t.charge_eval((i,))
    with pytest.raises(BudgetExceeded):
        t.charge_eval((9,))
    t2 = BudgetTracker(max_unique_evals=100, max_model_calls=2)
    t2.charge_eval((1,), model_calls=2)
    with pytest.raises(BudgetExceeded):
        t2.charge_eval((1,), model_calls=1)


def test_random_reproducible_and_ga_deterministic():
    rows = synth()
    r1 = random_run(rows, budget=12, seed=17)
    r2 = random_run(rows, budget=12, seed=17)
    assert r1 == r2 and r1["n_evals"] == 12
    g1 = ga_run(rows, population=6, generations=3, crossover_rate=0.8,
                mutation_rate=0.2, elitism=1, seed=17)
    g2 = ga_run(rows, population=6, generations=3, crossover_rate=0.8,
                mutation_rate=0.2, elitism=1, seed=17)
    assert g1["best"] == g2["best"] and len(g1["generations"]) == 3


def test_ga_beats_noul_only_on_structured_cache():
    rows = synth(n=200, seed=1)
    base = evaluate(CascadeGenome(lo=0.0, hi=1.0, use_second=0), rows)
    g = ga_run(rows, population=8, generations=4, crossover_rate=0.8,
               mutation_rate=0.2, elitism=1, seed=17)
    assert g["best_fit"]["accuracy"] > base["accuracy"]
    assert g["best"]["use_second"] == 1


def test_ga_stops_gracefully_when_budget_exhausted():
    from evolution import BudgetTracker
    rows = synth(n=30, seed=2)
    t = BudgetTracker(max_unique_evals=10, max_model_calls=0)
    g = ga_run(rows, population=8, generations=5, crossover_rate=0.8,
               mutation_rate=0.2, elitism=1, seed=5, tracker=t)
    assert g["stopped_early"] is True
    assert g["n_unique_evals"] <= 10
    assert g["best_fit"]["accuracy"] >= 0.0


def test_crossover_mutation_diversity():
    import random
    rng = random.Random(0)
    a = CascadeGenome(lo=0.1, hi=0.2, use_second=0)
    b = CascadeGenome(lo=0.9, hi=0.8, use_second=1)
    kids = {crossover(rng, a, b, 1.0).to_vector() for _ in range(20)}
    assert len(kids) > 1
    assert crossover(rng, a, b, 0.0) == a
    assert diversity([a, a]) == 0.0 and diversity([a, b]) > 0.0
    m = mutate(random.Random(0), a, 1.0)
    assert isinstance(m.use_second, int)
