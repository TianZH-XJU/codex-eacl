import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evolution import (AbstainGenome, BudgetExceeded, BudgetTracker,  # noqa: E402
                       CascadeGenome, crossover, crowding_distance, decide,
                       diversity, es_run, evaluate, evaluate_math, ga_run,
                       hypervolume_2d, mutate, nondominated_sort, nsga2_run,
                       random_run, sample_genome)


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
        assert t.charge_eval((i,)) is True
    with pytest.raises(BudgetExceeded):
        t.charge_eval((9,))
    assert t.state()["generated"] == 4 and t.state()["unique_evals"] == 3
    # duplicates: no unique consumed, counted separately
    t2 = BudgetTracker(max_unique_evals=3, max_model_calls=10)
    assert t2.charge_eval((0,)) is True
    assert t2.charge_eval((0,)) is False
    assert t2.state()["unique_evals"] == 1 and t2.state()["duplicate_hits"] == 1
    assert t2.state()["generated"] == 2
    t3 = BudgetTracker(max_unique_evals=100, max_model_calls=2)
    t3.charge_eval((1,), model_calls=2)
    with pytest.raises(BudgetExceeded):
        t3.charge_eval((2,), model_calls=1)
    assert t3.state()["model_calls"] == 2
    # duplicates never re-charge model calls (memoized, no re-evaluation)
    assert t3.charge_eval((1,), model_calls=99) is False
    assert t3.state()["model_calls"] == 2


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


def test_ga_memoizes_duplicates_without_unique_charge():
    rows = synth(n=40, seed=0)
    t = BudgetTracker(max_unique_evals=100, max_model_calls=0)
    g = ga_run(rows, population=4, generations=2, crossover_rate=0.0,
               mutation_rate=0.0, elitism=1, seed=3, tracker=t)
    # zero variation: children duplicate parents; uniques stay bounded
    assert g["n_unique_evals"] <= 4
    assert t.state()["generated"] >= t.state()["unique_evals"]


def test_ga_stops_gracefully_when_budget_exhausted():
    from evolution import BudgetTracker
    rows = synth(n=30, seed=2)
    t = BudgetTracker(max_unique_evals=10, max_model_calls=0)
    g = ga_run(rows, population=8, generations=5, crossover_rate=0.8,
               mutation_rate=0.2, elitism=1, seed=5, tracker=t)
    assert g["stopped_early"] is True
    assert g["n_unique_evals"] == 10  # cap exactly consumed, no off-by-one
    # best must come from actually evaluated genomes, not thin air
    assert g["best_fit"]["scalar"] == max(a["scalar"] for a in g["archive"])


def test_seeding_budget_exhaustion_returns_partial_best():
    from evolution import BudgetTracker, es_run, nsga2_run
    rows = synth(n=30, seed=2)
    t = BudgetTracker(max_unique_evals=3, max_model_calls=0)
    g = ga_run(rows, population=8, generations=5, crossover_rate=0.8,
               mutation_rate=0.2, elitism=1, seed=5, tracker=t)
    assert g["stopped_early"] is True and g["n_unique_evals"] <= 3
    t = BudgetTracker(max_unique_evals=3, max_model_calls=0)
    e = es_run(rows, mu=4, lam=8, budget=24, seed=5, tracker=t)
    assert e["stopped_early"] is True and e["n_unique_evals"] <= 3


def test_budget_always_enforced_without_caller_tracker():
    rows = synth(n=30, seed=2)
    g = ga_run(rows, population=8, generations=3, crossover_rate=0.8,
               mutation_rate=0.2, elitism=1, seed=5)
    # 8 seeding + 3*7 children = 29 slots max; unique bounded by slots taken
    assert g["n_unique_evals"] <= 8 + 3 * 7
    assert g["stopped_early"] is False  # internal cap (1e9) not hit


def test_es_deterministic_budgeted_and_second_gene_fixed():
    rows = synth(n=80, seed=4)
    t = BudgetTracker(max_unique_evals=24, max_model_calls=0)
    e1 = es_run(rows, mu=4, lam=8, sigma=0.15, budget=24, seed=17, tracker=t)
    e2 = es_run(rows, mu=4, lam=8, sigma=0.15, budget=24, seed=17,
                tracker=BudgetTracker(max_unique_evals=24, max_model_calls=0))
    assert e1["best"] == e2["best"] and e1["n_unique_evals"] <= 24
    assert all(a["genome"]["use_second"] == 1 for a in e1["archive"])
    assert e1["stopped_early"] is True  # mu+lam slots exceed 24 unique


def test_es_beats_nothing_claim_is_structural():
    # On structured cache ES must at least match fixed switch-only; the point
    # is structural validity, not superiority (verdict comes from real caches).
    rows = synth(n=200, seed=1)
    e = es_run(rows, budget=24, seed=17,
               tracker=BudgetTracker(max_unique_evals=24, max_model_calls=0))
    base = sum(r["switch_correct"] for r in rows) / len(rows)
    assert e["best_fit"]["accuracy"] >= base - 0.05


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


def test_nsga2_sort_crowding_hypervolume():
    objs = [(1.0, 0.0), (0.5, 0.5), (0.0, 1.0), (0.2, 0.2)]
    fronts = nondominated_sort(objs)
    assert sorted(fronts[0]) == [0, 1, 2]
    assert fronts[1] == [3]
    d = crowding_distance(objs, [0, 1, 2])
    assert d[0] == d[2] == float("inf") and d[1] > 0
    assert hypervolume_2d([(1.0, 0.0), (0.5, 0.5), (0.0, 1.0)]) == 0.25
    assert hypervolume_2d([(0.5, 0.5)]) == 0.25
    assert hypervolume_2d([(0.65, -1.0), (0.45, -2.0)], ref=(0.0, -2.0)) == 0.65


def test_nsga2_runs_with_budget_and_seed():
    import random
    rows = synth(n=40, seed=9)

    def ev(g):
        fit = evaluate(g, rows)
        return ((fit["accuracy"], -fit["calls_per_row"]), g.to_vector(), fit)

    def smp(rng):
        return sample_genome(rng)

    def cx(rng, a, b, rate):
        return crossover(rng, a, b, rate)

    def mu(rng, g, rate):
        return mutate(rng, g, rate)

    t = BudgetTracker(max_unique_evals=12, max_model_calls=0)
    r1 = nsga2_run(ev, smp, cx, mu, budget=12, seed=11, population=6,
                   generations=3, tracker=t)
    r2 = nsga2_run(ev, smp, cx, mu, budget=12, seed=11, population=6,
                   generations=3,
                   tracker=BudgetTracker(max_unique_evals=12, max_model_calls=0))
    assert r1["pareto"] == r2["pareto"] and r1["n_unique_evals"] <= 12
    assert r1["pareto"] and all(len(p["objectives"]) == 2 for p in r1["pareto"])
    # HV is 0.0 by construction here: objectives are (accuracy, -calls) with the
    # default ref (0, 0), so the second objective never exceeds ref. The nonzero-HV
    # math is covered by the hypervolume_2d unit tests above.
    assert r1["hypervolume"] == 0.0


def test_nsga2_mulambda_semantics():
    import random
    rows = synth(n=40, seed=9)

    def ev(g):
        fit = evaluate(g, rows)
        return ((fit["accuracy"], -fit["calls_per_row"]), g.to_vector(), fit)

    def smp(rng):
        return sample_genome(rng)

    def cx(rng, a, b, rate):
        return crossover(rng, a, b, rate)

    def mu(rng, g, rate):
        return mutate(rng, g, rate)
    t = BudgetTracker(max_unique_evals=100, max_model_calls=0)
    r = nsga2_run(ev, smp, cx, mu, budget=100, seed=11, population=8,
                  generations=2, tracker=t, mu=4, lams=6)
    assert all(len(g["objectives"]) == 2 for g in r["pareto"])
    assert r["n_unique_evals"] <= 100


def test_selective_policy_math():
    rows = [{"correct": True, "top1": 0.9, "margin": 0.5},
            {"correct": False, "top1": 0.4, "margin": 0.05},
            {"correct": True, "top1": 0.6, "margin": 0.2},
            {"correct": False, "top1": None, "margin": None}]
    g = AbstainGenome(t=0.5, use_margin=0, m=0.0).repaired()
    fit = evaluate_math(g, rows)
    assert fit["sel_accuracy"] == round(2 / 3, 4) and fit["coverage"] == 0.75
    assert fit["n_answered"] == 3 and fit["n_missing"] == 1
    g2 = AbstainGenome(t=0.5, use_margin=1, m=0.1).repaired()
    fit2 = evaluate_math(g2, rows)
    assert fit2["n_answered"] == 3 and fit2["sel_accuracy"] == round(2 / 3, 4)
    assert AbstainGenome.from_vector(g.to_vector()) == g


def test_cascade_missing_outputs_fail_closed():
    rows = [{"gold": True, "noul_p": None, "switch_correct": False},
            {"gold": False, "noul_p": None, "switch_correct": False},
            {"gold": True, "noul_p": 0.9, "switch_correct": True}]
    fit = evaluate(CascadeGenome(lo=0.0, hi=1.0, use_second=0), rows)
    assert fit["n"] == 3 and fit["n_missing"] == 2
    assert fit["accuracy"] == round(1 / 3, 4)
    # TP=1, FP=1, FN=1, TN=0 -> F1_pos=0.5, F1_neg=0.0
    assert fit["macro_f1"] == 0.25
    assert fit["balanced_accuracy"] == 0.25


def test_abstain_roundtrip_nontrivial_genes():
    # Regression for the swapped use_margin/m bug: non-zero genes must survive.
    for g in (AbstainGenome(t=0.3, use_margin=1, m=0.7),
              AbstainGenome(t=0.9, use_margin=1, m=0.1),
              AbstainGenome(t=0.0, use_margin=0, m=1.0),
              AbstainGenome(t=0.62, use_margin=1, m=0.44)):
        assert AbstainGenome.from_vector(g.to_vector()) == g.repaired(), g


def test_crossover_uses_correct_gene_positions():
    # cx_abstain path (mirrors run_nsga2) must preserve gene semantics.
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_nsga2 import cx_abstain, mu_abstain
    import random
    rng = random.Random(0)
    a = AbstainGenome(t=0.2, use_margin=0, m=0.9).repaired()
    b = AbstainGenome(t=0.8, use_margin=1, m=0.1).repaired()
    for _ in range(20):
        c = cx_abstain(rng, a, b, 1.0)
        assert c.use_margin in (0, 1)
        assert 0.0 <= c.t <= 1.0 and 0.0 <= c.m <= 1.0
        assert AbstainGenome.from_vector(c.to_vector()) == c
        m = mu_abstain(rng, c, 1.0)
        assert AbstainGenome.from_vector(m.to_vector()) == m


def test_primary_metric_drives_scalar():
    rows = ([{"gold": True, "noul_p": 0.9, "switch_correct": True}] * 6
            + [{"gold": False, "noul_p": 0.9, "switch_correct": False}] * 2
            + [{"gold": False, "noul_p": 0.1, "switch_correct": True}] * 2)
    g = CascadeGenome(lo=0.0, hi=1.0, use_second=0).repaired()
    fa = evaluate(g, rows, primary="accuracy")
    fm = evaluate(g, rows, primary="macro_f1")
    assert fa["scalar"] == fa["accuracy"] and fm["scalar"] == fm["macro_f1"]
    assert fa["macro_f1"] != fa["accuracy"]  # imbalanced: metrics differ
    with pytest.raises(ValueError):
        evaluate(g, rows, primary="nope")


def test_nsga2_mulambda_args_take_effect():
    import random
    rows = synth(n=30, seed=4)

    def ev(g):
        fit = evaluate(g, rows)
        return ((fit["accuracy"], -fit["calls_per_row"]), g.to_vector(), fit)

    def smp(rng):
        return sample_genome(rng)

    def cx(rng, a, b, rate):
        return crossover(rng, a, b, rate)

    def mu(rng, g, rate):
        return mutate(rng, g, rate)
    from evolution import BudgetTracker
    t = BudgetTracker(max_unique_evals=100, max_model_calls=10 ** 9)
    r = nsga2_run(ev, smp, cx, mu, budget=100, seed=1, population=8,
                  generations=1, tracker=t, mu=3, lams=9)
    # mu=3 parents kept; one generation creates at most 9 children
    assert r["n_unique_evals"] <= 3 + 9
    assert all(len(g["genome"]) == 3 for g in r["pareto"])


def test_mid_generation_budget_keeps_scored_children_es():
    # Budget exhausts mid-generation: already-scored children must be merged
    # into selection, not discarded with the unfilled generation.
    rows = synth(n=40, seed=11)
    t = BudgetTracker(max_unique_evals=6, max_model_calls=0)
    e = es_run(rows, mu=4, lam=8, budget=6, seed=5, tracker=t)
    assert e["stopped_early"] is True
    assert e["n_unique_evals"] == 6
    # best must come from the archive (all evaluated genomes), not thin air
    arch_best = max(a["scalar"] for a in e["archive"])
    assert abs(e["best_fit"]["scalar"] - arch_best) < 1e-9


def test_mid_generation_budget_keeps_scored_children_nsga2():
    rows = synth(n=40, seed=11)

    def ev(g):
        fit = evaluate(g, rows)
        return ((fit["accuracy"], -fit["calls_per_row"]), g.to_vector(), fit)

    def smp(rng):
        return sample_genome(rng)

    def cx(rng, a, b, rate):
        return crossover(rng, a, b, rate)

    def mu(rng, g, rate):
        return mutate(rng, g, rate)
    t = BudgetTracker(max_unique_evals=6, max_model_calls=0)
    r = nsga2_run(ev, smp, cx, mu, budget=6, seed=5, population=4,
                  generations=5, tracker=t)
    assert r["stopped_early"] is True
    assert r["n_unique_evals"] == 6
    arch_objs = {tuple(a["objectives"]) for a in r["archive"]}
    assert all(tuple(g["objectives"]) in arch_objs for g in r["pareto"])


def test_budget_param_enforced_without_explicit_tracker():
    # es/nsga2 must respect `budget` even when tracker is None (auto-tracker).
    rows = synth(n=40, seed=11)

    def ev(g):
        fit = evaluate(g, rows)
        return ((fit["accuracy"], -fit["calls_per_row"]), g.to_vector(), fit)

    def smp(rng):
        return sample_genome(rng)

    def cx(rng, a, b, rate):
        return crossover(rng, a, b, rate)

    def mu(rng, g, rate):
        return mutate(rng, g, rate)
    e = es_run(rows, mu=4, lam=8, budget=6, seed=5, tracker=None)
    assert e["stopped_early"] is True and e["n_unique_evals"] <= 6
    r = nsga2_run(ev, smp, cx, mu, budget=6, seed=5, population=4,
                  generations=5, tracker=None)
    assert r["stopped_early"] is True and r["n_unique_evals"] <= 6
