"""Phase 4 GA: tournament selection, uniform crossover, gaussian/flip mutation.

Generational loop with elitism. All randomness from one seeded RNG (reproducible).
Diversity = mean pairwise L1 distance of gene vectors, logged per generation.
"""
from __future__ import annotations

import random

from . import fitness as F
from . import random_search as RS
from .budget import BudgetExceeded, BudgetTracker
from .genome import CascadeGenome


def crossover(rng: random.Random, a: CascadeGenome, b: CascadeGenome, rate: float) -> CascadeGenome:
    if rng.random() > rate:
        return a
    va, vb = a.to_vector(), b.to_vector()
    return CascadeGenome(*[x if rng.random() < 0.5 else y for x, y in zip(va, vb)]).repaired()


def mutate(rng: random.Random, g: CascadeGenome, rate: float, sigma=0.1) -> CascadeGenome:
    lo, hi, us = g.lo, g.hi, g.use_second
    if rng.random() < rate:
        lo += rng.gauss(0.0, sigma)
    if rng.random() < rate:
        hi += rng.gauss(0.0, sigma)
    if rng.random() < rate:
        us = 1 - us
    return CascadeGenome(lo=lo, hi=hi, use_second=us).repaired()


def diversity(pop: list) -> float:
    vecs = [g.to_vector() for g in pop]
    n = len(vecs)
    if n < 2:
        return 0.0
    tot, cnt = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            tot += sum(abs(a - b) for a, b in zip(vecs[i], vecs[j]))
            cnt += 1
    return round(tot / cnt, 4)


def run(rows: list, population: int, generations: int, crossover_rate: float,
        mutation_rate: float, elitism: int, seed: int,
        tracker: BudgetTracker | None = None, lam: float = 0.05) -> dict:
    rng = random.Random(seed)
    pop = [RS.sample_genome(rng) for _ in range(population)]
    archive = []
    memo = {}

    def score(g):
        key = g.to_vector()
        if key in memo:
            if tracker is not None:
                tracker.charge_eval(key)
            return memo[key]
        if tracker is not None:
            tracker.charge_eval(key)
        fit = F.evaluate(g, rows, lam=lam)
        memo[key] = fit
        archive.append({"genome": {"lo": g.lo, "hi": g.hi, "use_second": g.use_second}, **fit})
        return fit

    fits = [score(g) for g in pop]
    log = []
    stopped_early = False
    for gen in range(generations):
        order = sorted(range(len(pop)), key=lambda i: (fits[i]["scalar"], fits[i]["accuracy"]), reverse=True)
        log.append({"generation": gen,
                    "best_scalar": fits[order[0]]["scalar"],
                    "mean_scalar": round(sum(f["scalar"] for f in fits) / len(fits), 4),
                    "diversity": diversity(pop)})
        elites = [pop[i] for i in order[:elitism]]

        def select():
            cand = rng.sample(range(len(pop)), min(3, len(pop)))
            return pop[max(cand, key=lambda i: (fits[i]["scalar"], fits[i]["accuracy"]))]

        new_pop = list(elites)
        new_fits = [fits[i] for i in order[:elitism]]
        try:
            while len(new_pop) < population:
                a, b = select(), select()
                c = mutate(rng, crossover(rng, a, b, crossover_rate), mutation_rate)
                f = score(c)
                new_pop.append(c)
                new_fits.append(f)
        except BudgetExceeded:
            stopped_early = True
        pop, fits = new_pop, new_fits
        if stopped_early:
            break
    order = sorted(range(len(pop)), key=lambda i: (fits[i]["scalar"], fits[i]["accuracy"]), reverse=True)
    best = pop[order[0]]
    return {"best": {"lo": best.lo, "hi": best.hi, "use_second": best.use_second},
            "best_fit": fits[order[0]], "generations": log, "archive": archive,
            "stopped_early": stopped_early,
            "n_unique_evals": len({(a["genome"]["lo"], a["genome"]["hi"], a["genome"]["use_second"]) for a in archive})}
