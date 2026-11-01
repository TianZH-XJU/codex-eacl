"""Second EA family: (mu+lambda) Evolution Strategy on continuous thresholds.

Searches (lo, hi) only, with use_second fixed to 1: the discrete gene is held
constant so the ablation (forced-0 vs evolved) isolates its value. Mutation is
isotropic Gaussian with fixed sigma; elitist (mu+lambda) selection on scalar
fitness; terminates on unique-eval budget exhaustion. Same cache-exact fitness
as GA (no surrogate). Seeded RNG throughout.
"""
from __future__ import annotations

import random

from . import fitness as F
from .budget import BudgetExceeded, BudgetTracker
from .genome import CascadeGenome


def mutate(rng: random.Random, x: tuple, sigma: float) -> CascadeGenome:
    lo = min(max(x[0] + rng.gauss(0.0, sigma), 0.0), 1.0)
    hi = min(max(x[1] + rng.gauss(0.0, sigma), 0.0), 1.0)
    if lo > hi:
        lo, hi = hi, lo
    return CascadeGenome(lo=lo, hi=hi, use_second=1)


def run(rows: list, mu: int = 4, lam: int = 8, sigma: float = 0.15,
        budget: int = 24, seed: int = 17,
        tracker: BudgetTracker | None = None, lam_w: float = 0.05,
        max_generations: int = 50) -> dict:
    rng = random.Random(seed)
    pop = [CascadeGenome(lo=rng.random(), hi=rng.random(), use_second=1).repaired()
           for _ in range(mu)]
    archive = []

    def score(g):
        if tracker is not None:
            tracker.charge_eval(g.to_vector())
        fit = F.evaluate(g, rows, lam=lam_w)
        archive.append({"genome": {"lo": g.lo, "hi": g.hi, "use_second": g.use_second}, **fit})
        return fit

    fits = [score(g) for g in pop]
    log = []
    gen, stopped_early = 0, False
    try:
        while gen < max_generations:
            order = sorted(range(len(pop)),
                           key=lambda i: (fits[i]["scalar"], fits[i]["accuracy"]), reverse=True)
            log.append({"generation": gen,
                        "best_scalar": fits[order[0]]["scalar"],
                        "mean_scalar": round(sum(f["scalar"] for f in fits) / len(fits), 4)})
            children = []
            scored = []
            while len(children) < lam:
                parent = pop[order[rng.randrange(min(mu, len(order)))]]
                c = mutate(rng, (parent.lo, parent.hi), sigma)
                f = score(c)  # may raise BudgetExceeded -> stop, keep scored so far
                children.append(c)
                scored.append(f)
            pool, pool_f = pop + children, fits + scored
            order = sorted(range(len(pool)),
                           key=lambda i: (pool_f[i]["scalar"], pool_f[i]["accuracy"]), reverse=True)
            pop = [pool[i] for i in order[:mu]]
            fits = [pool_f[i] for i in order[:mu]]
            gen += 1
    except BudgetExceeded:
        stopped_early = True
    order = sorted(range(len(pop)),
                   key=lambda i: (fits[i]["scalar"], fits[i]["accuracy"]), reverse=True)
    best = pop[order[0]]
    return {"best": {"lo": best.lo, "hi": best.hi, "use_second": best.use_second},
            "best_fit": fits[order[0]], "generations": log, "archive": archive,
            "stopped_early": stopped_early,
            "n_unique_evals": len({(a["genome"]["lo"], a["genome"]["hi"], a["genome"]["use_second"])
                                   for a in archive})}
