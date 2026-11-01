"""NSGA-II (Deb et al. 2002): fast non-dominated sort + crowding distance.

Generic over fixed-length gene vectors with caller-supplied evaluate/mutate/
crossover. Objectives are maximized. Terminates on unique-eval budget
exhaustion (stopped_early) or generation cap. Seeded RNG throughout.
"""
from __future__ import annotations

import random


def dominates(a: tuple, b: tuple) -> bool:
    return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))


def nondominated_sort(objs: list) -> list:
    """Returns fronts: lists of indices, front[0] = non-dominated."""
    n = len(objs)
    dom_count = [0] * n
    dominated = [[] for _ in range(n)]
    fronts = [[]]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if dominates(objs[i], objs[j]):
                dominated[i].append(j)
            elif dominates(objs[j], objs[i]):
                dom_count[i] += 1
        if dom_count[i] == 0:
            fronts[0].append(i)
    k = 0
    while fronts[k]:
        nxt = []
        for i in fronts[k]:
            for j in dominated[i]:
                dom_count[j] -= 1
                if dom_count[j] == 0:
                    nxt.append(j)
        k += 1
        fronts.append(nxt)
    return [f for f in fronts if f]


def crowding_distance(objs: list, front: list) -> dict:
    dist = {i: 0.0 for i in front}
    if len(front) <= 2:
        for i in front:
            dist[i] = float("inf")
        return dist
    m = len(objs[0])
    for k in range(m):
        ordered = sorted(front, key=lambda i: objs[i][k])
        fmin, fmax = objs[ordered[0]][k], objs[ordered[-1]][k]
        dist[ordered[0]] = dist[ordered[-1]] = float("inf")
        span = fmax - fmin
        if span <= 0:
            continue
        for a, b, c in zip(ordered, ordered[1:], ordered[2:]):
            dist[b] += (objs[c][k] - objs[a][k]) / span
    return dist


def hypervolume_2d(points: list, ref=(0.0, 0.0)) -> float:
    """Hypervolume of the non-dominated subset (both objectives maximized)."""
    nd = [p for p in points if not any(dominates(q, p) and q != p for q in points)]
    nd = sorted(set(nd))
    hv, prev_x = 0.0, ref[0]
    for x, y in sorted(nd):
        if x > prev_x and y > ref[1]:
            hv += (x - prev_x) * (y - ref[1])
            prev_x = x
    return round(hv, 4)


def run(evaluate, sample, crossover, mutate, budget: int, seed: int,
        population: int = 8, generations: int = 3, crossover_rate: float = 0.8,
        mutation_rate: float = 0.2, tracker=None, max_generations: int = 50) -> dict:
    """evaluate(genome) -> (objectives tuple, key tuple, record dict).
    sample(rng)/crossover(rng,a,b,rate)/mutate(rng,g,rate) caller-supplied."""
    from .budget import BudgetExceeded
    rng = random.Random(seed)
    pop = [sample(rng) for _ in range(population)]
    archive = []

    def score(g):
        objs, key, rec = evaluate(g)
        if tracker is not None:
            tracker.charge_eval(key)
        archive.append({"key": list(key), "objectives": list(objs), **rec})
        return objs

    fits = [score(g) for g in pop]
    log, gen, stopped_early = [], 0, False
    try:
        while gen < min(generations, max_generations):
            fronts = nondominated_sort(fits)
            crowd = {}
            for f in fronts:
                crowd.update(crowding_distance(fits, f))
            rank = {}
            for r, f in enumerate(fronts):
                for i in f:
                    rank[i] = r

            def select():
                cand = rng.sample(range(len(pop)), min(2, len(pop)))
                return pop[min(cand, key=lambda i: (rank[i], -crowd[i]))]

            children = []
            while len(children) < population:
                a, b = select(), select()
                children.append(mutate(rng, crossover(rng, a, b, crossover_rate), mutation_rate))
            scored = [(c, score(c)) for c in children]
            pool, pool_f = pop + [c for c, _ in scored], fits + [f for _, f in scored]
            fronts = nondominated_sort(pool_f)
            crowd = {}
            for f in fronts:
                crowd.update(crowding_distance(pool_f, f))
            rank = {}
            for r, f in enumerate(fronts):
                for i in f:
                    rank[i] = r
            order = sorted(range(len(pool)), key=lambda i: (rank[i], -crowd[i]))
            pop = [pool[i] for i in order[:population]]
            fits = [pool_f[i] for i in order[:population]]
            fronts0 = nondominated_sort(fits)
            log.append({"generation": gen, "front0_size": len(fronts0[0]),
                        "hypervolume": hypervolume_2d(fits)})
            gen += 1
    except BudgetExceeded:
        stopped_early = True
    fronts = nondominated_sort(fits)
    return {"pareto": [{"genome": pop[i].to_vector() if hasattr(pop[i], "to_vector") else pop[i],
                        "objectives": list(fits[i])} for i in fronts[0]],
            "hypervolume": hypervolume_2d(fits), "generations": log,
            "stopped_early": stopped_early,
            "n_unique_evals": len({tuple(a["key"]) for a in archive})}
