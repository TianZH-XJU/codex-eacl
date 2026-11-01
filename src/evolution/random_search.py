"""Phase 4 baselines: matched-budget random search over the genome space."""
from __future__ import annotations

import random

from . import fitness as F
from .budget import BudgetTracker
from .genome import CascadeGenome


def sample_genome(rng: random.Random) -> CascadeGenome:
    return CascadeGenome(lo=rng.random(), hi=rng.random(),
                         use_second=rng.randint(0, 1)).repaired()


def run(rows: list, budget: int, seed: int, tracker: BudgetTracker | None = None) -> dict:
    rng = random.Random(seed)
    best, best_fit, history = None, None, []
    seen = set()
    while len(seen) < budget:
        g = sample_genome(rng)
        key = g.to_vector()
        if key in seen:
            continue
        seen.add(key)
        if tracker is not None:
            tracker.charge_eval(key)
        fit = F.evaluate(g, rows)
        history.append({"genome": {"lo": g.lo, "hi": g.hi, "use_second": g.use_second}, **fit})
        if best_fit is None or (fit["scalar"], fit["accuracy"]) > (best_fit["scalar"], best_fit["accuracy"]):
            best, best_fit = g, fit
    return {"best": {"lo": best.lo, "hi": best.hi, "use_second": best.use_second}, "best_fit": best_fit,
            "n_evals": len(seen), "history": history}
