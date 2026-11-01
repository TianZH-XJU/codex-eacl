"""Phase 4 baselines: matched-budget random search over the genome space."""
from __future__ import annotations

import random

from . import fitness as F
from .budget import BudgetTracker
from .genome import CascadeGenome


def sample_genome(rng: random.Random) -> CascadeGenome:
    return CascadeGenome(lo=rng.random(), hi=rng.random(),
                         use_second=rng.randint(0, 1)).repaired()


def run(rows: list, budget: int, seed: int, tracker: BudgetTracker | None = None,
        lam: float = 0.05, metric: str = "accuracy") -> dict:
    from .budget import BudgetExceeded
    rng = random.Random(seed)
    best, best_fit, history = None, None, []
    seen = set()
    stopped_early = False
    while len(seen) < budget:
        g = sample_genome(rng)
        key = g.to_vector()
        if tracker is not None:
            # Charge BEFORE recording: a failed charge must not leak the key
            # into `seen`/`n_evals`. Duplicates are charged too (consistent
            # with GA/ES/NSGA2 memo-hit accounting) and skipped without eval.
            try:
                is_new = tracker.charge_eval(key)
            except BudgetExceeded:
                stopped_early = True
                break
            if not is_new:
                continue
        else:
            if key in seen:
                continue
        seen.add(key)
        fit = F.evaluate(g, rows, lam=lam, primary=metric)
        history.append({"genome": {"lo": g.lo, "hi": g.hi, "use_second": g.use_second}, **fit})
        if best_fit is None or (fit["scalar"], fit["accuracy"]) > (best_fit["scalar"], best_fit["accuracy"]):
            best, best_fit = g, fit
    if best is None:
        raise BudgetExceeded("budget exhausted before any evaluation")
    return {"best": {"lo": best.lo, "hi": best.hi, "use_second": best.use_second}, "best_fit": best_fit,
            "n_evals": len(seen), "history": history, "stopped_early": stopped_early}
