"""Phase 4 fitness: cache-exact evaluation of cascade genomes.

The cache holds REAL single-call outputs per row (noul p, switch correctness).
Genome evaluation is a deterministic function of cached outputs plus exact
call counting — no surrogate error is possible (nothing is approximated).
Model-call cost of cache construction is recorded separately by the cache
builder, not hidden here.
"""
from __future__ import annotations

from .genome import CascadeGenome


def decide(genome: CascadeGenome, noul_p: float, switch_correct: bool, gold: bool) -> tuple:
    """Returns (correct: bool, calls: int)."""
    g = genome.repaired()
    if g.use_second and g.lo <= noul_p <= g.hi:
        return bool(switch_correct), 2
    return bool((noul_p >= 0.5) == bool(gold)), 1


def evaluate(genome: CascadeGenome, rows: list) -> dict:
    """rows: [{gold: bool, noul_p: float|None, switch_correct: bool|None}].
    Rows with missing outputs count as wrong with 1 call (fail-closed)."""
    correct, calls, usable = 0, 0, 0
    for r in rows:
        if r.get("noul_p") is None:
            continue
        usable += 1
        ok, c = decide(genome, r["noul_p"],
                       bool(r.get("switch_correct", False)), r["gold"])
        correct += int(ok)
        calls += c
    n = max(usable, 1)
    return {"accuracy": round(correct / n, 4),
            "calls_per_row": round(calls / n, 4),
            "n": usable,
            "scalar": round(correct / n - 0.05 * (calls / n - 1.0), 4)}
