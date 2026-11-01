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


def evaluate(genome: CascadeGenome, rows: list, lam: float = 0.05) -> dict:
    """rows: [{gold: bool, noul_p: float|None, switch_correct: bool|None}].
    Fail-closed: rows with missing outputs count as WRONG (1 call) and stay in
    the denominator. Reports accuracy, macro-F1 and balanced accuracy over ALL
    rows. scalar = accuracy - lam*(calls_per_row - 1): lam prices the 2nd call."""
    correct, calls = 0, 0
    n = len(rows)
    tp = tn = fp = fn = 0
    n_missing = 0
    for r in rows:
        if r.get("noul_p") is None:
            n_missing += 1
            calls += 1
            if bool(r["gold"]):
                fn += 1
            else:
                fp += 1
            continue
        ok, c = decide(genome, r["noul_p"],
                       bool(r.get("switch_correct", False)), r["gold"])
        correct += int(ok)
        calls += c
        if ok and bool(r["gold"]):
            tp += 1
        elif ok:
            tn += 1
        elif bool(r["gold"]):
            fn += 1
        else:
            fp += 1
    n = max(n, 1)
    prec_pos = tp / max(tp + fp, 1)
    rec_pos = tp / max(tp + fn, 1)
    f1_pos = 2 * prec_pos * rec_pos / max(prec_pos + rec_pos, 1e-12)
    prec_neg = tn / max(tn + fn, 1)
    rec_neg = tn / max(tn + fp, 1)
    f1_neg = 2 * prec_neg * rec_neg / max(prec_neg + rec_neg, 1e-12)
    tnr = tn / max(tn + fp, 1)
    return {"accuracy": round(correct / n, 4),
            "calls_per_row": round(calls / n, 4),
            "n": n, "n_missing": n_missing,
            "macro_f1": round((f1_pos + f1_neg) / 2, 4),
            "balanced_accuracy": round((rec_pos + tnr) / 2, 4),
            "lam": lam,
            "scalar": round(correct / n - lam * (calls / n - 1.0), 4)}
