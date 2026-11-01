"""Phase 4 fitness: cache-exact evaluation of cascade genomes.

The cache holds REAL single-call outputs per row (noul_p, switch correctness).
Genome evaluation is a deterministic function of cached outputs plus exact
call counting — no surrogate error is possible (nothing is approximated).
Model-call cost of cache construction is recorded separately by the cache
builder, not hidden here.
"""
from __future__ import annotations

import math

from .genome import CascadeGenome


def clean_p(p):
    """Usable noul probability or None. Non-numeric, non-finite (NaN/Inf) and
    out-of-[0,1] values are NOT predictions — they are corrupt outputs and must
    fail closed downstream, never become silent correct/incorrect labels."""
    try:
        v = float(p)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v) or not 0.0 <= v <= 1.0:
        return None
    return v


def decide(genome: CascadeGenome, noul_p: float, switch_correct: bool, gold: bool) -> tuple:
    """Returns (correct: bool, calls: int). Garbage noul_p fails closed as
    (False, 1): an unusable model output cannot be rescued by any policy."""
    g = genome.repaired()
    v = clean_p(noul_p)
    if v is None:
        return False, 1
    if g.use_second and g.lo <= v <= g.hi:
        return bool(switch_correct), 2
    return bool((v >= 0.5) == bool(gold)), 1


def evaluate(genome: CascadeGenome, rows: list, lam: float = 0.05,
             primary: str = "accuracy") -> dict:
    """rows: [{gold: bool, noul_p: float|None, switch_correct: bool|None}].
    Fail-closed: rows with missing outputs count as WRONG (1 call) and stay in
    the denominator. Reports accuracy, macro-F1 and balanced accuracy over ALL
    rows. scalar = primary - lam*(calls_per_row - 1); primary must be the task
    primary metric (v4.1 protocol). "accuracy" is the v4-frozen legacy."""
    if not rows:
        raise ValueError("evaluate: empty rows (fail-closed, no silent scalar)")
    correct, calls = 0, 0
    n = len(rows)
    tp = tn = fp = fn = 0
    n_missing = 0
    for r in rows:
        if clean_p(r.get("noul_p")) is None:
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
    prec_pos = tp / max(tp + fp, 1)
    rec_pos = tp / max(tp + fn, 1)
    f1_pos = 2 * prec_pos * rec_pos / max(prec_pos + rec_pos, 1e-12)
    prec_neg = tn / max(tn + fn, 1)
    rec_neg = tn / max(tn + fp, 1)
    f1_neg = 2 * prec_neg * rec_neg / max(prec_neg + rec_neg, 1e-12)
    tnr = tn / max(tn + fp, 1)
    macro_f1 = round((f1_pos + f1_neg) / 2, 4)
    bal_acc = round((rec_pos + tnr) / 2, 4)
    acc = round(correct / n, 4)
    if primary == "macro_f1":
        quality = macro_f1
    elif primary == "balanced_accuracy":
        quality = bal_acc
    elif primary == "accuracy":
        quality = acc
    else:
        raise ValueError(f"unknown primary metric {primary!r}")
    return {"accuracy": acc,
            "calls_per_row": round(calls / n, 4),
            "n": n, "n_missing": n_missing,
            "macro_f1": macro_f1,
            "balanced_accuracy": bal_acc,
            "lam": lam, "primary": primary,
            "scalar": round(quality - lam * (calls / n - 1.0), 4)}
