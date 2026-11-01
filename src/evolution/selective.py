"""Math selective-prediction policy: abstain genome for step-localization.

Genome = {t in [0,1] (top-1 threshold), use_margin in {0,1}, m in [0,1]}.
Abstain iff top1 < t OR (use_margin AND margin < m). Objectives (maximized):
selective accuracy (accuracy on answered) and coverage (answer rate).
1 call/row always; abstention is the only lever. Cache-exact evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AbstainGenome:
    t: float
    use_margin: int
    m: float

    def repaired(self) -> "AbstainGenome":
        t = min(max(float(self.t), 0.0), 1.0)
        m = min(max(float(self.m), 0.0), 1.0)
        return AbstainGenome(t=t, use_margin=1 if self.use_margin else 0, m=m)

    def to_vector(self) -> tuple:
        return (self.t, float(self.use_margin), self.m)

    @staticmethod
    def from_vector(v) -> "AbstainGenome":
        return AbstainGenome(t=v[0], use_margin=int(round(v[1])), m=v[2]).repaired()


def evaluate_math(genome: AbstainGenome, rows: list) -> dict:
    g = genome.repaired()
    answered, correct = 0, 0
    for r in rows:
        if r.get("top1") is None:
            continue
        abstain = r["top1"] < g.t or (g.use_margin and (r.get("margin") or 0.0) < g.m)
        if abstain:
            continue
        answered += 1
        correct += int(bool(r.get("correct")))
    cov = answered / max(len(rows), 1)
    return {"sel_accuracy": round(correct / max(answered, 1), 4),
            "coverage": round(cov, 4), "n_answered": answered, "n": len(rows)}
