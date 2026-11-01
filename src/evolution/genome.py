"""Phase 4 genome: two-stage noul->switch cascade policy for binary tasks.

Policy: always run noul (1 call). If use_second and lo <= p <= hi, also run
switch (2nd call) and take the switch verdict; else take the noul verdict.
Genes: lo, hi in [0,1] (uncertainty band), use_second in {0,1}.
Repair: swap lo/hi if inverted; clip to [0,1]; use_second coerced to 0/1.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CascadeGenome:
    lo: float
    hi: float
    use_second: int

    def repaired(self) -> "CascadeGenome":
        lo, hi = float(self.lo), float(self.hi)
        lo = min(max(lo, 0.0), 1.0)
        hi = min(max(hi, 0.0), 1.0)
        if lo > hi:
            lo, hi = hi, lo
        return CascadeGenome(lo=lo, hi=hi, use_second=1 if self.use_second else 0)

    def to_vector(self) -> tuple:
        return (self.lo, self.hi, float(self.use_second))

    @staticmethod
    def from_vector(v) -> "CascadeGenome":
        return CascadeGenome(lo=v[0], hi=v[1], use_second=int(round(v[2]))).repaired()


GENE_BOUNDS = {"lo": (0.0, 1.0), "hi": (0.0, 1.0), "use_second": (0.0, 1.0)}
