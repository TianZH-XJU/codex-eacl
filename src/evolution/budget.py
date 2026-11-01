"""Phase 4 budget: hard caps on unique genome evaluations and model calls."""
from __future__ import annotations


class BudgetExceeded(Exception):
    pass


class BudgetTracker:
    def __init__(self, max_unique_evals: int, max_model_calls: int):
        self.max_unique_evals = max_unique_evals
        self.max_model_calls = max_model_calls
        self.unique_evals = 0
        self.generated = 0
        self.duplicate_hits = 0
        self.model_calls = 0
        self._seen = set()

    def charge_eval(self, genome_key, model_calls=0):
        """Charge one evaluation. Returns True iff the genome is new (unique).
        Duplicates never consume unique budget. Raises BudgetExceeded BEFORE
        counting when the charge would exceed a cap, so counters never cross
        the limit and the caller must not run the evaluation."""
        self.generated += 1
        if genome_key not in self._seen:
            if self.unique_evals + 1 > self.max_unique_evals:
                raise BudgetExceeded(f"unique evals would exceed {self.max_unique_evals}")
            if self.model_calls + model_calls > self.max_model_calls:
                raise BudgetExceeded(f"model calls would exceed {self.max_model_calls}")
            self._seen.add(genome_key)
            self.unique_evals += 1
            self.model_calls += model_calls
            return True
        self.duplicate_hits += 1
        return False

    def state(self):
        return {"unique_evals": self.unique_evals, "generated": self.generated,
                "duplicate_hits": self.duplicate_hits, "model_calls": self.model_calls,
                "max_unique_evals": self.max_unique_evals,
                "max_model_calls": self.max_model_calls}
