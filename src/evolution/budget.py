"""Phase 4 budget: hard caps on unique genome evaluations and model calls."""
from __future__ import annotations


class BudgetExceeded(Exception):
    pass


class BudgetTracker:
    def __init__(self, max_unique_evals: int, max_model_calls: int):
        self.max_unique_evals = max_unique_evals
        self.max_model_calls = max_model_calls
        self.unique_evals = 0
        self.model_calls = 0
        self._seen = set()

    def charge_eval(self, genome_key, model_calls=0):
        if genome_key not in self._seen:
            self._seen.add(genome_key)
            self.unique_evals += 1
        self.model_calls += model_calls
        if self.unique_evals > self.max_unique_evals:
            raise BudgetExceeded(f"unique evals {self.unique_evals} > {self.max_unique_evals}")
        if self.model_calls > self.max_model_calls:
            raise BudgetExceeded(f"model calls {self.model_calls} > {self.max_model_calls}")
        return True

    def state(self):
        return {"unique_evals": self.unique_evals, "model_calls": self.model_calls,
                "max_unique_evals": self.max_unique_evals,
                "max_model_calls": self.max_model_calls}
