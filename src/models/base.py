"""Stage 1 unified model interface (thin adapters over proven call patterns).

Does NOT hide model differences: every call returns a ModelCallRecord with
model/revision/task/input length/output types/success/parse/latency/RSS/call
count/adapter+prompt versions. Unsupported output types raise explicitly.
CPU-only; one model resident at a time (enforced by caller, not here).
"""
from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelCallRecord:
    model: str
    revision: str
    task: str
    input_len: int
    output_types: tuple
    ok: bool
    parse_ok: bool
    latency_ms: float
    rss_mb: float
    n_calls: int
    adapter_version: str
    prompt_version: str
    error: str | None = None
    extra: dict = field(default_factory=dict)


class UnsupportedOutputError(ValueError):
    pass


class DecisionModel(abc.ABC):
    name: str = "base"
    revision: str = "unknown"
    adapter_version: str = "v1"

    @abc.abstractmethod
    def load(self, device: str = "cpu"):
        """Load the frozen checkpoint. Must refuse CUDA (CPU-only policy)."""

    @abc.abstractmethod
    def supported_outputs(self, task: str) -> set:
        """Output types this model supports for the task (e.g. choice/noul/score)."""

    @abc.abstractmethod
    def predict(self, task: str, state, questions: dict, prompt_version: str = "default"):
        """Return (answers, usage, record). Raises UnsupportedOutputError for
        task/question types outside supported_outputs(task)."""

    @abc.abstractmethod
    def unload(self):
        """Release the resident model (free the single-model slot)."""

    def check_outputs(self, task: str, questions: dict):
        allowed = self.supported_outputs(task)
        bad = {q: v.get("type") for q, v in questions.items()
               if not isinstance(v, dict) or v.get("type") not in allowed}
        if bad:
            raise UnsupportedOutputError(f"{self.name}/{task}: unsupported types {bad}")
        return True

    @staticmethod
    def now_ms(t0: float) -> float:
        return round((time.perf_counter() - t0) * 1000.0, 1)
