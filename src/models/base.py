"""Stage 1 unified model interface (thin adapters over proven call patterns).

Does NOT hide model differences: every call returns a ModelCallRecord with
model/revision/task/input length/output types/success/parse/latency/RSS/call
count/adapter+prompt versions. Unsupported output types raise explicitly.
CPU-only; one model resident at a time (enforced by caller, not here).
"""
from __future__ import annotations

import abc
import math
import time
from dataclasses import dataclass, field


def validate_prob_map(probs) -> tuple:
    """Validate a {label: p} map: finite floats in [0, 1].
    Returns (clean_dict, error_or_None). Never raises."""
    if not isinstance(probs, dict) or not probs:
        return {}, "empty_or_not_a_map"
    try:
        clean = {str(k): float(v) for k, v in probs.items()}
    except (TypeError, ValueError, AttributeError):
        return {}, "not_numeric"
    for v in clean.values():
        if not (math.isfinite(v) and 0.0 <= v <= 1.0):
            return {}, f"out_of_range_or_nonfinite:{v!r}"
    return clean, None


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


def _norm_key(s):
    return str(s or "").strip().replace(" ", "_").lower()


def validate_answers(questions: dict, answers: dict):
    """Per-question output validation. Returns (parse_ok, failures dict).

    - choice: non-empty string choice; when criteria are given it must match
      one of them (normalized) else "choice_out_of_criteria".
    - noul: "noul" must be a finite number in [0, 1].
    - score: "score" must be a finite number.
    - span/set: declared-but-unverified (no experiment has issued span/set
      questions); fail closed with "unvalidated_type".
    Missing answers or wrong shapes are parse failures, never silent."""
    failures = {}
    if not isinstance(answers, dict) or not answers:
        return False, {"_all": "empty_or_missing_answers"}
    for name, q in questions.items():
        if not isinstance(q, dict):
            failures[name] = "bad_question_schema"
            continue
        ans = answers.get(name)
        if not isinstance(ans, dict):
            failures[name] = "missing_answer"
            continue
        qtype = q.get("type")
        if ans.get("type") != qtype and "type" in ans:
            failures[name] = f"type_mismatch:{ans.get('type')}"
            continue
        if qtype == "choice":
            ch = ans.get("choice")
            if not isinstance(ch, str) or not ch.strip():
                failures[name] = "empty_choice"
                continue
            crit = q.get("criteria")
            if isinstance(crit, dict) and crit:
                allowed = {_norm_key(k) for k in crit}
                if _norm_key(ch) not in allowed:
                    failures[name] = f"choice_out_of_criteria:{ch!r}"
                    continue
        elif qtype == "noul":
            try:
                p = float(ans.get("noul"))
            except (TypeError, ValueError):
                failures[name] = "noul_not_a_number"
                continue
            if not (math.isfinite(p) and 0.0 <= p <= 1.0):
                failures[name] = f"noul_out_of_range:{ans.get('noul')!r}"
                continue
        elif qtype == "score":
            try:
                s = float(ans.get("score"))
            except (TypeError, ValueError):
                failures[name] = "score_not_a_number"
                continue
            if not math.isfinite(s):
                failures[name] = "score_not_finite"
                continue
        elif qtype in ("span", "set"):
            # Declared-but-unverified: no experiment has ever issued span/set
            # questions, so no response schema has been observed. Fail closed
            # rather than silently accept an unchecked shape.
            failures[name] = f"unvalidated_type:{qtype}"
            continue
        else:
            failures[name] = f"unknown_question_type:{qtype!r}"
            continue
    return (len(failures) == 0), failures


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
