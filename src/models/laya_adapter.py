"""Laya adapter: wraps the proven laya.load('english') call pattern.

Supports choice/noul/score where the task adapter declares them; option-order
and tournament parameters are task-config, never silently defaulted.
CPU-only: refuses CUDA. Thread counts are set by the caller process.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import psutil
import torch

from .base import DecisionModel, ModelCallRecord, UnsupportedOutputError

ROOT = Path(__file__).resolve().parents[2]
LAYA_ROOT = ROOT / "external" / "laya"

_TASK_OUTPUTS = {
    "intent": {"choice"},
    "code_defect": {"choice", "noul"},
    "findver": {"choice", "noul"},
    "processbench": {"choice"},
}


class LayaAdapter(DecisionModel):
    name = "laya"
    adapter_version = "src-v1"

    def __init__(self, revision="7b928d828b7b0e022f929d9bd2e44165aa270148"):
        self.revision = revision
        self._agent = None

    def load(self, device="cpu"):
        if device != "cpu":
            raise RuntimeError("CPU-only policy: Laya adapter refuses non-cpu device")
        if torch.cuda.is_available() and os.getenv("ALLOW_CUDA", "") != "1":
            raise RuntimeError("CPU-only run refused: CUDA is available")
        sys.path.insert(0, str(LAYA_ROOT / "research" / "benchmarks" / "tournament"))
        import laya  # noqa: E402
        self._agent = laya.load("english", device="cpu")
        return self

    def supported_outputs(self, task):
        try:
            return set(_TASK_OUTPUTS[task])
        except KeyError:
            raise UnsupportedOutputError(f"laya: unknown task {task!r}") from None

    def predict(self, task, state, questions, prompt_version="default"):
        self.check_outputs(task, questions)
        if self._agent is None:
            raise RuntimeError("LayaAdapter.load() must be called first")
        text = state if isinstance(state, str) else state.get("message", "")
        t0 = time.perf_counter()
        proc = psutil.Process()
        err, answers = None, {}
        try:
            out = self._agent.predict({"message": text}, questions)
            answers = out.get("answers", {}) or {}
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
        ms = self.now_ms(t0)
        types = tuple(sorted({v.get("type") for v in answers.values()
                              if isinstance(v, dict) and v.get("type")}))
        parse_ok = bool(answers) and all(isinstance(v, dict) and v.get("type") for v in answers.values())
        record = ModelCallRecord(
            model=self.name, revision=self.revision, task=task,
            input_len=len(text), output_types=types, ok=err is None, parse_ok=parse_ok,
            latency_ms=ms, rss_mb=round(proc.memory_info().rss / 2**20, 1),
            n_calls=1, adapter_version=self.adapter_version,
            prompt_version=prompt_version, error=err)
        return answers, {}, record

    def unload(self):
        self._agent = None
        return True
