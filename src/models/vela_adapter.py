"""Vela adapter: wraps the proven AutoModel.from_pretrained(models/vela) pattern.

Supports choice/noul/score plus Vela-only span/set (declared per task; the
common cross-model subspace is choice/noul/score only). Laya-only knobs
(max_len, head_max_len, group_size, shortlist_k, window_overlap) are rejected
here by construction: they do not exist in this adapter. CPU Torch FP32 only.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import psutil
import torch

from .base import DecisionModel, ModelCallRecord, UnsupportedOutputError

ROOT = Path(__file__).resolve().parents[2]

_COMMON = {"choice", "noul", "score"}
_TASK_OUTPUTS = {
    "intent": _COMMON,
    "code_defect": _COMMON,
    "findver": _COMMON,
    "processbench": {"choice"},
    "span_ext": _COMMON | {"span", "set"},
}


class VelaAdapter(DecisionModel):
    name = "vela"
    adapter_version = "src-v1"

    def __init__(self, revision="d6f03aa9baca4f9017fa8f8e9c2764d3bed3f409"):
        self.revision = revision
        self._agent = None

    def load(self, device="cpu"):
        if device != "cpu":
            raise RuntimeError("CPU-only policy: Vela adapter refuses non-cpu device")
        if torch.cuda.is_available() and os.getenv("ALLOW_CUDA", "") != "1":
            raise RuntimeError("CPU-only run refused: CUDA is available")
        from transformers import AutoModel
        self._agent = AutoModel.from_pretrained(str(ROOT / "models" / "vela"),
                                                trust_remote_code=True)
        return self

    def supported_outputs(self, task):
        try:
            return set(_TASK_OUTPUTS[task])
        except KeyError:
            raise UnsupportedOutputError(f"vela: unknown task {task!r}") from None

    def predict(self, task, state, questions, prompt_version="default"):
        self.check_outputs(task, questions)
        if self._agent is None:
            raise RuntimeError("VelaAdapter.load() must be called first")
        t0 = time.perf_counter()
        proc = psutil.Process()
        err, answers, usage = None, {}, {}
        try:
            out = self._agent.system_one(state, questions)
            answers = out.get("answers", {}) or {}
            usage = out.get("usage", {}) or {}
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
        ms = self.now_ms(t0)
        text = state if isinstance(state, str) else state.get("message", "")
        types = tuple(sorted({v.get("type") for v in answers.values()
                              if isinstance(v, dict) and v.get("type")}))
        parse_ok = bool(answers) and all(isinstance(v, dict) and v.get("type") for v in answers.values())
        record = ModelCallRecord(
            model=self.name, revision=self.revision, task=task,
            input_len=len(text), output_types=types, ok=err is None, parse_ok=parse_ok,
            latency_ms=ms, rss_mb=round(proc.memory_info().rss / 2**20, 1),
            n_calls=1, adapter_version=self.adapter_version,
            prompt_version=prompt_version, error=err)
        return answers, usage, record

    def unload(self):
        self._agent = None
        return True
