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

from .base import DecisionModel, ModelCallRecord, UnsupportedOutputError, validate_answers

ROOT = Path(__file__).resolve().parents[2]

WEIGHTS_FILE = "model.safetensors"
PINNED_WEIGHT_SHA256 = "bd37dd0db177d766bc554eded78ae8493bc248172d0480f6a543b30932a0cc8e"


def verify_weights(snapshot_dir, expected=PINNED_WEIGHT_SHA256):
    """SHA-256 of the local Vela snapshot weights vs the pinned value.
    Raises RuntimeError on mismatch (fail-closed). Set VELA_SKIP_HASH=1 to
    skip (recorded in the call record)."""
    import hashlib
    import os
    if os.getenv("VELA_SKIP_HASH") == "1":
        return {"verified": False, "skipped": True}
    path = Path(snapshot_dir) / WEIGHTS_FILE
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    digest = h.hexdigest()
    if digest != expected:
        raise RuntimeError(f"Vela weights hash mismatch: got {digest[:16]}..., want {expected[:16]}...")
    return {"verified": True, "sha256": digest}

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
        self._weights = {"verified": False, "skipped": None}

    def load(self, device="cpu"):
        if device != "cpu":
            raise RuntimeError("CPU-only policy: Vela adapter refuses non-cpu device")
        if torch.cuda.is_available() and os.getenv("ALLOW_CUDA", "") != "1":
            raise RuntimeError("CPU-only run refused: CUDA is available")
        snap = ROOT / "models" / "vela"
        t0 = time.perf_counter()
        self._weights = verify_weights(snap)
        self._weights["seconds"] = round(time.perf_counter() - t0, 1)
        from transformers import AutoModel
        self._agent = AutoModel.from_pretrained(str(snap), trust_remote_code=True)
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
        model_err, answers, usage = None, {}, {}
        try:
            out = self._agent.system_one(state, questions)
            answers = out.get("answers", {}) or {}
            usage = out.get("usage", {}) or {}
        except Exception as e:  # noqa: BLE001
            model_err = f"model:{type(e).__name__}: {e}"
        ms = self.now_ms(t0)
        text = state if isinstance(state, str) else state.get("message", "")
        types = tuple(sorted({v.get("type") for v in answers.values()
                              if isinstance(v, dict) and v.get("type")}))
        if model_err is None:
            parse_ok, failures = validate_answers(questions, answers)
            err = None if parse_ok else "parse:" + ";".join(f"{k}={v}" for k, v in failures.items())
        else:
            parse_ok, err = False, model_err
        record = ModelCallRecord(
            model=self.name, revision=self.revision, task=task,
            input_len=len(text), output_types=types, ok=model_err is None,
            parse_ok=parse_ok,
            latency_ms=ms, rss_mb=round(proc.memory_info().rss / 2**20, 1),
            n_calls=1, adapter_version=self.adapter_version,
            prompt_version=prompt_version, error=err,
            extra={"weights": self._weights})
        return answers, usage, record

    def unload(self):
        self._agent = None
        return True
