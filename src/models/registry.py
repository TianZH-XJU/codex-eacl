"""Model registry: single source of pinned model identities.

Pins must match configs/v4/models.yaml (enforced by tests/test_model_registry.py).
No new revision may be introduced by code; change the YAML + test together.
"""
from __future__ import annotations

from .laya_adapter import LayaAdapter
from .vela_adapter import VelaAdapter

MODEL_PINS = {
    "laya": {
        "package": "laya==0.4.16",
        "source_commit": "b9fe351ce81e2d974bca2bf219c1e4030c0a05ea",
        "revision": "7b928d828b7b0e022f929d9bd2e44165aa270148",
        "backend": "torch_fp32_cpu",
    },
    "vela": {
        "repo": "vllm-sr/Vela-2.0-0.3B",
        "revision": "d6f03aa9baca4f9017fa8f8e9c2764d3bed3f409",
        "backend": "torch_fp32_cpu",
        "weights_file": "model.safetensors",
        "weights_sha256": "bd37dd0db177d766bc554eded78ae8493bc248172d0480f6a543b30932a0cc8e",
    },
}

ADAPTERS = {
    "laya": LayaAdapter,
    "vela": VelaAdapter,
}


def get_adapter(name: str):
    try:
        cls = ADAPTERS[name]
    except KeyError:
        raise KeyError(f"unknown model {name!r}; registered: {sorted(ADAPTERS)}") from None
    pin = MODEL_PINS[name]
    return cls(revision=pin["revision"])
