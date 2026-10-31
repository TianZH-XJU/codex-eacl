import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models.base import DecisionModel, ModelCallRecord, UnsupportedOutputError  # noqa: E402
from models.registry import ADAPTERS, MODEL_PINS, get_adapter  # noqa: E402


class FakeModel(DecisionModel):
    name = "fake"
    revision = "test"

    def load(self, device="cpu"):
        if device != "cpu":
            raise RuntimeError("cpu only")
        return self

    def supported_outputs(self, task):
        if task != "intent":
            raise UnsupportedOutputError(task)
        return {"choice"}

    def predict(self, task, state, questions, prompt_version="default"):
        self.check_outputs(task, questions)
        return {"q": {"type": "choice", "choice": "a"}}, {}, ModelCallRecord(
            model=self.name, revision=self.revision, task=task, input_len=1,
            output_types=("choice",), ok=True, parse_ok=True, latency_ms=1.0,
            rss_mb=1.0, n_calls=1, adapter_version="t", prompt_version=prompt_version)

    def unload(self):
        return True


def test_interface_conformance_without_weights():
    m = FakeModel().load()
    ans, usage, rec = m.predict("intent", {"message": "hi"},
                                {"q": {"type": "choice", "criteria": {"a": None}}})
    assert ans["q"]["choice"] == "a" and rec.ok and rec.parse_ok
    assert rec.prompt_version == "default"
    with pytest.raises(UnsupportedOutputError):
        m.predict("intent", {"message": "hi"}, {"q": {"type": "span"}})
    with pytest.raises(UnsupportedOutputError):
        m.predict("nope", {"message": "hi"}, {"q": {"type": "choice"}})
    with pytest.raises(RuntimeError):
        FakeModel().load(device="cuda")


def test_registry_pins_match_v4_yaml():
    y = yaml.safe_load((ROOT / "configs/v4/models.yaml").read_text(encoding="utf-8"))
    assert MODEL_PINS["laya"]["revision"] == y["laya"]["revision"]
    assert MODEL_PINS["vela"]["revision"] == y["vela"]["revision"]
    assert MODEL_PINS["laya"]["source_commit"] == y["laya"]["source_commit"]
    assert set(ADAPTERS) == {"laya", "vela"}


def test_get_adapter_rejects_unknown_and_constructs_pinned():
    assert get_adapter("laya").revision == MODEL_PINS["laya"]["revision"]
    assert get_adapter("vela").revision == MODEL_PINS["vela"]["revision"]
    with pytest.raises(KeyError):
        get_adapter("gptX")


def test_record_is_frozen_and_complete():
    rec = ModelCallRecord(model="m", revision="r", task="t", input_len=3,
                          output_types=("choice",), ok=True, parse_ok=True,
                          latency_ms=1.0, rss_mb=1.0, n_calls=1,
                          adapter_version="a", prompt_version="p")
    with pytest.raises(Exception):
        rec.ok = False
    assert rec.error is None and rec.extra == {}
