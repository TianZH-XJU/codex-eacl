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


def test_vela_weight_pin_matches_adapter_and_verifier():
    from models.vela_adapter import PINNED_WEIGHT_SHA256, verify_weights
    assert MODEL_PINS["vela"]["weights_sha256"] == PINNED_WEIGHT_SHA256
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "model.safetensors"
        p.write_bytes(b"fake-weights")
        import hashlib
        good = hashlib.sha256(b"fake-weights").hexdigest()
        assert verify_weights(d, expected=good)["verified"] is True
        with pytest.raises(RuntimeError):
            verify_weights(d, expected="0" * 64)


def test_get_adapter_rejects_unknown_and_constructs_pinned():
    assert get_adapter("laya").revision == MODEL_PINS["laya"]["revision"]
    assert get_adapter("vela").revision == MODEL_PINS["vela"]["revision"]
    with pytest.raises(KeyError):
        get_adapter("gptX")


def test_validate_answers_rejects_bad_outputs():
    from models.base import validate_answers
    ok, _ = validate_answers({"q": {"type": "choice", "criteria": {"a": None, "b": None}}},
                             {"q": {"type": "choice", "choice": "a"}})
    assert ok is True
    ok, f = validate_answers({"q": {"type": "choice", "criteria": {"a": None}}},
                             {"q": {"type": "choice", "choice": "zzz"}})
    assert ok is False and "choice_out_of_criteria" in f["q"]
    ok, f = validate_answers({"q": {"type": "noul"}},
                             {"q": {"type": "noul", "noul": 1.5}})
    assert ok is False and "noul_out_of_range" in f["q"]
    ok, f = validate_answers({"q": {"type": "noul"}}, {"q": {"type": "noul"}})
    assert ok is False and "noul_not_a_number" in f["q"]
    ok, _ = validate_answers({"q": {"type": "score"}}, {"q": {"type": "score", "score": 2.5}})
    assert ok is True
    ok, _ = validate_answers({"q": {"type": "choice"}}, {})
    assert ok is False


def test_adapters_surface_ok_and_parse_ok_separately():
    from models.laya_adapter import LayaAdapter
    from models.vela_adapter import VelaAdapter
    la, va = LayaAdapter(), VelaAdapter()

    class Boom:
        def predict(self, *a, **k):
            raise RuntimeError("backend down")
        def system_one(self, *a, **k):
            raise RuntimeError("backend down")
    la._agent = Boom()
    _, _, rec = la.predict("intent", {"message": "x"},
                           {"q": {"type": "choice", "criteria": {"a": None}}})
    assert rec.ok is False and rec.parse_ok is False and rec.error.startswith("model:")
    va._agent = Boom()
    _, _, rec = va.predict("intent", "x", {"q": {"type": "choice", "criteria": {"a": None}}})
    assert rec.ok is False and rec.parse_ok is False

    class BadChoice:
        def predict(self, *a, **k):
            return {"answers": {"q": {"type": "choice", "choice": "zzz"}}}
        def system_one(self, *a, **k):
            return {"answers": {"q": {"type": "choice", "choice": "zzz"}}, "usage": {}}
    la._agent = BadChoice()
    _, _, rec = la.predict("intent", {"message": "x"},
                           {"q": {"type": "choice", "criteria": {"a": None}}})
    assert rec.ok is True and rec.parse_ok is False and rec.error.startswith("parse:")
    va._agent = BadChoice()
    _, _, rec = va.predict("intent", "x", {"q": {"type": "choice", "criteria": {"a": None}}})
    assert rec.ok is True and rec.parse_ok is False


def test_record_is_frozen_and_complete():
    rec = ModelCallRecord(model="m", revision="r", task="t", input_len=3,
                          output_types=("choice",), ok=True, parse_ok=True,
                          latency_ms=1.0, rss_mb=1.0, n_calls=1,
                          adapter_version="a", prompt_version="p")
    with pytest.raises(Exception):
        rec.ok = False
    assert rec.error is None and rec.extra == {}


def test_span_set_rejected_as_unverified():
    from models.base import validate_answers
    ok, errs = validate_answers(
        {"q": {"type": "span"}}, {"q": {"type": "span", "spans": []}})
    assert ok is False and errs["q"].startswith("unvalidated_type:")
    ok, errs = validate_answers(
        {"q": {"type": "set"}}, {"q": {"type": "set", "items": []}})
    assert ok is False and errs["q"].startswith("unvalidated_type:")
    ok, errs = validate_answers(
        {"q": {"type": "mystery"}}, {"q": {"type": "mystery"}})
    assert ok is False and errs["q"].startswith("unknown_question_type:")


def test_validate_prob_map():
    from models.base import validate_prob_map
    clean, err = validate_prob_map({"a": 0.7, "b": 0.3})
    assert err is None and clean == {"a": 0.7, "b": 0.3}
    clean, err = validate_prob_map({"a": 1.5})
    assert clean == {} and err is not None
    clean, err = validate_prob_map({"a": float("nan")})
    assert clean == {} and err is not None
    clean, err = validate_prob_map({"a": float("inf")})
    assert clean == {} and err is not None
    clean, err = validate_prob_map({"a": "x"})
    assert clean == {} and err is not None
    clean, err = validate_prob_map({})
    assert clean == {} and err is not None
    clean, err = validate_prob_map(None)
    assert clean == {} and err is not None
