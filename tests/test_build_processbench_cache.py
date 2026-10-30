import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_processbench_cache as b


def _row(**kw):
    r = {"dataset": "processbench", "split": "pilot_train", "seed": 17, "position": 0,
         "row_id": "x", "source_file": "gsm8k.json", "gold_rendered": "step 0", "gold_index": 0,
         "n_candidates": 5, "n_steps": 4, "strategy": "single", "choice": "step 0",
         "correct": True, "latency_ms": 100.0, "calls": 1, "error": None,
         "strategy_features": {}, "tournament": {"rounds": 0, "n": 5},
         "initial_choice": "step 0", "initial_answer_confidence": 0.9,
         "initial_entropy_confidence": 0.5, "initial_top1_prob": 0.9, "initial_top2_prob": 0.05,
         "initial_margin": 0.85, "initial_entropy": 0.5, "initial_act_probability": 1.0,
         "initial_latency_ms": 100.0, "initial_calls": 1, "initial_error": None,
         "initial_n_options": 5, "initial_text_len": 50,
         "incremental_latency_ms": 0.0, "cumulative_latency_ms": 100.0, "cumulative_calls": 1}
    r.update(kw)
    return r


def test_row_features_handles_none():
    f = b.row_features(None)
    assert f["top1_prob"] is None and f["margin"] is None and f["entropy"] is None


def test_row_features_margin_entropy():
    f = b.row_features({"answer_confidence": 0.7, "confidence": 0.6,
                        "action": {"act_probability": 1.0},
                        "probabilities": {"a": 0.7, "b": 0.2, "c": 0.1}})
    assert abs(f["margin"] - 0.5) < 1e-9
    assert f["entropy"] > 0


def test_record_schema_keys():
    r = _row()
    required = {"strategy", "correct", "latency_ms", "calls", "cumulative_latency_ms",
                "cumulative_calls", "initial_margin", "initial_entropy", "gold_index",
                "tournament", "row_id", "source_file"}
    assert required.issubset(r.keys())
