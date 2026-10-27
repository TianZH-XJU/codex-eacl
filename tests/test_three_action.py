import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_three_action import choose_action, estimate_train_costs


def test_choose_action_uses_train_costs_not_dev_latency():
    costs = {"single": {"mean_cumulative_ms": 2000.0, "mean_cumulative_calls": 1.0},
             "wide": {"mean_cumulative_ms": 4500.0, "mean_cumulative_calls": 2.0},
             "tournament": {"mean_cumulative_ms": 6000.0, "mean_cumulative_calls": 3.0}}
    # wide has highest p but with a large lambda the cheaper single wins,
    # even though per-sample dev latency is not an input at all.
    probs = {"single": 0.5, "wide": 0.9, "tournament": 0.55}
    assert choose_action(probs, costs, 0.0) == "wide"
    assert choose_action(probs, costs, 2e-04) == "single"


def test_estimate_train_costs_keys():
    by = {0: [{"strategy": s, "cumulative_latency_ms": 1000.0 * (i + 1),
               "cumulative_calls": i + 1} for i, s in enumerate(["single", "wide", "tournament"])]}
    costs = estimate_train_costs(by)
    assert set(costs) == {"single", "wide", "tournament"}
    assert costs["single"]["mean_cumulative_ms"] == 1000.0
