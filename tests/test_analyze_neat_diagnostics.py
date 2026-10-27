import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_neat_diagnostics import fold_stability


def test_fold_stability_filters_lambda_and_kinds():
    recs = [
        {"fold": 0, "mode": "choice", "lambda_per_ms": 2e-05, "seed": 17,
         "best_genome": {"fitness": 0.5, "nodes": 3, "connections_enabled": 4},
         "fold_oof_metrics": {"accuracy": 0.6, "mean_cumulative_ms": 4000.0,
                              "action_mix": {"single": 1, "wide": 0, "tournament": 0}}},
        {"fold": 0, "mode": "choice", "lambda_per_ms": 0.0, "seed": 17,
         "best_genome": {}, "fold_oof_metrics": {}},
        {"fold": 1, "mode": "choice", "lambda_per_ms": 2e-05, "seed": 17,
         "topology_type": "fixed_full_direct",
         "best_genome": {"fitness": 0.4, "nodes": 2, "connections_enabled": 3},
         "fold_oof_metrics": {"accuracy": 0.5, "mean_cumulative_ms": 3000.0,
                              "action_mix": {"single": 0, "wide": 1, "tournament": 0}}},
    ]
    rows = fold_stability(recs)
    assert len(rows) == 2
    assert rows[0]["kind"] == "fixed_ea_choice"
    assert rows[1]["kind"] == "neat_choice"
    assert rows[1]["train_fitness"] == 0.5
