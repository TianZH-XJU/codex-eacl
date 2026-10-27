from scripts.run_neat_cv import config_text, metrics


def toy_cache():
    by = {}
    labels = ["alpha", "beta", "gamma"]
    for idx in range(9):
        gold = labels[idx // 3]
        by[idx] = []
        for j, strategy in enumerate(("single", "wide", "tournament")):
            pred = gold if (idx + j) % 3 != 0 else labels[(idx // 3 + 1) % 3]
            by[idx].append({
                "index": idx, "strategy": strategy,
                "gold": gold, "gold_normalized": gold,
                "choice": pred, "correct": pred == gold,
                "cumulative_latency_ms": (2000.0, 4500.0, 6000.0)[j],
                "cumulative_calls": j + 1,
            })
    return by


def test_neat_config_uses_three_actions_and_sparse_initial_topology():
    cfg = config_text(83, 30)
    assert "num_inputs = 83" in cfg
    assert "num_outputs = 3" in cfg
    assert "initial_connection = partial_direct 0.15" in cfg
    assert "fitness_threshold = 2.0" in cfg


def test_metrics_report_correctness_latency_and_action_mix():
    by = toy_cache()
    only_single = {i: "single" for i in by}
    m = metrics(by, only_single)
    assert m["n"] == 9
    assert m["action_mix"] == {"single": 9, "wide": 0, "tournament": 0}
    assert m["mean_cumulative_ms"] == 2000.0
    assert 0 <= m["accuracy"] <= 1
    assert 0 <= m["macro_f1"] <= 1


def test_metrics_use_selected_action_for_each_sample_cost():
    by = toy_cache()
    choices = {i: ("single" if i % 3 == 0 else "tournament") for i in by}
    m = metrics(by, choices)
    assert m["action_mix"]["single"] == 3
    assert m["action_mix"]["tournament"] == 6
    assert m["mean_cumulative_ms"] == 4666.7
