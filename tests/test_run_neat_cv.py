from scripts.run_neat_cv import build_stratified_folds, config_text, metrics, pareto_frontier


def toy_cache(examples_per_class):
    by = {}
    labels = ["alpha", "beta", "gamma", "delta"]
    idx = 0
    for gold in labels:
        for sample in range(examples_per_class):
            by[idx] = []
            for j, strategy in enumerate(("single", "wide", "tournament")):
                pred = gold if (idx + j) % 3 != 0 else labels[(labels.index(gold) + 1) % len(labels)]
                by[idx].append({
                    "index": idx, "strategy": strategy,
                    "gold": gold, "gold_normalized": gold,
                    "choice": pred, "correct": pred == gold,
                    "cumulative_latency_ms": (2000.0, 4500.0, 6000.0)[j],
                    "cumulative_calls": j + 1,
                })
            idx += 1
    return by


def test_neat_config_uses_three_actions_and_sparse_initial_topology():
    cfg = config_text(83, 30)
    assert "num_inputs = 83" in cfg
    assert "num_outputs = 3" in cfg
    assert "initial_connection = partial_direct 0.15" in cfg
    assert "fitness_threshold = 2.0" in cfg


def test_metrics_report_correctness_latency_and_action_mix():
    by = toy_cache(3)
    only_single = {i: "single" for i in by}
    m = metrics(by, only_single)
    assert m["n"] == 12
    assert m["action_mix"] == {"single": 12, "wide": 0, "tournament": 0}
    assert m["mean_cumulative_ms"] == 2000.0
    assert 0 <= m["accuracy"] <= 1
    assert 0 <= m["macro_f1"] <= 1


def test_metrics_use_selected_action_for_each_sample_cost():
    by = toy_cache(3)
    choices = {i: ("single" if i % 2 == 0 else "tournament") for i in by}
    m = metrics(by, choices)
    assert m["action_mix"]["single"] == 6
    assert m["action_mix"]["tournament"] == 6
    assert m["mean_cumulative_ms"] == 4000.0


def test_stratified_fold_builder_supports_both_shot_sizes_without_leakage():
    for k in (3, 5):
        by = toy_cache(k)
        folds, examples_per_class = build_stratified_folds(by)
        assert examples_per_class == k
        assert len(folds) == k
        assert all(len(f["test"]) == 4 for f in folds)
        assert all(len(f["train"]) == 4 * (k - 1) for f in folds)
        held_out = [i for f in folds for i in f["test"]]
        assert sorted(held_out) == sorted(by)
        assert len(set(held_out)) == len(by)


def test_pareto_frontier_removes_dominated_policies_and_oracle():
    report = {
        "baseline::slow": {"accuracy": 0.7, "mean_cumulative_ms": 6000, "mean_cumulative_calls": 3},
        "neat::faster": {"accuracy": 0.68, "mean_cumulative_ms": 3000, "mean_cumulative_calls": 1.8},
        "baseline::dominated": {"accuracy": 0.65, "mean_cumulative_ms": 6500, "mean_cumulative_calls": 2.5},
        "oracle::posthoc": {"accuracy": 0.9, "mean_cumulative_ms": 1000, "mean_cumulative_calls": 1},
    }
    frontier = pareto_frontier(report)
    assert {x["policy"] for x in frontier} == {"baseline::slow", "neat::faster"}
