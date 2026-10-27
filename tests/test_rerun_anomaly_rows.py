from scripts.rerun_anomaly_rows import TARGETS, norm, resolve_targets


def test_exact_anomaly_targets():
    assert TARGETS == ((25, "wide"), (26, "tournament"), (27, "tournament"))


def test_label_normalization():
    assert norm("wrong_amount_of_cash_received") == "wrong amount of cash received"


def _cache(manifest, rows, text_override=None):
    out = []
    for pos, strategy in TARGETS:
        text, gold = rows[pos]
        if pos == 25 and text_override:
            text = text_override
        for action in ("single", strategy):
            out.append({
                "index": pos, "strategy": action, "text": text,
                "gold": gold, "gold_normalized": norm(gold),
                "choice": norm(gold), "correct": True,
                "latency_ms": 1000, "cumulative_latency_ms": 2000,
            })
    return out


def test_resolve_targets_matches_frozen_manifest_positions():
    rows = [(f"utterance {i}", "foo_bar") for i in range(77)]
    manifest = {"confirmation_indices": list(range(77))}
    result = resolve_targets(manifest, _cache(manifest, rows), rows)
    assert [(r["cache_position"], r["strategy"], r["source_dataset_index"]) for r in result] == [
        (25, "wide", 25), (26, "tournament", 26), (27, "tournament", 27)
    ]


def test_refuses_source_text_mismatch():
    rows = [(f"utterance {i}", "foo_bar") for i in range(77)]
    manifest = {"confirmation_indices": list(range(77))}
    try:
        resolve_targets(manifest, _cache(manifest, rows, text_override="wrong text"), rows)
    except ValueError as exc:
        assert "text mismatch" in str(exc)
    else:
        raise AssertionError("expected source mismatch to be rejected")
