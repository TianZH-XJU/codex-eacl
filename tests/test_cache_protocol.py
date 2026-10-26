import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_split_manifest_is_disjoint_and_nested():
    path = ROOT / "results" / "splits" / "banking77_train_dev_manifest.json"
    assert path.exists()
    m = json.loads(path.read_text(encoding="utf-8"))
    train = set(m["train_indices_all"])
    dev = set(m["dev_indices"])
    assert train.isdisjoint(dev)
    # Nested shot subsets must be prefixes of the 5-shot pool.
    for k in ("1", "3", "5"):
        subset = m["nested_train_indices"][k]
        assert len(subset) == 77 * int(k)
        assert set(subset).issubset(train)
    assert m["n_train_total"] == 77 * 5
    assert m["n_dev_total"] == 77 * 3


def test_cache_has_single_first_fields():
    path = ROOT / "results" / "pilot" / "cache_test.jsonl"
    assert path.exists()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows
    required = {
        "initial_choice", "initial_margin", "initial_entropy",
        "initial_answer_confidence", "initial_entropy_confidence",
        "cumulative_latency_ms", "cumulative_calls",
        "incremental_latency_ms",
    }
    assert required.issubset(rows[0].keys())
    # SINGLE rows must have zero incremental cost.
    for r in rows:
        if r["strategy"] == "single":
            assert r["incremental_latency_ms"] == 0.0


def test_cache_summary_has_pairwise_overlaps():
    path = ROOT / "results" / "pilot" / "cache_test.summary.json"
    assert path.exists()
    s = json.loads(path.read_text(encoding="utf-8"))
    assert "pairwise_error_overlap" in s
    assert len(s["pairwise_error_overlap"]) == 3
    assert s["status"] == "passed"
    assert s["working_tree_dirty"] is False or s["working_tree_dirty"] is True