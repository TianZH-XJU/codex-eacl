import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "results/v2/evolutionary_bank.json"


def test_bank_entries_unique_and_valid():
    assert BANK.exists(), "run scripts/build_evolutionary_bank.py first"
    d = json.loads(BANK.read_text(encoding="utf-8"))
    assert d["n_entries"] == len(d["entries"]) > 0
    keys = [e["entry"] for e in d["entries"]]
    assert len(set(keys)) == len(keys), "duplicate bank entries"
    for e in d["entries"]:
        assert 0.0 <= e["accuracy"] <= 1.0
        assert e["n"] > 0 and e["mean_latency_ms"] > 0
    tasks = {e["task"] for e in d["entries"]}
    assert {"code_defect", "findver"} <= tasks
