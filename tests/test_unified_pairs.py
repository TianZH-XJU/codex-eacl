import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_unified_pairs as P  # noqa: E402


def _rows():
    rows = []
    for pos in range(6):
        g = f"g{pos % 3}"
        rows.append({"position": pos, "rep": "noul", "correct": pos % 2 == 0,
                     "latency_ms": 100.0, "group_id": g, "error": None})
        rows.append({"position": pos, "rep": "switch", "correct": pos % 2 == 1,
                     "latency_ms": 200.0, "group_id": g, "error": None})
    return rows


def test_paired_stats_on_synthetic(tmp_path):
    p = tmp_path / "u.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in _rows()), encoding="utf-8")
    out = tmp_path / "o.json"
    sys.argv = ["x", "--in", str(p), "--out", str(out)]
    P.main()
    d = json.loads(out.read_text(encoding="utf-8"))
    assert d["n_positions"] == 6 and d["reps"] == ["noul", "switch"]
    assert d["per_rep"]["noul"]["accuracy"]["est"] == 0.5
    assert d["paired"]["both_correct"] == 0 and d["paired"]["both_wrong"] == 0
    assert d["paired"]["only_fix"] == {"noul": 3, "switch": 3}
    assert d["paired"]["oracle_accuracy"]["est"] == 1.0
    ci = d["paired"]["switch_minus_noul"]
    assert ci["est"] == 0.0 and ci["ci95"][0] <= 0.0 <= ci["ci95"][1]


def test_incomplete_positions_dropped(tmp_path):
    rows = _rows()[:3]  # position 2 missing switch
    p = tmp_path / "u.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    out = tmp_path / "o.json"
    sys.argv = ["x", "--in", str(p), "--out", str(out)]
    P.main()
    d = json.loads(out.read_text(encoding="utf-8"))
    assert d["n_positions"] == 1
