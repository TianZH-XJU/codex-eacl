import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import json
import make_processbench_split as m


def test_split_manifest_is_problem_disjoint(tmp_path):
    rows = [
        {"id": f"r{i}", "problem": f"p{i // 2}", "steps": ["a", "b"], "label": -1 if i % 2 else 0,
         "generator": "g", "final_answer_correct": True}
        for i in range(8)
    ]
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "gsm8k.json").write_text(json.dumps(rows[:4]), encoding="utf-8")
    (raw / "math.json").write_text(json.dumps(rows[4:]), encoding="utf-8")
    (raw / "olympiadbench.json").write_text("[]", encoding="utf-8")
    (raw / "omnimath.json").write_text("[]", encoding="utf-8")
    out = tmp_path / "split.json"
    sys.argv = ["x", "--raw-dir", str(raw), "--seed", "17", "--out", str(out)]
    m.main()
    manifest = json.loads(out.read_text(encoding="utf-8"))
    sets = [set(manifest["problems"][k]) for k in ("train", "dev", "test")]
    assert sets[0].isdisjoint(sets[1]) and sets[0].isdisjoint(sets[2]) and sets[1].isdisjoint(sets[2])
    assert sum(len(s) for s in sets) == 4  # 4 unique problems, no leakage
    assert manifest["n_rows"]["train"] + manifest["n_rows"]["dev"] + manifest["n_rows"]["test"] == 8
