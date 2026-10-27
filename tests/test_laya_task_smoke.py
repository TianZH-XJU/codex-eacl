import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def test_smoke_script_imports_without_running_model():
    import ast
    tree = ast.parse(Path(ROOT / "scripts" / "laya_task_smoke.py").read_text(encoding="utf-8"))
    names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert "main" in names
    src = Path(ROOT / "scripts" / "laya_task_smoke.py").read_text(encoding="utf-8")
    for token in ("predict_tournament", "noul", "score", "head_max_len",
                  "tournament_passthrough_small", "noul_custom_labels", "score_shape",
                  "P(true)", "legend", "--task", "--limit"):
        assert token in src, token
