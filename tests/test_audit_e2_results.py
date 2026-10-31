import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import audit_e2_results as A


def test_audit_passes_on_committed_results():
    assert A.main() == 0


def test_audit_script_is_offline():
    src = (ROOT / "scripts" / "audit_e2_results.py").read_text(encoding="utf-8")
    assert "AutoModel" not in src and "laya.load" not in src
    assert "system_one" not in src and "agent.predict" not in src
