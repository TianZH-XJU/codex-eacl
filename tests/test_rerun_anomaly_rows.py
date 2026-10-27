import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import rerun_anomaly_rows as m


def test_sleep_api_wrapper_returns_bool():
    assert isinstance(m.prevent_sleep(), bool)


def test_git_sha_helper_handles_bad_path():
    assert m.git_sha(Path("C:/definitely/not/here")) is None
