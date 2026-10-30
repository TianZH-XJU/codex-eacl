import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_config_sensitivity import config_hash, validate_config


def test_validate_config_accepts_preregistered():
    for cfg in [{"action": "wide", "kwargs": {"head_max_len": 96, "max_len": 512}},
                {"action": "tournament", "kwargs": {"group_size": 8}},
                {"action": "order", "kwargs": {"order": "REVERSE"}}]:
        assert validate_config(cfg, 20) == (True, "")


def test_validate_config_rejects_bad():
    assert validate_config({"action": "tournament", "kwargs": {"group_size": 1}}, 20)[0] is False
    assert validate_config({"action": "wide", "kwargs": {"head_max_len": -1}}, 20)[0] is False
    assert validate_config({"action": "order", "kwargs": {"order": "RANDOM4"}}, 20)[0] is False


def test_config_hash_stable_and_short():
    a = {"name": "x", "action": "wide", "kwargs": {"head_max_len": 96}}
    assert config_hash(a) == config_hash(dict(a))
    assert len(config_hash(a)) == 12
