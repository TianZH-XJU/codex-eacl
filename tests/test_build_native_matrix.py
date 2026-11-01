import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_native_matrix import code_rows_for_groups


def _manifest():
    raw = [{"func": "int a() {return 0;}", "target": 0},   # idx 0 train
           {"func": "int b() {return 1;}", "target": 1},   # idx 1 train
           {"func": "int c() {return 2;}", "target": 0},   # idx 2 valid
           {"func": "int a() {return 0;}", "target": 0}]   # idx 3 test (dup of 0)
    manifest = {"members": {"g_train": [0, 1], "g_dup": [0, 3]},
                "excluded_cross_split_leaked_idx": [3],
                "splits": {"search_train": ["g_train"], "search_dev": ["g_dup"]}}
    return manifest, raw


def test_members_only_resolution():
    manifest, raw = _manifest()
    rows = code_rows_for_groups(manifest, raw, ["g_train"], {2}, set())
    assert [t for t, _ in rows] == ["int a() {return 0;}", "int b() {return 1;}"]


def test_cross_split_duplicate_raises():
    manifest, raw = _manifest()
    with pytest.raises(AssertionError, match="non-train idx 3"):
        code_rows_for_groups(manifest, raw, ["g_dup"], {2}, set())


def test_valid_idx_raises():
    manifest = {"members": {"g": [2]}, "excluded_cross_split_leaked_idx": []}
    raw = [{"func": "x", "target": 0}, {"func": "y", "target": 1}, {"func": "z", "target": 0}]
    with pytest.raises(AssertionError, match="non-train idx 2"):
        code_rows_for_groups(manifest, raw, ["g"], {2}, set())


def test_missing_members_manifest_raises():
    with pytest.raises(AssertionError, match="train-only members"):
        code_rows_for_groups({}, [{"func": "x", "target": 0}], ["g"], set(), set())
