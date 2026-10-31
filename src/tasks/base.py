"""Stage 2 task adapters: unified per-task interface over proven loaders.

Each adapter declares: task id, domain, label set (full, never subset),
row loading (group-safe splits only), question building (full + menu),
and the primary metric. No model calls here. Adapters must refuse test
splits: only search_train/search_dev/confirm pools are addressable.
"""
from __future__ import annotations

import abc


class TaskAdapter(abc.ABC):
    task_id: str = "base"
    domain: str = "unknown"
    group_unit: str = "unknown"
    adapter_version: str = "src-tasks-v1"

    @abc.abstractmethod
    def labels(self) -> list:
        """Full candidate label set (never a sample subset)."""

    @abc.abstractmethod
    def load_rows(self, split: str, limit: int, seed: int) -> list:
        """Rows from group-safe search pools only. split in
        {search_train, search_dev, confirm}. Must be deterministic in seed."""

    @abc.abstractmethod
    def build_question(self, row: dict, menu: list | None = None) -> dict:
        """Typed question for one row; menu=None means full label set."""

    @abc.abstractmethod
    def primary_metric(self) -> str:
        """Pre-registered primary metric name for this task."""

    def check_split(self, split: str):
        allowed = {"search_train", "search_dev", "confirm"}
        if split not in allowed:
            raise ValueError(f"test splits are not addressable; got {split!r}")
        return True
