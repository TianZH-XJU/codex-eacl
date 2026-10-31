"""Intent task adapter: thin wrapper over the audited v4 intent pipeline.

Reuses scripts/build_menu_bank.load_pool (group-safe, train-only members) so
row identity matches every published v4 intent result. Adds menu-question
building with the same underscore/space convention as run_intent_baseline.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from build_menu_bank import load_pool  # noqa: E402

from .base import TaskAdapter


class IntentAdapter(TaskAdapter):
    adapter_version = "src-tasks-v1"

    def __init__(self, dataset: str):
        if dataset not in ("banking77", "clinc150"):
            raise ValueError(f"unknown intent dataset {dataset!r}")
        self.dataset = dataset
        self.task_id = f"{dataset}_intent"
        self.domain = "general"
        self.group_unit = "normalized_utterance"

    def labels(self):
        train = load_pool(self.dataset, "search_train", None, 17)
        return sorted({r["label"] for r in train})

    def load_rows(self, split, limit, seed):
        self.check_split(split)
        return load_pool(self.dataset, split, limit, seed)

    def build_question(self, row, menu=None):
        cands = list(menu) if menu is not None else self.labels()
        if row["label"] not in cands:
            cands = list(cands) + [row["label"]]
        return {"type": "choice",
                "instructions": "Which intent does this message express?",
                "criteria": {c.replace("_", " "): None for c in cands}}

    def primary_metric(self):
        return "accuracy"
