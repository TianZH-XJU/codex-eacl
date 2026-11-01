"""Code defect adapter: wraps the audited v2 function-group manifest.

Row identity matches scripts/run_schema_ablation._code_rows exactly:
official-train-only members, group-safe splits, no valid/test leakage.
Questions mirror the proven pair: noul (vulnerable?) + choice switch
(secure/vulnerable). No model calls here.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from group_oof import function_group_id  # noqa: E402

from .base import TaskAdapter

MANIFEST = ROOT / "results/v2/data_manifest/codexglue_function_groups.json"
RAW = ROOT / "data/raw/codexglue_defect"
NOUL_Q = {"type": "noul", "instructions": "Is this function vulnerable?"}
SWITCH_Q = {"type": "choice", "instructions": "Is this function secure or vulnerable?",
            "criteria": {"secure": None, "vulnerable": None}}


class CodeAdapter(TaskAdapter):
    task_id = "codexglue_defect"
    domain = "code"
    group_unit = "function_text_hash"  # NOT project holdout: exact-duplicate
    # functions share a group; cross-project leakage is NOT controlled here.
    adapter_version = "src-tasks-v1"

    def labels(self):
        return ["secure", "vulnerable"]

    def load_rows(self, split, limit, seed):
        self.check_split(split)
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        js_all = json.loads((RAW / "function.json").read_text(encoding="utf-8"))
        members = manifest.get("members")
        if members is None:
            train_idx = {int(x) for x in (RAW / "train.txt").read_text(encoding="utf-8").split()}
            members = {}
            for i, r in enumerate(js_all):
                if i in train_idx:
                    members.setdefault(function_group_id(r["func"]), []).append(i)
        rng = random.Random(seed)
        groups = list(manifest["splits"][split])
        rng.shuffle(groups)
        rows = []
        for g in groups:
            for i in sorted(members.get(g, [])):
                rows.append({"key": f"idx:{i}", "text": js_all[i]["func"],
                             "gold": bool(js_all[i]["target"]), "group_id": g,
                             "meta": {"idx": i, "project": js_all[i].get("project", "?")}})
                if len(rows) >= limit:
                    return rows
        return rows

    def build_question(self, row, menu=None):
        return {"noul": dict(NOUL_Q), "switch": {k: (dict(v) if isinstance(v, dict) else v)
                                                 for k, v in SWITCH_Q.items()}}

    def primary_metric(self):
        return "macro_f1"
