"""ProcessBench adapter: wraps the audited v2 problem-group manifest.

v2 manifest stores group hashes only (no members map); members are rebuilt by
scanning official-train rows, excluding history-touched groups (manifest already
excludes them; rebuild asserts the same). Row state text and candidate rendering
match scripts/build_processbench_cache.py byte-for-byte in structure:
"problem\\n\\nSteps:\\n[k] step" + step labels + ALL_STEPS_CORRECT sentinel last.
Gold: label if != -1 else sentinel position (prepare_processbench.to_choice_task).
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from group_oof import problem_group_id  # noqa: E402

from .base import TaskAdapter

MANIFEST = ROOT / "results/v2/data_manifest/processbench_problem_groups.json"
RAW = ROOT / "data/raw/processbench"
FILES = ("gsm8k.json", "math.json", "olympiadbench.json", "omnimath.json")
SENTINEL = "ALL_STEPS_CORRECT"
INSTRUCTIONS = ("Which step contains the earliest error? "
                "Choose the sentinel if all steps are correct.")


def _raw_by_id():
    by_id = {}
    for fn in FILES:
        for r in json.loads((RAW / fn).read_text(encoding="utf-8")):
            by_id[r["id"]] = r
    return by_id


class ProcessbenchAdapter(TaskAdapter):
    task_id = "processbench"
    domain = "math"
    group_unit = "problem_text"
    adapter_version = "src-tasks-v1"

    def labels(self):
        # Candidate set is row-dependent (one per step + sentinel); the only
        # task-global label is the sentinel. Per-row candidates via candidates().
        return [SENTINEL]

    def candidates(self, row):
        steps = row["meta"]["steps"]
        return [f"step {k}" for k in range(len(steps))] + [SENTINEL]

    def load_rows(self, split, limit, seed):
        self.check_split(split)
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        by_id = _raw_by_id()
        groups = list(manifest["splits"][split])
        rng = random.Random(seed)
        rng.shuffle(groups)
        wanted = set(groups)
        rows = []
        # deterministic scan order, then group shuffle order decides emission
        g2rows = {}
        for rid, r in by_id.items():
            g = problem_group_id(r["problem"])
            if g in wanted:
                g2rows.setdefault(g, []).append((rid, r))
        for g in groups:
            for rid, r in sorted(g2rows.get(g, [])):
                steps = r["steps"]
                gold = r["label"] if r["label"] != -1 else len(steps)
                rows.append({"key": rid,
                             "text": r["problem"] + "\n\nSteps:\n" + "\n".join(
                                 f"[{k}] {s}" for k, s in enumerate(steps)),
                             "gold": gold, "group_id": g,
                             "meta": {"steps": steps, "label": r["label"]}})
                if len(rows) >= limit:
                    return rows
        return rows

    def build_question(self, row, menu=None):
        cands = list(menu) if menu is not None else self.candidates(row)
        render = [c.replace("_", " ") for c in cands]
        return {"type": "choice", "instructions": INSTRUCTIONS,
                "criteria": {c: None for c in render}}

    def primary_metric(self):
        return "step_accuracy"
