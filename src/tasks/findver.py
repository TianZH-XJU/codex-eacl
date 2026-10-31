"""FinDVer adapter: wraps the audited v2 report-group manifest.

Row identity matches scripts/run_schema_ablation._findver_rows exactly:
testmini rows grouped by report, state text Claim+Evidence, group rp:<report>.
Questions mirror the proven pair: noul + choice switch (supported/not
supported). Official test labels never read. No model calls here.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

from .base import TaskAdapter

MANIFEST = ROOT / "results/v2/data_manifest/findver_report_groups.json"
TESTMINI = ROOT / "external/FinDVer/data/testmini.json"
REPORTS = ROOT / "external/FinDVer/financial_reports"
NOUL_Q = {"type": "noul", "instructions": "Is the claim supported by the evidence?"}
SWITCH_Q = {"type": "choice", "instructions": "Is the claim supported by the evidence?",
            "criteria": {"supported": None, "not supported": None}}


class FindverAdapter(TaskAdapter):
    task_id = "findver"
    domain = "finance"
    group_unit = "report_id"
    adapter_version = "src-tasks-v1"

    def labels(self):
        return ["not_supported", "supported"]

    def load_rows(self, split, limit, seed):
        self.check_split(split)
        rng = random.Random(seed)
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        data = {r["example_id"]: r for r in json.loads(TESTMINI.read_text(encoding="utf-8"))}
        ids = [r["example_id"] for rep in manifest["splits"][split] for r in
               json.loads(TESTMINI.read_text(encoding="utf-8")) if r["report"] == rep]
        rng.shuffle(ids)
        rows = []
        for eid in ids[:limit]:
            r = data[eid]
            rep = json.loads((REPORTS / r["report"]).read_text(encoding="utf-8"))
            ev = "\n".join(rep["context"][i]["context"] for i in r["relevant_context"])
            rows.append({"key": eid,
                         "text": f"Claim: {r['statement']}\n\nEvidence:\n{ev}",
                         "gold": bool(r["entailment_label"]),
                         "group_id": f"rp:{r['report']}"})
        return rows

    def build_question(self, row, menu=None):
        return {"noul": dict(NOUL_Q), "switch": {k: (dict(v) if isinstance(v, dict) else v)
                                                 for k, v in SWITCH_Q.items()}}

    def primary_metric(self):
        return "macro_f1"
