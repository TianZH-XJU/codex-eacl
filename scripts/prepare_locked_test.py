#!/usr/bin/env python3
"""Freeze the v4.1 locked-test manifest and protocol BEFORE any test call.

Sources are the official held-out splits only:
  * CodeXGLUE defect: data/raw/codexglue_defect/test.txt
  * FinDVer: external/FinDVer/data/test.json (official test, not testmini)

The manifest is group-aware (function hash / report), samples 60 rows per cell
with a fixed seed, and records a SHA-256 hash.  The protocol records templates,
frozen GA genomes (seeds 17/23/41), metrics and bootstrap settings.  This script
does not load a model or read any model output.
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from group_oof import function_group_id  # noqa: E402

SEED = 20261102
N = 60
OUT = ROOT / "results/v4_1/locked_test"
OUT.mkdir(parents=True, exist_ok=True)

NOUL_Q = {"type": "noul", "instructions": "Is this function vulnerable?"}
CHOICE_Q = {"type": "choice", "instructions": "Is this function secure or vulnerable?",
            "criteria": {"secure": None, "vulnerable": None}}
FINDVER_NOUL = {"type": "noul", "instructions": "Is the claim supported by the evidence?"}
FINDVER_CHOICE = {"type": "choice", "instructions": "Is the claim supported by the evidence?",
                  "criteria": {"supported": None, "not supported": None}}


def sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def code_rows():
    raw = ROOT / "data/raw/codexglue_defect"
    js = json.loads((raw / "function.json").read_text(encoding="utf-8"))
    idx = [int(x) for x in (raw / "test.txt").read_text(encoding="utf-8").split()]
    groups = {}
    for i in idx:
        groups.setdefault(function_group_id(js[i]["func"]), []).append(i)
    keys = sorted(groups)
    random.Random(SEED).shuffle(keys)
    rows = []
    for g in keys:
        i = groups[g][0]
        rows.append({"key": f"idx:{i}", "text": js[i]["func"],
                     "gold": bool(js[i]["target"]), "group_id": g,
                     "meta": {"idx": i, "project": js[i].get("project", "?")}})
        if len(rows) >= N:
            break
    return rows


def findver_rows():
    root = ROOT / "external/FinDVer"
    data = json.loads((root / "data/test.json").read_text(encoding="utf-8"))
    by_report = {}
    for r in data:
        by_report.setdefault(r["report"], []).append(r)
    reports = sorted(by_report)
    random.Random(SEED).shuffle(reports)
    rows = []
    for rep in reports:
        for r in sorted(by_report[rep], key=lambda x: x["example_id"]):
            ev = "\n".join(json.loads((root / "financial_reports" / r["report"]).read_text(encoding="utf-8"))["context"][i]["context"]
                           for i in r["relevant_context"])
            rows.append({"key": r["example_id"],
                         "text": f"Claim: {r['statement']}\n\nEvidence:\n{ev}",
                         "gold": bool(r["entailment_label"]), "group_id": f"rp:{r['report']}",
                         "meta": {"subset": r["subset"], "report": r["report"]}})
            if len(rows) >= N:
                return rows
    return rows


def genomes():
    out = {}
    for cell in ("code_laya", "code_vela", "findver_laya", "findver_vela"):
        p = ROOT / f"results/v4_1/evolution/evo_train_{cell}.json"
        d = json.loads(p.read_text(encoding="utf-8"))
        out[cell] = {seed: d["per_seed"][seed]["ga"]["best"] for seed in ("17", "23", "41")}
    return out


def main():
    manifests = {
        "code": {"source": "data/raw/codexglue_defect/test.txt", "seed": SEED,
                 "rows": code_rows()},
        "findver": {"source": "external/FinDVer/data/test.json", "seed": SEED,
                    "rows": findver_rows()},
    }
    for name, m in manifests.items():
        m["n"] = len(m["rows"])
        m["n_groups"] = len({r["group_id"] for r in m["rows"]})
        m["sha256"] = sha({"source": m["source"], "seed": m["seed"], "rows": m["rows"]})
        (OUT / f"manifest_{name}.json").write_text(json.dumps(m, indent=2) + "\n", encoding="utf-8")
        print(name, "n", m["n"], "groups", m["n_groups"], "sha", m["sha256"][:16])
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_test_calls",
        "test_sources": {k: {"source": v["source"], "sha256": v["sha256"], "n": v["n"]}
                         for k, v in manifests.items()},
        "templates": {
            "code": {"noul": NOUL_Q, "choice": CHOICE_Q},
            "findver": {"noul": FINDVER_NOUL, "choice": FINDVER_CHOICE},
        },
        "genomes": genomes(),
        "metrics": ["accuracy", "macro_f1", "net_repair", "calls_per_row"],
        "primary_comparison": "GA (train-selected, seed 17/23/41) minus fixed switch-only, macro-F1",
        "secondary_comparisons": ["GA minus fixed Noul", "fixed switch minus fixed Noul"],
        "bootstrap": {"resamples": 5000, "seed": 17, "unit": "group_id", "ci": 0.95},
        "cost": {"lambda": 0.05, "fixed_switch_calls": 2, "ga_calls": "1 + switch"},
        "stop_rule": "one-shot evaluation; no threshold, prompt, genome or metric changes after this file is written",
    }
    protocol["protocol_sha256"] = sha(protocol)
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    print("protocol sha", protocol["protocol_sha256"][:16])


if __name__ == "__main__":
    raise SystemExit(main())
