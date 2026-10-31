#!/usr/bin/env python3
"""Semantic Noul/Choice schema registry for the post-gate diagnostic track.

Legacy (no-criteria) Noul is preserved as a historical ablation only.
New main comparisons use explicit true/false semantics + a neutral Choice.
"""
from __future__ import annotations

CODE_DEFECT_NOUL = {
    "type": "noul",
    "instructions": "Does this function contain an exploitable security vulnerability?",
    "criteria": {
        "false": "The function does not contain an exploitable security vulnerability.",
        "true": "The function contains an exploitable security vulnerability.",
    },
}

FINDVER_NOUL = {
    "type": "noul",
    "instructions": "Is the claim supported by the supplied report evidence?",
    "criteria": {
        "false": "The supplied report evidence does not support the claim.",
        "true": "The supplied report evidence supports the claim.",
    },
}

CODE_DEFECT_CHOICE = {
    "type": "choice",
    "instructions": "Is this function secure or vulnerable?",
    "criteria": {"secure": "The function is secure.",
                 "vulnerable": "The function is vulnerable."},
}

FINDVER_CHOICE = {
    "type": "choice",
    "instructions": "Is the claim supported by the supplied report evidence?",
    "criteria": {"supported": "The claim is supported.",
                 "not_supported": "The claim is not supported."},
}

SCHEMAS = {"code_noul": CODE_DEFECT_NOUL, "findver_noul": FINDVER_NOUL,
           "code_choice": CODE_DEFECT_CHOICE, "findver_choice": FINDVER_CHOICE}


def validate_noul_schema(q):
    """Noul must carry explicit, distinct true/false semantic criteria."""
    if not isinstance(q, dict) or q.get("type") != "noul":
        return False, "not a noul question"
    crit = q.get("criteria")
    if not isinstance(crit, dict) or set(crit) != {"false", "true"}:
        return False, "noul criteria must map exactly false/true to semantic text"
    if not all(isinstance(crit[k], str) and crit[k].strip() for k in ("false", "true")):
        return False, "noul criteria texts must be non-empty strings"
    if crit["false"].strip().lower() == crit["true"].strip().lower():
        return False, "false/true semantics must differ"
    return True, ""


def schema_hash(name):
    import hashlib, json
    return hashlib.sha256(json.dumps(SCHEMAS[name], sort_keys=True).encode()).hexdigest()[:12]
