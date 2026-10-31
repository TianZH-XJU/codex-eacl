#!/usr/bin/env python3
"""Stage 0 audit: independently re-verify E2 joint analysis against raw pilots.

Offline only (no model calls). Checks:
1. project_git_sha in each result file resolves to a real commit.
2. sample/seed/config consistency across the four pilot runs.
3. Recomputed ranges, flip sets, effective-space counts match e2_joint_analysis.json.
4. Determinism files match the analysis within-process records.
5. best{} pointers reference the true argmax genome.
Exit code 0 = all checks pass; prints PASS/FAIL per check.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MENU = ROOT / "results/v2/menu"
PILOTS = {
    "vela_exh": "evo_pilot_banking77_vela_exhaustive.json",
    "vela_rnd": "evo_pilot_banking77_vela_random12.json",
    "laya_exh": "evo_pilot_banking77_laya_exhaustive.json",
    "laya_rnd": "evo_pilot_banking77_laya_random12.json",
}

failures = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f" [{detail}]" if detail else ""))
    if not cond:
        failures.append(name)


def sha_exists(s):
    r = subprocess.run(["git", "cat-file", "-t", s], capture_output=True, text=True, cwd=ROOT)
    return r.stdout.strip() == "commit"


def gkey(g):
    return (g["retriever"], g["k"], g["norm"], g["order"])


def main():
    D = {k: json.loads((MENU / v).read_text(encoding="utf-8")) for k, v in PILOTS.items()}
    A = json.loads((MENU / "e2_joint_analysis.json").read_text(encoding="utf-8"))

    for k, d in D.items():
        check(f"{k} sha resolves", bool(d.get("project_git_sha")) and sha_exists(d["project_git_sha"]),
              str(d.get("project_git_sha"))[:12])
    # explicit consistency: same task/rows/train/seed/device/threads across all four
    base = {k: (D[k]["task"], D[k]["n_rows"], D[k]["n_train"], D[k]["seed"], D[k]["device"], D[k]["threads"])
            for k in D}
    check("same task/rows/train/seed/device/threads",
          len(set(base.values())) == 1 and base["vela_exh"][0] == "banking77", str(set(base.values())))
    check("genome budgets as designed",
          D["vela_exh"]["n_genomes"] == 24 and D["laya_exh"]["n_genomes"] == 24
          and D["vela_rnd"]["n_genomes"] == 12 and D["laya_rnd"]["n_genomes"] == 12)

    for m in ("vela", "laya"):
        accs = [g["accuracy"] for g in D[m + "_exh"]["genomes"]]
        check(f"{m} range matches analysis",
              [min(accs), max(accs)] == A["models"][m]["exhaustive_range"],
              f"{min(accs)}-{max(accs)}")
        ek = {gkey(g["genome"]): g["accuracy"] for g in D[m + "_exh"]["genomes"]}
        rk = {gkey(g["genome"]): g["accuracy"] for g in D[m + "_rnd"]["genomes"]}
        common = set(ek) & set(rk)
        flips = {k for k in common if ek[k] != rk[k]}
        check(f"{m} flip count matches", len(flips) == A["models"][m]["n_flips"],
              f"{len(flips)}/{len(common)}")
        check(f"{m} effective space = 12",
              len([k for k in ek if k[2] == "raw"]) == 12 == A["models"][m]["n_effective_distinct"])
        for s in ("exh", "rnd"):
            d = D[m + "_" + s]
            b = d["best"]
            g = d["genomes"][b["genome_id"]]
            check(f"{m}_{s} best pointer valid",
                  g["genome"] == b["genome"] and g["accuracy"] == b["accuracy"]
                  and b["accuracy"] == max(x["accuracy"] for x in d["genomes"]))

    for m in ("vela", "laya"):
        det = json.loads((MENU / f"determinism_{m}_banking77.json").read_text(encoding="utf-8"))
        a = A[m + "_within_process"]
        check(f"{m} determinism matches analysis",
              (det["acc_pass1"], det["acc_pass2"], det["n_flips"])
              == (a["acc_pass1"], a["acc_pass2"], a["n_flips"]))

    print(f"\n{len(failures)} failures" if failures else "\nALL CHECKS PASS")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
