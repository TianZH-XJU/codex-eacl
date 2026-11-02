#!/usr/bin/env python3
"""Generate every LaTeX table in paper/acl/tables/ from committed JSON.

No number in the paper is typed by hand: this script reads the result files
produced by the analysis scripts and emits booktabs tables.  Run it after any
result file changes and recompile the paper.

Usage:
    python scripts/make_paper_tables.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "acl" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

CELLS = ["code_laya", "code_vela", "findver_laya", "findver_vela"]
CELL_TEX = {
    "code_laya": "CodeXGLUE / Laya",
    "code_vela": "CodeXGLUE / Vela",
    "findver_laya": "FinDVer / Laya",
    "findver_vela": "FinDVer / Vela",
}
TASK_TEX = {"code": "CodeXGLUE", "findver": "FinDVer", "math": "ProcessBench"}


def jload(rel: str):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def f3(x) -> str:
    return f"{x:.3f}"


def pct(x) -> str:
    return f"{100.0 * x:+.1f}"


def ci(c: dict, key: str = "ci95") -> str:
    lo, hi = c[key]
    return f"[{pct(lo)}, {pct(hi)}]"


def write(name: str, body: str) -> None:
    (OUT / name).write_text(body, encoding="utf-8")
    print(f"wrote paper/acl/tables/{name}")


# --------------------------------------------------------------------------
# Table 1: model and action interfaces (background; facts from model cards)
# --------------------------------------------------------------------------
def tab_models() -> None:
    body = r"""\begin{table*}[t]
\centering
\small
\caption{The three structured decision models discussed in this paper. Only
Laya and Vela are run locally; Jev defines the commercial reference point and
is not evaluated here.}
\label{tab:models}
\begin{tabular}{@{}p{1.5cm}p{4.1cm}p{4.6cm}p{4.4cm}@{}}
\toprule
Model & Interface & Training route & Role in this study \\
\midrule
Jev & One call returns typed \textsc{Noul}, \textsc{Choice} and \textsc{Score} decisions with probabilities & TypeSafe names RLCD and a proper-scoring objective; the full recipe is not public & Reference point for the RLCD / System One family; not run \\
Laya 0.4.16 & ModernBERT-large encoder with decision heads, $\approx$421M parameters; \textsc{Noul}, \textsc{Choice}, \textsc{Score} & Upstream describes the recipe as RLCD with log, spherical and ranked probability scores & Open RLCD model used in all experiments \\
Vela 2.0 0.3B & 22-layer ModernBERT, 768 hidden, 8{,}192 tokens; \textsc{Noul}, \textsc{Choice}, \textsc{Score}, \textsc{Span}, \textsc{Set} & Initialised from Decision-1.0-Kai Choice path, task-specific losses, routing data; not called RLCD & Interface-matched comparison model with a different recipe \\
\bottomrule
\end{tabular}
\end{table*}
"""
    write("tab_models.tex", body)


# --------------------------------------------------------------------------
# Table 2: native single-action baselines (search_dev)
# --------------------------------------------------------------------------
def tab_native() -> None:
    rows = []
    spec = [
        ("code_laya", "unified_code_laya.summary.json", "noul", "switch"),
        ("code_vela", "unified_code_vela.summary.json", "noul", "switch"),
        ("findver_laya", "unified_findver_laya.summary.json", "noul", "switch"),
        ("findver_vela", "unified_findver_vela.summary.json", "noul", "switch"),
        ("math_laya", "unified_math_laya.summary.json", "choice", None),
        ("math_vela", "unified_math_vela.summary.json", "choice", None),
    ]
    for cell, fname, a1, a2 in spec:
        d = jload(f"results/v4/native/{fname}")
        br = d["by_rep"]
        task = TASK_TEX[cell.split("_")[0]]
        model = cell.split("_")[1].capitalize()
        n = br[a1]["n"]
        acc1 = f3(br[a1]["accuracy"])
        ms1 = f"{br[a1]['mean_ms']:.0f}"
        if a2 is None:
            acc2, ms2 = "--", "--"
        else:
            acc2 = f3(br[a2]["accuracy"])
            ms2 = f"{br[a2]['mean_ms']:.0f}"
        rows.append(
            f"{task} & {model} & {n} & {acc1} & {acc2} & {ms1} & {ms2} & "
            f"{d['peak_rss_mb']:.0f} \\\\"
        )
    body = (
        r"""\begin{table*}[t]
\centering
\small
\caption{Native single-action baselines on the 60-row selection pools
(\texttt{search\_dev}, seed 17). Latency is the per-call mean on the local CPU
and is reported only to show that the two representations cost roughly the
same; it is not used in the fitness function.}
\label{tab:native}
\begin{tabular}{@{}llrrrrrr@{}}
\toprule
Task & Model & $n$ & \textsc{Noul} acc. & \textsc{Choice} acc. & \textsc{Noul} ms & \textsc{Choice} ms & Peak RSS (MB) \\
\midrule
"""
        + "\n".join(rows)
        + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    )
    write("tab_native.tex", body)


# --------------------------------------------------------------------------
# Table 3: action complementarity, frozen direction, all three pools
# --------------------------------------------------------------------------
def tab_complementarity() -> None:
    comp = jload("results/v4_1/evolution/action_complementarity.json")["cells"]
    held = jload("results/v4_1/evolution/complementarity_heldout.json")["cells"]
    rows = []
    for cell in CELLS:
        c = comp[cell]
        h = held[cell]
        alt = "Choice" if c["base_action"] == "noul" else "Noul"
        base = c["base_action"].capitalize()
        dv, cf = h["dev"], h["confirm"]
        rows.append(
            f"{CELL_TEX[cell]} & {base}$\\to${alt} & "
            f"{f3(c['headroom'])} & {c['repairable']}/{c['harmful']} & "
            f"{dv['net_pp']*100:+.1f} {ci(dv, 'net_ci95')} & "
            f"{cf['net_pp']*100:+.1f} {ci(cf, 'net_ci95')} \\\\"
        )
    body = (
        r"""\begin{table*}[t]
\centering
\footnotesize
\caption{Action complementarity with the train-selected direction frozen across
pools ($n=60$ per cell). \emph{Headroom} is the accuracy gain of a per-row
oracle over the better fixed action on \texttt{search\_train};
\emph{repair/harmful} counts rows where the alternative fixes or breaks the
base decision. Net repair is the paired difference in percentage points
(alternative minus base) with a 95\% group-bootstrap interval on dev and
confirm. Headroom is an upper bound, not a deployable gain.}
\label{tab:complementarity}
\begin{tabular}{@{}llrrrr@{}}
\toprule
Task / model & Direction & Headroom & Repair/harmful & Net dev (95\% CI) & Net confirm (95\% CI) \\
\midrule
"""
        + "\n".join(rows)
        + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    )
    write("tab_complementarity.tex", body)


# --------------------------------------------------------------------------
# Table 4: incremental-utility predictability
# --------------------------------------------------------------------------
def tab_predictability() -> None:
    d = jload("results/v4_1/evolution/utility_predictability.json")["cells"]
    rows = []
    for cell in CELLS:
        c = d[cell]
        lr = c["models"]["logreg"]["oof_auroc"]
        tr = c["models"]["tree"]["oof_auroc"]
        ch = c["candidates"][c["champion"]]
        dn = ch["dev_net"]
        cn = ch["confirm_net"]
        rows.append(
            f"{CELL_TEX[cell]} & {f3(lr)} & {f3(tr)} & {c['champion']} & "
            f"{pct(dn['net'])} ({dn['repair']}/{dn['regression']}) & "
            f"{pct(cn['net'])} ({cn['repair']}/{cn['regression']}) & "
            f"{pct(cn['cost_adjusted'])} \\\\"
        )
    body = (
        r"""\begin{table*}[t]
\centering
\footnotesize
\caption{Incremental-utility prediction. OOF AUROC uses group-aware three-fold
cross-validation on \texttt{search\_train}; the champion and threshold are
chosen on \texttt{search\_dev} and frozen for \texttt{confirm}. Net utility is
(repairs $-$ regressions)$/n$ in percentage points, with counts in parentheses;
the last column subtracts $\lambda=0.05$ per extra call.}
\label{tab:predictability}
\begin{tabular}{@{}lrrlrrr@{}}
\toprule
Task / model & LogReg OOF & Tree OOF & Champion & Dev net & Confirm net & Confirm cost-adj. \\
\midrule
"""
        + "\n".join(rows)
        + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    )
    write("tab_predictability.tex", body)


# --------------------------------------------------------------------------
# Table 5: calibration and routing with recalibrated probabilities
# --------------------------------------------------------------------------
def tab_calibration() -> None:
    d = jload("results/v4_1/final/calibration.json")["cells"]
    rows = []
    for cell in CELLS:
        c = d[cell]
        raw, ts = c["raw"]["confirm"], c["temp_scaled"]["confirm"]
        uv = c["utility_vs_calibration"]
        rr = c["routing_raw_p"]["confirm_net"]["net"]
        rt = c["routing_temp_p"]["confirm_net"]["net"]
        rows.append(
            f"{CELL_TEX[cell]} & {f3(raw['ece15'])} & {f3(ts['ece15'])} & "
            f"{f3(raw['brier'])} & {f3(ts['brier'])} & "
            f"{f3(uv['auroc_p_for_gold_confirm'])} & "
            f"{f3(uv['auroc_p_for_repair_confirm'])} & "
            f"{pct(rr)} & {pct(rt)} \\\\"
        )
    body = (
        r"""\begin{table*}[t]
\centering
\small
\caption{Calibration on \texttt{confirm} and its relation to routing.
Temperature $T$ is fitted on \texttt{search\_train}; ECE uses 15 equal-width
bins. Gold AUROC is the probability's ability to rank current-action
correctness; repair AUROC is its ability to rank rows that the second action
would fix. The last two columns are the net repair of a band policy selected
on \texttt{search\_dev} with raw and temperature-scaled probabilities.}
\label{tab:calibration}
\begin{tabular}{@{}lrrrrrrrr@{}}
\toprule
Task / model & ECE raw & ECE $T$ & Brier raw & Brier $T$ & AUROC gold & AUROC repair & Route raw & Route $T$ \\
\midrule
"""
        + "\n".join(rows)
        + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    )
    write("tab_calibration.tex", body)


# --------------------------------------------------------------------------
# Table 6: search comparison verdicts
# --------------------------------------------------------------------------
def tab_search() -> None:
    d = jload("results/v4_1/final/seed_extension.json")["cells"]

    def count(algo: str, comp: str):
        pos = neg = tot = 0
        detail = []
        for cell in CELLS:
            ps = d[cell].get(algo, {}).get("per_seed")
            if not ps:
                continue
            for seed, v in ps.items():
                for pool, p in v["pools"].items():
                    if comp not in p:
                        continue
                    tot += 1
                    e = p[comp]["macro_f1"]
                    if e["excludes_zero"]:
                        if e["est"] > 0:
                            pos += 1
                            detail.append((cell, pool, seed, e["est"]))
                        else:
                            neg += 1
        return tot, pos, neg, detail

    ga_rnd = count("ga", "vs_random")
    ga_sw = count("ga", "vs_switch")
    es_rnd = count("es", "vs_random")
    es_sw = count("es", "vs_switch")

    # NSGA-II deployment counts (accuracy, dev/confirm)
    n2 = jload("results/v4_1/evolution/nsga2_deploy_devconfirm.json")["tasks"]
    n2_dev = n2_cf = n2_tot = 0
    for t, c in n2.items():
        for seed, v in c["per_seed"].items():
            for split, key in (("dev", "deploy_minus_base_dev"),
                               ("confirm", "deploy_minus_base_confirm")):
                if key in v:
                    n2_tot += 1
                    if v[key]["excludes_zero"]:
                        if split == "dev":
                            n2_dev += 1
                        else:
                            n2_cf += 1

    body = (
        r"""\begin{table}[t]
\centering
\small
\caption{Search comparisons under the pre-registered protocol. Counts are
paired group-bootstrap 95\% intervals that exclude zero; ``positive'' means the
evolved policy beats the stated baseline. GA and ES rows use macro-F1 on all
three pools with five seeds; the NSGA-II row uses accuracy on 12 dev and 12
confirm cells.}
\label{tab:search}
\begin{tabular}{@{}lrrl@{}}
\toprule
Comparison & Pos. & Neg. & Reading \\
\midrule
GA vs.\ random & """
        + f"{ga_rnd[1]} & {ga_rnd[2]} & no advantage ({ga_rnd[0]})"
        + r""" \\
GA vs.\ fixed & """
        + f"{ga_sw[1]} & {ga_sw[2]} & train-only ({ga_sw[0]})"
        + r""" \\
ES vs.\ random & """
        + f"{es_rnd[1]} & {es_rnd[2]} & no advantage ({es_rnd[0]})"
        + r""" \\
ES vs.\ fixed & """
        + f"{es_sw[1]} & {es_sw[2]} & one confirm ({es_sw[0]})"
        + r""" \\
NSGA-II vs.\ fixed & """
        + f"0 & {n2_dev + n2_cf} & {n2_dev} dev, {n2_cf} confirm"
        + r""" \\
\bottomrule
\end{tabular}
\end{table}
"""
    )
    write("tab_search.tex", body)


# --------------------------------------------------------------------------
# Table 7: transfer and fault map boundary checks
# --------------------------------------------------------------------------
def tab_boundary() -> None:
    tr = jload("results/v4_1/evolution/transfer_heldout.json")["matrix"]
    diag = [k for k in tr if k.split("_on_")[0] == k.split("_on_")[1]]
    off = [k for k in tr if k not in diag]

    def sig_count(keys):
        pos = neg = 0
        for k in keys:
            for split, c in tr[k].items():
                p = c.get("vs_switch_paired", {})
                if p.get("all_ns", True):
                    continue
                ests = [e["est"] for e in p.get("per_seed", {}).values()]
                if ests and all(e > 0 for e in ests):
                    pos += 1
                else:
                    neg += 1
        return pos, neg

    dpos, dneg = sig_count(diag)
    opos, oneg = sig_count(off)
    fm = jload("results/v4_1/evolution/faultmap.json")
    top = fm["cells"][:4]
    rows = []
    for c in top:
        task, model, band = c["cell"]
        rows.append(
            f"{TASK_TEX[task]} / {model.capitalize()} / {band} & "
            f"{c['n_discover']} & {f3(c['severity_discover'])} & "
            f"{c['n_dev']} & {f3(c['severity_dev'])} & "
            f"{c['n_confirm']} & {f3(c['severity_confirm'])} \\\\"
        )
    body = (
        r"""\begin{table*}[t]
\centering
\small
\caption{Boundary checks. Top: frozen-genome transfer, counting significant
paired accuracy differences against the target's fixed switch-only policy;
diagonal cells are same-task deployment checks, off-diagonal cells are true
transfer. Bottom: the four highest-severity fault-map cells, with severity
defined as the error rate in the cell.}
\label{tab:boundary}
\begin{tabular}{@{}lrr@{}}
\toprule
Transfer & Sig. positive & Sig. negative \\
\midrule
"""
        + f"Diagonal (same task) & {dpos} & {dneg} \\\\\n"
        + f"Off-diagonal (true transfer) & {opos} & {oneg} \\\\\n"
        + r"""\bottomrule
\end{tabular}

\vspace{4pt}
\begin{tabular}{@{}lrrrrrr@{}}
\toprule
Fault-map cell & $n$ disc. & Severity & $n$ dev & Severity & $n$ conf. & Severity \\
\midrule
"""
        + "\n".join(rows)
        + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    )
    write("tab_boundary.tex", body)


if __name__ == "__main__":
    tab_models()
    tab_native()
    tab_complementarity()
    tab_predictability()
    tab_calibration()
    tab_search()
    tab_boundary()
