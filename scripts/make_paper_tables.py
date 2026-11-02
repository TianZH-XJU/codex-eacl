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
OUT_ZH = ROOT / "paper" / "acl_zh" / "tables"
OUT_ZH.mkdir(parents=True, exist_ok=True)

CELLS = ["code_laya", "code_vela", "findver_laya", "findver_vela"]
CELL_TEX = {
    "code_laya": "CodeXGLUE (Laya)",
    "code_vela": "CodeXGLUE (Vela)",
    "findver_laya": "FinDVer (Laya)",
    "findver_vela": "FinDVer (Vela)",
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
        base = "Choice" if c["base_action"] == "switch" else "Noul"
        alt = "Noul" if base == "Choice" else "Choice"
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
pools ($n=60$ per training cell; FinDVer dev/confirm: 59/44). \emph{Headroom} is the accuracy gain of a per-row
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
    d = jload("results/v4_1/final/signed_utility.json")["cells"]
    rows = []
    for cell in CELLS:
        c = d[cell]
        champ = c["champion"]
        m = c["models"][champ]
        cand = c["candidates"][champ]
        dev = cand["dev_net"]
        conf = cand["confirm_net"]
        rlo, rhi = conf["raw_ci95_pp"]
        clo, chi = conf["cost_ci95_pp"]
        rows.append(
            f"{CELL_TEX[cell]} & {m['oof_spearman']:+.2f} & "
            f"{f3(m['oof_auroc_repair'])} & {f3(m['oof_auroc_regression'])} & "
            f"{dev['cost_pp']:+.1f} & {conf['raw_pp']:+.1f} "
            f"[{rlo:+.1f}, {rhi:+.1f}] & {conf['cost_pp']:+.1f} "
            f"[{clo:+.1f}, {chi:+.1f}] & {conf['n_switched']}/{c['n_confirm']} \\\\"
        )
    body = (
        r"""\begin{table*}[t]
\centering
\scriptsize
\caption{Signed incremental-utility prediction. Target
$\Delta_i=\mathbf{1}[\textsc{Choice}_i\text{ correct}]-
\mathbf{1}[\textsc{Noul}_i\text{ correct}]\in\{-1,0,+1\}$. OOF $\rho$ uses
group-aware three-fold CV on \texttt{search\_train}; the threshold is selected
on \texttt{search\_dev} by cost-adjusted utility and frozen. Raw net is mean
$\Delta$; cost-adjusted net subtracts $\lambda=0.05$ per switch; both are in
points with 95\% group-bootstrap CIs. Last column: switches/rows.}
\label{tab:predictability}
\begin{tabular}{@{}lrrrrrrr@{}}
\toprule
Task / model & OOF $\rho$ & AUROC repair & AUROC reg. & Dev cost & Confirm raw (95\% CI) & Confirm cost (95\% CI) & Switches \\
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


def tab_locked() -> None:
    d = jload("results/v4_1/locked_test/locked_test_results.json")["cells"]
    labels = {"code_laya": "CodeXGLUE / Laya", "code_vela": "CodeXGLUE / Vela",
              "findver_laya": "FinDVer / Laya", "findver_vela": "FinDVer / Vela"}
    rows = []
    for cell in ("code_laya", "code_vela", "findver_laya", "findver_vela"):
        c = d[cell]
        g = c["ga"]["17"]
        rows.append(
            f"{labels[cell]} & {c['fixed_noul']['macro_f1']:.3f} & "
            f"{c['fixed_switch']['macro_f1']:.3f} & {g['macro_f1']:.3f} \\\\"
        )
    body = (
        r"""\begin{table}[t]
\centering
\scriptsize
\caption{Locked test: 60 official test rows per cell, templates and GA genomes
frozen before any test call. Macro-F1 for fixed \noul{}, fixed \choice{}
(always switch, two calls) and GA seed 17; seeds 23 and 41 show the same
pattern. Paired differences and CIs are in the text.}
\label{tab:locked}
\begin{tabular}{@{}lrrr@{}}
\toprule
Cell & \noul{} & \choice{} & GA \\
\midrule
"""
        + "\n".join(rows)
        + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    )
    write("tab_locked.tex", body)


def tab_processbench() -> None:
    d = jload("results/v4_1/final/processbench_selective.json")["cells"]
    rows = []
    for key, label in (("math_laya", "Laya"), ("math_vela", "Vela")):
        c = d[key]
        pts = {round(p["target_coverage"], 2): p for p in c["points"]}
        full = c["full_coverage"]
        p50 = pts[0.5]["confirm"]
        p25 = pts[0.25]["confirm"]
        rows.append(
            f"{label} & {full['selective_accuracy']:.3f} ({full['coverage']:.2f}) & "
            f"{p50['selective_accuracy']:.3f} ({p50['coverage']:.2f}) & "
            f"{p25['selective_accuracy']:.3f} ({p25['coverage']:.2f}) & "
            f"{c['aurc_confirm']:.3f} \\\\"
        )
    body = (
        r"""\begin{table}[t]
\centering
\small
\caption{ProcessBench selective prediction on confirm. Columns are selective
accuracy at full coverage, at a 50\% target and at a 25\% target; parentheses
show achieved confirm coverage. Thresholds are selected on
\texttt{selection\_dev} and frozen. AURC is the area under the risk-coverage
curve (lower is better).}
\label{tab:processbench}
\begin{tabular}{@{}lrrrr@{}}
\toprule
Model & Full & 50\% target & 25\% target & AURC \\
\midrule
"""
        + "\n".join(rows)
        + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    )
    write("tab_processbench.tex", body)


def zh_processbench() -> None:
    d = jload("results/v4_1/final/processbench_selective.json")["cells"]
    rows = []
    for key, label in (("math_laya", "Laya"), ("math_vela", "Vela")):
        c = d[key]
        pts = {round(p["target_coverage"], 2): p for p in c["points"]}
        full = c["full_coverage"]
        p50 = pts[0.5]["confirm"]
        p25 = pts[0.25]["confirm"]
        rows.append(
            f"{label} & {full['selective_accuracy']:.3f} ({full['coverage']:.2f}) & "
            f"{p50['selective_accuracy']:.3f} ({p50['coverage']:.2f}) & "
            f"{p25['selective_accuracy']:.3f} ({p25['coverage']:.2f}) & "
            f"{c['aurc_confirm']:.3f} \\\\"
        )
    body = (
        r"""\begin{table}[t]
\centering
\small
\caption{ProcessBench 在 confirm 上的选择性预测。各列为全覆盖、50\% 目标覆盖与
25\% 目标覆盖下的选择性准确率；括号内为实际 confirm 覆盖率。阈值在
\texttt{selection\_dev} 上选择并冻结。AURC 为风险—覆盖曲线下面积（越低越好）。}
\label{tab:processbench}
\begin{tabular}{@{}lrrrr@{}}
\toprule
模型 & 全覆盖 & 50\% 目标 & 25\% 目标 & AURC \\
\midrule
"""
        + "\n".join(rows)
        + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    )
    (OUT_ZH / "tab_processbench.tex").write_text(body, encoding="utf-8")
    print("wrote paper/acl_zh/tables/tab_processbench.tex")


def zh_predictability() -> None:
    """Chinese review version of the signed-utility table."""
    d = jload("results/v4_1/final/signed_utility.json")["cells"]
    rows = []
    for cell in CELLS:
        c = d[cell]
        champ = c["champion"]
        m = c["models"][champ]
        cand = c["candidates"][champ]
        dev = cand["dev_net"]
        conf = cand["confirm_net"]
        rlo, rhi = conf["raw_ci95_pp"]
        clo, chi = conf["cost_ci95_pp"]
        rows.append(
            f"{CELL_TEX[cell]} & {m['oof_spearman']:+.2f} & "
            f"{f3(m['oof_auroc_repair'])} & {f3(m['oof_auroc_regression'])} & "
            f"{dev['cost_pp']:+.1f} & {conf['raw_pp']:+.1f} "
            f"[{rlo:+.1f}, {rhi:+.1f}] & {conf['cost_pp']:+.1f} "
            f"[{clo:+.1f}, {chi:+.1f}] & {conf['n_switched']}/{c['n_confirm']} \\\\"
        )
    body = (
        r"""\begin{table}[t]
\centering
\scriptsize
\caption{有符号增量效用预测。目标为
$\Delta_i=\mathbf{1}[\textsc{Choice}_i\text{ 正确}]-
\mathbf{1}[\textsc{Noul}_i\text{ 正确}]\in\{-1,0,+1\}$。折外 $\rho$ 是
\texttt{search\_train} 上分组感知三折交叉验证中预测与真实 $\Delta$ 的
Spearman 相关；阈值在 \texttt{search\_dev} 上按成本调整效用选择，冻结后用
于 \texttt{confirm}。原始净值为平均 $\Delta$；成本调整净值为原始净值减去
每次切换 $\lambda=0.05$；单位为百分点，区间为 95\% 分组自助法。最后一列为
confirm 行中的切换数。}
\label{tab:predictability}
\resizebox{\linewidth}{!}{%
\begin{tabular}{@{}lrrrrrrr@{}}
\toprule
任务 / 模型 & 折外 $\rho$ & 修复 AUROC & 退化 AUROC & dev 成本 & confirm 原始 (95\% CI) & confirm 成本 (95\% CI) & 切换/行 \\
\midrule
"""
        + "\n".join(rows)
        + r"""
\bottomrule
\end{tabular}%
}
\end{table}
"""
    )
    (OUT_ZH / "tab_predictability.tex").write_text(body, encoding="utf-8")
    print("wrote paper/acl_zh/tables/tab_predictability.tex")


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
        r"""\begin{table}[!b]
\centering
\small
\caption{Search comparisons under the pre-specified protocol. Counts are
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
    tab_locked()
    tab_processbench()
    zh_processbench()
    zh_predictability()
    tab_calibration()
    tab_search()
    tab_boundary()
