#!/usr/bin/env python3
"""Publication figures for the paper (deterministic, Agg backend, vector PDF).

Every plotted number is read from a committed JSON produced by an analysis
script; the two exceptions (repair profile, reliability curves) are recomputed
from the same committed caches because they are curves, not scalars, and the
underlying rows are the source of truth either way.

Outputs (paper/acl/figures/):
  fig_headroom.pdf        repair/harmful/headroom on train + held-out net repair
  fig_repair_profile.pdf  repair rate vs Noul probability bin, per pool
  fig_calibration.pdf     reliability diagrams, raw vs temperature-scaled
  fig_predictability.pdf  OOF AUROC vs frozen confirm net utility
  fig_search.pdf          neutral mutations, train-dev rank, best-so-far traces
  fig_faultmap.pdf        severity heat map over task x model x confidence band
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "acl" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
RED = "#D55E00"
PURPLE = "#CC79A7"
GREY = "#8C8C8C"

CELLS = ["code_laya", "code_vela", "findver_laya", "findver_vela"]
LABEL = {"code_laya": "Code/Laya", "code_vela": "Code/Vela",
         "findver_laya": "FinDVer/Laya", "findver_vela": "FinDVer/Vela"}
SHORT = {"code_laya": "C-L", "code_vela": "C-V", "findver_laya": "F-L", "findver_vela": "F-V"}


def jload(rel):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def rows(rel):
    return jload(rel)["rows"]


CACHES = {
    "code_laya": ("results/v4/evolution/cache_train_code_laya.json",
                  "results/v4/evolution/cache_code_laya.json",
                  "results/v4/evolution/cache_confirm_code_laya.json"),
    "code_vela": ("results/v4/evolution/cache_train_code_vela.json",
                  "results/v4/evolution/cache_code_vela.json",
                  "results/v4/evolution/cache_confirm_code_vela.json"),
    "findver_laya": ("results/v4/evolution/cache_train_findver_laya.json",
                     "results/v4/evolution/cache_findver_laya.json",
                     "results/v4/evolution/cache_confirm_findver_laya.json"),
    "findver_vela": ("results/v4/evolution/cache_train_findver_vela.json",
                     "results/v4/evolution/cache_findver_vela.json",
                     "results/v4/evolution/cache_confirm_findver_vela.json"),
}


def repair_profile(rs, bins=5):
    p = np.array([float(r["noul_p"]) for r in rs])
    y = []
    for r in rs:
        noul_ok = (float(r["noul_p"]) >= 0.5) == bool(r["gold"])
        y.append(0.0 if noul_ok else (1.0 if bool(r.get("switch_correct", False)) else 0.0))
    y = np.array(y)
    edges = np.linspace(0, 1, bins + 1)
    rate, cnt, centers = [], [], []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p < hi) if i < bins - 1 else (p >= lo) & (p <= hi)
        cnt.append(int(m.sum()))
        rate.append(float(y[m].mean()) if m.sum() else np.nan)
        centers.append((lo + hi) / 2)
    return np.array(centers), np.array(rate), np.array(cnt)


def fig_headroom():
    comp = jload("results/v4_1/evolution/action_complementarity.json")["cells"]
    held = jload("results/v4_1/evolution/complementarity_heldout.json")["cells"]
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.25), gridspec_kw={"width_ratios": [1.15, 1]})

    ax = axes[0]
    x = np.arange(len(CELLS))
    rep = [comp[c]["repairable"] for c in CELLS]
    harm = [comp[c]["harmful"] for c in CELLS]
    ax.bar(x - 0.19, rep, width=0.36, color=BLUE, label="repair")
    ax.bar(x + 0.19, harm, width=0.36, color=ORANGE, label="harmful switch")
    for i, c in enumerate(CELLS):
        ax.text(i, max(rep[i], harm[i]) + 0.6, f"net {comp[c]['net_repair']:+d}",
                ha="center", va="bottom", fontsize=6.5, color=GREY)
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[c] for c in CELLS], rotation=18, ha="right")
    ax.set_ylabel("rows (n=60)")
    ax.set_ylim(0, max(harm) + 3.0)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("(a) search pool: gross repairs are offset", loc="left")

    ax = axes[1]
    dev = [held[c]["dev"]["net_pp"] for c in CELLS]
    dev_lo = [held[c]["dev"]["net_pp"] - held[c]["dev"]["net_ci95"][0] for c in CELLS]
    dev_hi = [held[c]["dev"]["net_ci95"][1] - held[c]["dev"]["net_pp"] for c in CELLS]
    conf = [held[c]["confirm"]["net_pp"] for c in CELLS]
    conf_lo = [held[c]["confirm"]["net_pp"] - held[c]["confirm"]["net_ci95"][0] for c in CELLS]
    conf_hi = [held[c]["confirm"]["net_ci95"][1] - held[c]["confirm"]["net_pp"] for c in CELLS]
    ax.errorbar(x - 0.08, dev, yerr=[dev_lo, dev_hi], fmt="o", color=GREEN, ms=4,
                capsize=2, lw=1, label="dev (selection)")
    ax.errorbar(x + 0.08, conf, yerr=[conf_lo, conf_hi], fmt="s", color=RED, ms=4,
                capsize=2, lw=1, label="confirm (held out)")
    ax.axhline(0, color="black", lw=0.6, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[c] for c in CELLS], rotation=18, ha="right")
    ax.set_ylabel("net repair rate (95% CI)")
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("(b) held-out: sign is pool-dependent", loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_headroom.pdf")
    plt.close(fig)


def fig_repair_profile():
    fig, axes = plt.subplots(2, 2, figsize=(6.9, 3.5), sharex=True, sharey=True)
    colors = {"train": BLUE, "dev": GREEN, "confirm": RED}
    for ax, c in zip(axes.ravel(), CELLS):
        for tag, rel in zip(("train", "dev", "confirm"), CACHES[c]):
            cx, rate, cnt = repair_profile(rows(rel))
            ax.plot(cx, rate, "-o", color=colors[tag], ms=3, lw=1, label=tag)
            for xi, ri, ni in zip(cx, rate, cnt):
                if ni >= 8 and not np.isnan(ri):
                    ax.annotate(f"{ni}", (xi, ri), textcoords="offset points",
                                xytext=(0, 4), ha="center", fontsize=5.5, color=colors[tag])
        ax.set_title(LABEL[c], loc="left", fontsize=7.5)
        ax.set_ylim(-0.03, 0.80)
        ax.grid(axis="y", lw=0.3, color="#DDDDDD")
    for ax in axes[1]:
        ax.set_xlabel("Noul probability bin")
    for ax in axes[:, 0]:
        ax.set_ylabel("repair rate")
    axes[0, 0].legend(frameon=False, loc="upper left", ncol=1)
    fig.tight_layout()
    fig.savefig(OUT / "fig_repair_profile.pdf")
    plt.close(fig)


def reliability(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    xs, ys, ns = [], [], []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p < hi) if i < bins - 1 else (p >= lo) & (p <= hi)
        if m.sum() == 0:
            continue
        xs.append(float(p[m].mean()))
        ys.append(float(y[m].mean()))
        ns.append(int(m.sum()))
    return np.array(xs), np.array(ys), np.array(ns)


def fig_calibration():
    cal = jload("results/v4_1/final/calibration.json")["cells"]
    fig, axes = plt.subplots(1, 4, figsize=(6.9, 1.95), sharey=True)
    for ax, c in zip(axes, CELLS):
        rs = rows(CACHES[c][2])
        p = np.array([float(r["noul_p"]) for r in rs])
        y = np.array([1.0 if r["gold"] else 0.0 for r in rs])
        T = cal[c]["temperature"]["T"]
        pc = 1.0 / (1.0 + np.exp(-np.log(np.clip(p, 1e-6, 1 - 1e-6) /
                                         (1 - np.clip(p, 1e-6, 1 - 1e-6))) / T))
        for arr, col, lab in ((p, GREY, "raw"), (pc, BLUE, f"T={T:.1f}")):
            xs, ys, _ = reliability(arr, y)
            ax.plot(xs, ys, "-o", color=col, ms=2.5, lw=1, label=lab)
        ax.plot([0, 1], [0, 1], color="black", lw=0.5, ls=":")
        ax.set_title(LABEL[c], fontsize=7)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(frameon=False, loc="lower right", fontsize=6)
        ax.set_xlabel("predicted P(true)")
    axes[0].set_ylabel("observed frequency")
    fig.tight_layout()
    fig.savefig(OUT / "fig_calibration.pdf")
    plt.close(fig)


def fig_predictability():
    up = jload("results/v4_1/evolution/utility_predictability.json")["cells"]
    fig, ax = plt.subplots(figsize=(3.35, 2.45))
    offsets = {"code_laya": (4, 4), "code_vela": (4, -9),
               "findver_laya": (-8, 6), "findver_vela": (-10, -10)}
    for c in CELLS:
        lr = up[c]["models"]["logreg"]["oof_auroc"]
        tr = up[c]["models"]["tree"]["oof_auroc"]
        ch = up[c]["champion"]
        conf = up[c]["candidates"][ch]["confirm_net"]["net"]
        ax.plot([lr, tr], [conf, conf], color=GREY, lw=0.8, zorder=1)
        ax.scatter([lr], [conf], s=24, color=BLUE, zorder=3,
                   label="LogReg" if c == CELLS[0] else None)
        ax.scatter([tr], [conf], s=24, color=ORANGE, marker="s", zorder=3,
                   label="tree" if c == CELLS[0] else None)
        ax.annotate(LABEL[c], (max(lr, tr), conf), textcoords="offset points",
                    xytext=offsets[c], fontsize=6.5, ha="right" if offsets[c][0] < 0 else "left")
    ax.axhline(0, color="black", lw=0.6, ls="--")
    ax.axvline(0.5, color="black", lw=0.4, ls=":")
    ax.set_xlim(0.40, 1.0)
    ax.set_xlabel("OOF AUROC on the search pool")
    ax.set_ylabel("frozen confirm net utility")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_predictability.pdf")
    plt.close(fig)


def fig_search():
    dyn = jload("results/v4_1/final/search_dynamics.json")["cells"]
    land = jload("results/v4_1/evolution/search_landscape.json")["families"]
    fig = plt.figure(figsize=(6.9, 2.15))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.05, 1.15], wspace=0.55)

    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(CELLS))
    neu = [dyn[c]["neutral_mean"] for c in CELLS]
    ax.bar(x, neu, color=PURPLE, width=0.6)
    for i, v in enumerate(neu):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[c] for c in CELLS])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("neutral mutation rate")
    ax.set_title("(a) neutral mutations", loc="left")

    ax = fig.add_subplot(gs[0, 1])
    sp = [dyn[c]["spearman_mean"] for c in CELLS]
    ax.bar(x, sp, color=[BLUE if v >= 0 else RED for v in sp], width=0.6)
    ax.axhline(0, color="black", lw=0.6)
    for i, v in enumerate(sp):
        ax.text(i, v + (0.03 if v >= 0 else -0.10), f"{v:+.2f}", ha="center", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[c] for c in CELLS])
    ax.set_ylim(-0.8, 1.15)
    ax.set_ylabel("train-dev Spearman")
    ax.set_title("(b) train rank vs dev", loc="left")

    ax = fig.add_subplot(gs[0, 2])
    for i, c in enumerate(CELLS):
        fam = land[f"evo_train_{c}"]["per_seed"]
        series = fam["17"]["ga"]["best_so_far"]["series"]
        ax.step(range(len(series)), series, where="post", lw=1.2,
                ls=["-", "--", ":", "-."][i],
                color=[BLUE, ORANGE, GREEN, RED][i], label=LABEL[c])
    ax.set_xlabel("generation")
    ax.set_ylabel("best train scalar")
    ax.set_ylim(0.36, 0.62)
    ax.legend(frameon=False, loc="lower right", fontsize=6)
    ax.set_title("(c) best-so-far (seed 17)", loc="left")
    fig.savefig(OUT / "fig_search.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_faultmap():
    fm = jload("results/v4_1/evolution/faultmap.json")["cells"]
    tasks = ["code", "findver", "math"]
    models = ["laya", "vela"]
    bands = ["low", "mid", "high"]
    val = np.full((len(tasks) * len(models), len(bands)), np.nan)
    ann = [["" for _ in bands] for _ in range(len(tasks) * len(models))]
    rowlab = []
    for ti, t in enumerate(tasks):
        for mi, m in enumerate(models):
            rowlab.append(f"{t}/{m}")
            for bi, b in enumerate(bands):
                for cell in fm:
                    if cell["cell"] == [t, m, b]:
                        v = cell["severity_confirm"]
                        star = "" if cell.get("confirmed_confirm") else "*"
                        val[ti * len(models) + mi, bi] = v
                        ann[ti * len(models) + mi][bi] = f"{v:.2f}{star}\n(n={cell['n_confirm']})"
    fig, ax = plt.subplots(figsize=(3.2, 2.5))
    im = ax.imshow(val, cmap="RdYlBu_r", vmin=0, vmax=1, aspect="auto")
    for i in range(val.shape[0]):
        for j in range(val.shape[1]):
            if not np.isnan(val[i, j]):
                ax.text(j, i, ann[i][j], ha="center", va="center", fontsize=6,
                        color="white" if val[i, j] > 0.55 else "black")
    ax.set_xticks(range(len(bands)))
    ax.set_xticklabels(["low conf.", "mid", "high conf."])
    ax.set_yticks(range(len(rowlab)))
    ax.set_yticklabels(rowlab)
    ax.set_xlabel("confidence band")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("failure severity", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig(OUT / "fig_faultmap.pdf")
    plt.close(fig)


def main():
    fig_headroom()
    fig_repair_profile()
    fig_calibration()
    fig_predictability()
    fig_search()
    fig_faultmap()
    for p in sorted(OUT.glob("fig_*.pdf")):
        print(p.name, p.stat().st_size)


if __name__ == "__main__":
    main()
