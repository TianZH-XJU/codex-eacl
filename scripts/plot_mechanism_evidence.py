#!/usr/bin/env python3
"""Publication figures for the mechanism evidence (offline, deterministic).

Reads committed v4_1 JSONs only; writes paper/figures/*.png with the Agg
backend. No resampling happens here (bootstrap draws live in the JSONs), so
output is byte-identical across runs given identical inputs. Every number
plotted also appears in the paper text or the evidence-grade table — figures
illustrate, never introduce, numbers.

Cell selection (explicit): EXP-01 plots all 4 binary cells; EXP-02 plots all
4 binary cells with champion-matched OOF AUROC; EXP-04 plots the 4 hardest
confirm-severity math cells plus code/vela/mid as the non-confirming contrast
(it is high on discover then collapses — the counter-example to "stable").
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from pathlib import Path  # noqa: E402
import json  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "paper" / "figures"

FAULTMAP_CELLS = [("math", "vela", "low"), ("math", "laya", "high"),
                  ("math", "laya", "mid"), ("math", "vela", "mid"),
                  ("code", "vela", "mid")]


def load(name):
    return json.loads((ROOT / "results/v4_1/evolution" / name).read_text(encoding="utf-8"))


def fig_headroom():
    d = load("action_complementarity.json")["cells"]
    keys = ["code_laya", "code_vela", "findver_laya", "findver_vela"]
    x = np.arange(len(keys))
    head = [d[k]["headroom"] for k in keys]
    lo = [d[k]["headroom_ci"]["ci95"][0] for k in keys]
    hi = [d[k]["headroom_ci"]["ci95"][1] for k in keys]
    net = [(d[k]["repairable"] - d[k]["harmful"]) / d[k]["n"] for k in keys]
    assert len({d[k]["n"] for k in keys}) == 1  # shared pool size, else label per-bar
    n_pool = d[keys[0]]["n"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(x, head, yerr=[np.array(head) - np.array(lo), np.array(hi) - np.array(head)],
           capsize=4, label="oracle headroom + group 95% CI")
    ax.scatter(x, net, color="red", zorder=5, label="net repair/n (point)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, [k.replace("_", "\n") for k in keys])
    ax.set_ylabel(f"fraction of pool (n={n_pool})")
    ax.set_title("EXP-01: oracle headroom vs net repair (best-fixed baseline)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig, "headroom.png"


def fig_predictability():
    d = load("utility_predictability.json")["cells"]
    keys = ["code_laya", "code_vela", "findver_laya", "findver_vela"]
    x = np.arange(len(keys))
    # champion-matched OOF AUROC (same model as the utility bar)
    oof = []
    for k in keys:
        champ = d[k]["champion"]
        oof.append(d[k]["models"][champ]["oof_auroc"])
    conf_net = [d[k]["candidates"][d[k]["champion"]]["confirm_net"]["net"] for k in keys]
    fig, ax1 = plt.subplots(figsize=(7, 3.6))
    w = 0.35
    ax1.bar(x - w / 2, oof, w, label="champion OOF AUROC (ranking)", color="steelblue")
    ax1.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("OOF AUROC (chance = 0.5)")
    ax2 = ax1.twinx()
    ax2.bar(x + w / 2, conf_net, w, label="frozen confirm net utility", color="darkorange")
    ax2.axhline(0.0, color="black", linewidth=0.8)
    ax2.set_ylabel("confirm net accuracy gain")
    ax1.set_xticks(x, [k.replace("_", "\n") for k in keys])
    ax1.set_title("EXP-02: ranking signal does not transfer to utility")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    fig.tight_layout()
    return fig, "predictability.png"


def fig_faultmap():
    d = {tuple(c["cell"]): c for c in load("group_faultmap.json")["cells"]}
    cells = [d[t] for t in FAULTMAP_CELLS]
    labels = ["/".join(c["cell"]) + (" *" if c["cell"] == ["code", "vela", "mid"] else "")
              for c in cells]
    x = np.arange(len(cells))
    w = 0.24
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    for i, split in enumerate(("discover", "dev", "confirm")):
        est = [c[split]["est"] for c in cells]
        lo = [c[split]["ci95"][0] for c in cells]
        hi = [c[split]["ci95"][1] for c in cells]
        ax.bar(x + (i - 1) * w, est, w,
               yerr=[np.array(est) - np.array(lo), np.array(hi) - np.array(est)],
               capsize=3, label=split)
    ax.set_xticks(x, labels, rotation=12, fontsize=8)
    ax.set_ylabel("joint-failure severity + group 95% CI")
    ax.set_title("EXP-04: stable conditional failure (* = non-confirming contrast)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig, "faultmap.png"


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    names = []
    for fn in (fig_headroom, fig_predictability, fig_faultmap):
        fig, name = fn()
        fig.savefig(FIG / name, dpi=150)
        plt.close(fig)
        names.append(name)
    print("wrote", sorted(names))


if __name__ == "__main__":
    raise SystemExit(main())
