import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import plot_mechanism_evidence as PLT  # noqa: E402


def _render_all(d):
    import matplotlib
    assert matplotlib.get_backend().lower() == "agg"
    import matplotlib.pyplot as plt
    d.mkdir(parents=True, exist_ok=True)
    figs = {}
    for fn in (PLT.fig_headroom, PLT.fig_predictability, PLT.fig_faultmap):
        fig, name = fn()
        p = d / name
        fig.savefig(p, dpi=150)
        plt.close(fig)
        figs[name] = p
    return figs


def test_figures_regenerate_deterministically(tmp_path):
    ha = {k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in _render_all(tmp_path / "a").items()}
    hb = {k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in _render_all(tmp_path / "b").items()}
    assert ha == hb


def test_figure_content_sanity():
    import matplotlib.pyplot as plt
    # headroom: 4 bars + 4 scatter markers, titled
    fig, _ = PLT.fig_headroom()
    assert len(fig.axes[0].patches) == 4
    assert "EXP-01" in fig.axes[0].get_title()
    plt.close(fig)
    # predictability: twin axes (AUROC + net), 4+4 bars
    fig, _ = PLT.fig_predictability()
    assert len(fig.axes) == 2
    assert sum(len(ax.patches) for ax in fig.axes) == 8
    assert "EXP-02" in fig.axes[0].get_title()
    plt.close(fig)
    # faultmap: 5 cells x 3 splits = 15 bars, own figure (not overlaid)
    fig, _ = PLT.fig_faultmap()
    assert len(fig.axes) == 1 and len(fig.axes[0].patches) == 15
    assert "EXP-04" in fig.axes[0].get_title()
    plt.close(fig)


def test_committed_figures_exist():
    for name in ("headroom.png", "predictability.png", "faultmap.png"):
        p = ROOT / "paper" / "figures" / name
        assert p.exists() and p.stat().st_size > 5000
