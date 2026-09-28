"""Figure 4 — Cross-FM benchmark per basin.

Four bars per basin (A baseline; TerraMind+DEM; TerraMind+DEM+S2L2A;
Prithvi-EO-2.0+S2), two panels (ROC, PR), with Δ-AUC annotations and
significance markers.
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from paths import RESULTS
from paths import FIGURES_DIR as OUT_DIR

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Liberation Serif"],
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "axes.linewidth": 0.7, "lines.linewidth": 1.2, "patch.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
})

CM = 1 / 2.54
FULL_W = 17.4 * CM

BASINS = [("06_rio_huasco", "Huasco", "Semi-arid"),
          ("09_rio_maipo",  "Maipo",  "Mediterranean"),
          ("11_rio_maule",  "Maule",  "Temperate-humid")]

COLOR_A   = "#888888"   # baseline
COLOR_TMD = "#1b6f9e"   # TerraMind DEM
COLOR_TMM = "#0d3d5c"   # TerraMind multimodal
COLOR_PR  = "#bb5500"   # Prithvi


def load(b, model, mod=""):
    if model == "terramind":
        suf = "" if mod == "dem" else "_dem+s2l2a"
        return json.loads((RESULTS / f"{b}_terramind_linprobe_spatial{suf}.json").read_text())
    return json.loads((RESULTS / f"{b}_prithvi-300m_linprobe_spatial.json").read_text())


def save_fig(fig, name):
    pdf = OUT_DIR / f"{name}.pdf"
    png = OUT_DIR / f"{name}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    print(f"  wrote {pdf.name} + {png.name}")



T_975_DF4 = 2.776  # two-sided 95% t quantile, df = 4


def fold_tci(d, key="delta_roc"):
    """Fold-level paired 95% t-interval (df=4) from per-fold deltas."""
    import math
    vals = [f[key] for f in d["fold_results"]]
    n = len(vals)
    m = sum(vals) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1))
    h = T_975_DF4 * sd / math.sqrt(n)
    return m, m - h, m + h


def sig_marker(d, key):
    _, lo, hi = fold_tci(d, key=f"delta_{key}")
    return "*" if not (lo <= 0 <= hi) else "n.s."


def main():
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 8.5 * CM), constrained_layout=True)

    panels = [("roc", "AUC ROC", "(a) AUC ROC"),
              ("pr",  "AUC PR",  "(b) AUC PR")]
    for j, (key, ylabel, title) in enumerate(panels):
        ax = axes[j]
        x = np.arange(len(BASINS))
        a, tmd, tmm, pr = [], [], [], []
        a_s, tmd_s, tmm_s, pr_s = [], [], [], []
        for b_slug, _, _ in BASINS:
            d_dem = load(b_slug, "terramind", "dem")
            d_mm  = load(b_slug, "terramind", "mm")
            d_pr  = load(b_slug, "prithvi")
            a.append(d_dem[f"A_{key}_mean"]); a_s.append(d_dem[f"A_{key}_std"])
            tmd.append(d_dem[f"B_{key}_mean"]); tmd_s.append(d_dem[f"B_{key}_std"])
            tmm.append(d_mm[f"B_{key}_mean"]); tmm_s.append(d_mm[f"B_{key}_std"])
            pr.append(d_pr[f"B_{key}_mean"]); pr_s.append(d_pr[f"B_{key}_std"])

        w = 0.20
        offsets = [-1.5 * w, -0.5 * w, 0.5 * w, 1.5 * w]
        bars = [
            (a, a_s, COLOR_A, "A: 17 features"),
            (tmd, tmd_s, COLOR_TMD, "TerraMind + DEM"),
            (tmm, tmm_s, COLOR_TMM, "TerraMind + DEM + S2"),
            (pr, pr_s, COLOR_PR, "Prithvi-EO-2.0 + S2"),
        ]
        for off, (m, s, c, lab) in zip(offsets, bars):
            ax.bar(x + off, m, w, yerr=s, capsize=2.0,
                   color=c, edgecolor="black", linewidth=0.5, label=lab, zorder=2)

        for k, (b_slug, _, _) in enumerate(BASINS):
            d_dem = load(b_slug, "terramind", "dem")
            d_mm  = load(b_slug, "terramind", "mm")
            d_pr  = load(b_slug, "prithvi")
            sig_d = sig_marker(d_dem, key)
            sig_m = sig_marker(d_mm, key)
            sig_p = sig_marker(d_pr, key)
            # Anchored to the floor of the axes: with the axis correctly bounded
            # at AUC = 1, a top placement collides with the panel title wherever
            # the bars approach the ceiling.
            ax.text(
                x[k], 0.365,
                f"TM-D: {d_dem[f'delta_{key}_mean']:+.3f}$^{{{sig_d}}}$\n"
                f"TM-M: {d_mm[f'delta_{key}_mean']:+.3f}$^{{{sig_m}}}$\n"
                f"Pr-S: {d_pr[f'delta_{key}_mean']:+.3f}$^{{{sig_p}}}$",
                ha="center", va="bottom", fontsize=6.0,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor="none", alpha=0.85), zorder=4,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([f"{nm}\n({rg})" for _, nm, rg in BASINS], fontsize=6.5)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylim(0.35, 1.0)   # AUC is bounded at 1; annotations go inside the axes
        ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6, zorder=0)
        if j == 0:
            pass  # legend is drawn once at figure level

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               fontsize=7, bbox_to_anchor=(0.5, 1.04))

    save_fig(fig, "fig4_benchmark_per_basin")
    plt.close(fig)


if __name__ == "__main__":
    main()
