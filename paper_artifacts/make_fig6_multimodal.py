"""Figure 6 — Multimodal extension benchmark per basin.

Three bars per basin (A: 17 features, B: TerraMind+DEM, B: TerraMind+DEM+S2L2A),
in two panels (ROC, PR), with Δ-AUC annotations and significance markers.

Style consistent with figs 3-5 of the same paper.
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
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.7, "lines.linewidth": 1.2, "patch.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
})

CM = 1 / 2.54

T_975_DF4 = 2.776  # two-sided 95% t quantile, df = 4


def fold_tci_sig(d, key):
    import math
    vals = [f[key] for f in d["fold_results"]]
    n = len(vals)
    m = sum(vals) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1))
    h = T_975_DF4 * sd / math.sqrt(n)
    return "*" if not (m - h <= 0 <= m + h) else "n.s."
FULL_W = 17.4 * CM

BASINS = ["06_rio_huasco", "09_rio_maipo", "11_rio_maule"]
SHORT = {"06_rio_huasco": "Huasco", "09_rio_maipo": "Maipo", "11_rio_maule": "Maule"}
REGIME = {"06_rio_huasco": "Semi-arid", "09_rio_maipo": "Mediterranean",
          "11_rio_maule": "Temperate-humid"}

COLOR_A   = "#888888"
COLOR_B_D = "#1b6f9e"
COLOR_B_M = "#0d3d5c"


def save_fig(fig, name):
    pdf = OUT_DIR / f"{name}.pdf"
    png = OUT_DIR / f"{name}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    print(f"  wrote {pdf.name} + {png.name}")


def main():
    data_dem = {b: json.loads((RESULTS / f"{b}_terramind_linprobe_spatial.json").read_text())
                for b in BASINS}
    data_mm = {b: json.loads((RESULTS / f"{b}_terramind_linprobe_spatial_dem+s2l2a.json").read_text())
               for b in BASINS}

    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 8 * CM), constrained_layout=True)

    for j, (key, ylabel, title) in enumerate([
        ("roc", "AUC ROC", "(a) AUC ROC"),
        ("pr",  "AUC PR",  "(b) AUC PR"),
    ]):
        ax = axes[j]
        x = np.arange(len(BASINS))
        a_m = [data_dem[b][f"A_{key}_mean"] for b in BASINS]
        a_s = [data_dem[b][f"A_{key}_std"]  for b in BASINS]
        bd_m = [data_dem[b][f"B_{key}_mean"] for b in BASINS]
        bd_s = [data_dem[b][f"B_{key}_std"]  for b in BASINS]
        bm_m = [data_mm[b][f"B_{key}_mean"] for b in BASINS]
        bm_s = [data_mm[b][f"B_{key}_std"]  for b in BASINS]

        w = 0.26
        ax.bar(x - w, a_m,  w, yerr=a_s,  capsize=2.5,
               color=COLOR_A, edgecolor="black", linewidth=0.5,
               label="A: 17 features", zorder=2)
        ax.bar(x,     bd_m, w, yerr=bd_s, capsize=2.5,
               color=COLOR_B_D, edgecolor="black", linewidth=0.5,
               label="B: TerraMind + DEM", zorder=2)
        ax.bar(x + w, bm_m, w, yerr=bm_s, capsize=2.5,
               color=COLOR_B_M, edgecolor="black", linewidth=0.5,
               label="B: TerraMind + DEM + S2", zorder=2)

        for k, b in enumerate(BASINS):
            dd = data_dem[b][f"delta_{key}_mean"]
            sig_d = fold_tci_sig(data_dem[b], f"delta_{key}")
            dm = data_mm[b][f"delta_{key}_mean"]
            sig_m = fold_tci_sig(data_mm[b], f"delta_{key}")
            y_top = max(max(a_m[k] + a_s[k], bd_m[k] + bd_s[k], bm_m[k] + bm_s[k]), 0.85) + 0.025
            ax.text(k, y_top,
                    f"DEM: $\\Delta$={dd:+.3f}$^{{{sig_d}}}$\n"
                    f"MM: $\\Delta$={dm:+.3f}$^{{{sig_m}}}$",
                    ha="center", va="bottom", fontsize=6.5)

        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{SHORT[b]}\n({REGIME[b]})" for b in BASINS],
            fontsize=7.5,
        )
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontsize=9, fontweight="bold")
        ax.set_ylim(0.4, 1.1)
        ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6, zorder=0)
        if j == 0:
            ax.legend(loc="lower right", frameon=False, fontsize=7)

    save_fig(fig, "fig6_multimodal")
    plt.close(fig)


if __name__ == "__main__":
    main()
