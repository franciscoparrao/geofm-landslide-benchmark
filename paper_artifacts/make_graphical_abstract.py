"""Graphical abstract for the C&G submission (Elsevier spec: >= 1328x531 px).

Three basin panels (AUC ROC of the point baseline, the context-augmented
manual baseline, and the best GeoFM) plus a title/verdict column. Numbers are
the within-environment results of Tables 3 and 6.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from paths import FIGURES_DIR as OUT_DIR

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Liberation Serif"],
    "pdf.fonttype": 42,
})

C_A = "#888888"      # point baseline
C_CTX = "#b0691f"    # context-augmented manual
C_FM = "#1b6f9e"     # best GeoFM
INK = "#26221c"

BASINS = [
    ("Huasco\nsemi-arid",       0.937, 0.996, 0.990, "GeoFM $=$ manual context"),
    ("Maipo\nmediterranean",    0.811, 0.876, 0.932, "GeoFM $\\approx$ context (n.s.)"),
    ("Maule\ntemperate-humid",  0.914, 0.918, 0.867, "every GeoFM fails$^{*}$"),
]


def main():
    fig = plt.figure(figsize=(13.28, 5.31), dpi=100)
    gs = fig.add_gridspec(1, 4, width_ratios=[1.35, 1, 1, 1],
                          left=0.02, right=0.98, top=0.86, bottom=0.14, wspace=0.34)

    # ── title / verdict column ──
    ax0 = fig.add_subplot(gs[0]); ax0.set_axis_off()
    ax0.text(0, 1.02, "Where do geospatial\nfoundation\nmodels help?",
             fontsize=20, fontweight="bold", va="top", color=INK, linespacing=1.25)
    ax0.text(0, 0.60, "TerraMind & Prithvi-EO-2.0 vs.\nhand-engineered features,\nthree Chilean climate regimes",
             fontsize=12.5, va="top", color=INK, linespacing=1.4)
    ax0.text(0, 0.32, "The benchmark a GeoFM\nmust beat: a context-augmented\nmanual stack (multi-scale\nwindow statistics,\nseconds of compute).",
             fontsize=12, va="top", color=C_CTX, fontweight="bold", linespacing=1.35)

    # ── basin panels ──
    labels = ["17 features\nat the pixel", "+ multi-scale\ncontext", "best\nGeoFM"]
    for i, (name, a, ctx, fm, verdict) in enumerate(BASINS):
        ax = fig.add_subplot(gs[i + 1])
        bars = ax.bar([0, 1, 2], [a, ctx, fm], width=0.62,
                      color=[C_A, C_CTX, C_FM], edgecolor="black", linewidth=0.8)
        for x, v in zip([0, 1, 2], [a, ctx, fm]):
            ax.text(x, v + 0.006, f"{v:.3f}", ha="center", fontsize=10.5,
                    fontweight="bold", color=INK)
        ax.set_ylim(0.60, 1.02)
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(name, fontsize=13, fontweight="bold", pad=8)
        if i == 0:
            ax.set_ylabel("AUC ROC (spatial holdout)", fontsize=11)
        else:
            ax.set_yticklabels([])
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(1, 0.625, verdict, ha="center", fontsize=10.5,
                style="italic", color=INK)

    fig.text(0.985, 0.02,
             "$^{*}$fold-level 95% CIs exclude zero; failure amplifies under a hardened negative-sampling task",
             ha="right", fontsize=9, color="#6b645a")

    out = OUT_DIR / "graphical_abstract"
    fig.savefig(f"{out}.png", dpi=150)   # 1992 x 797 px
    fig.savefig(f"{out}.pdf")
    print(f"  wrote {out}.png (+ pdf)")


if __name__ == "__main__":
    main()
