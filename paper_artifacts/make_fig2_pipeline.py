"""Figure 2 — Pipeline diagram for CAGEO paper.

Four-row workflow:
  Pipeline A (manual point baseline): 17 per-pixel geomorphometric features -> RF
  Pipeline B (TerraMind, frozen):     DEM [+S2] patch -> 192-d embedding -> RF
  Pipeline B' (Prithvi, frozen):      6-band HLS-equivalent patch -> 1024-d embedding -> RF
  Point-probe controls:               multi-scale window stats (terrain+spectral) -> RF
Both FMs carry a random-initialization ablation. Shared downstream: identical RF,
5-fold spatial CV (10 km blocks), fold-level paired t-inference.

Style follows the /paper-figures skill conventions:
  - Elsevier double-column width (190 mm).
  - Serif font matching the manuscript body.
  - Per-entity colors consistent with figs 3-5 (A: gray, B: blue).
  - No axes spines (diagram, not a plot).
  - Vector PDF output for LaTeX inclusion.
  - No ax.set_title (caption lives in LaTeX).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path("/home/franciscoparrao/proyectos/no_supervisado_superficie")
OUT_DIR = ROOT / "paper/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Liberation Serif"],
    "font.size": 9,
    "axes.linewidth": 0.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

CM = 1 / 2.54
FIG_W = 17.4 * CM
FIG_H = 11.0 * CM

COLOR_DEM    = "#f0d28d"
COLOR_A      = "#888888"
COLOR_B      = "#1b6f9e"
COLOR_RF     = "#3a3a3a"
COLOR_EVAL   = "#1f7a4d"
COLOR_TEXT   = "#222222"


def rounded_box(ax, xy, w, h, color, label, zorder=2, text_color="white",
                fontsize=7.5, fontweight="normal"):
    x, y = xy
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.0,rounding_size=0.4",
        facecolor=color, edgecolor="black", linewidth=0.7, zorder=zorder,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2, y + h / 2, label,
        ha="center", va="center",
        fontsize=fontsize, fontweight=fontweight,
        color=text_color, zorder=zorder + 1,
        linespacing=1.3,
    )


def arrow(ax, src, dst, color="black", lw=1.0, zorder=1):
    arr = FancyArrowPatch(
        src, dst,
        arrowstyle="->,head_length=5,head_width=3",
        color=color, linewidth=lw, mutation_scale=1,
        shrinkA=1, shrinkB=1, zorder=zorder,
    )
    ax.add_patch(arr)


def main():
    fig, ax = plt.subplots(figsize=(FIG_W, 10.5 * CM), constrained_layout=False)
    ax.set_xlim(0, 100)
    ax.set_ylim(16, 100)
    ax.set_axis_off()

    C_PR = "#7a4a9e"   # Prithvi
    C_CTL = "#b0691f"  # manual controls

    # === Inputs (left column) ===
    rounded_box(ax, (2, 66), 13, 16, COLOR_DEM,
                "DEM 30 m\nCopernicus\nGLO-30", text_color=COLOR_TEXT, fontsize=7.5)
    rounded_box(ax, (2, 24), 13, 16, "#9ec7e0",
                "Sentinel-2\nL2A comp.\n(2023)", text_color=COLOR_TEXT, fontsize=7.5)

    # === Row 1: Pipeline A ===
    ax.text(20, 95, "Pipeline A $\\bullet$ manual point baseline",
            ha="left", va="center", fontsize=8.5, fontweight="bold", color=COLOR_A)
    rounded_box(ax, (20, 84), 24, 9, COLOR_A,
                "17 geomorphometric\nfeatures at the pixel", fontsize=7.5)

    # === Row 2: TerraMind ===
    ax.text(20, 78, "Pipeline B $\\bullet$ TerraMind v1-tiny (frozen)",
            ha="left", va="center", fontsize=8.5, fontweight="bold", color=COLOR_B)
    rounded_box(ax, (20, 67), 17, 9, COLOR_B,
                "224$\\times$224 patch\nDEM [+S2L2A]", fontsize=7)
    rounded_box(ax, (40, 67), 14, 9, COLOR_B,
                "encoder\n5.4 M par.", fontsize=7.5)
    rounded_box(ax, (57, 67), 10, 9, COLOR_B, "192-d\nembed.", fontsize=7.5)
    arrow(ax, (37, 71.5), (40, 71.5)); arrow(ax, (54, 71.5), (57, 71.5))
    ax.text(47, 64.5, "random-init ablation (Xavier, seed 42)",
            ha="center", va="center", fontsize=6.5, style="italic", color=COLOR_B)

    # === Row 3: Prithvi ===
    ax.text(20, 60, "Pipeline B\' $\\bullet$ Prithvi-EO-2.0-300M (frozen)",
            ha="left", va="center", fontsize=8.5, fontweight="bold", color=C_PR)
    rounded_box(ax, (20, 49), 17, 9, C_PR,
                "224$\\times$224 patch\n6 HLS bands", fontsize=7)
    rounded_box(ax, (40, 49), 14, 9, C_PR,
                "encoder\n300 M par.", fontsize=7.5)
    rounded_box(ax, (57, 49), 10, 9, C_PR, "1024-d\nembed.", fontsize=7.5)
    arrow(ax, (37, 53.5), (40, 53.5)); arrow(ax, (54, 53.5), (57, 53.5))
    ax.text(47, 46.5, "random-init ablation (Xavier, seed 42)",
            ha="center", va="center", fontsize=6.5, style="italic", color=C_PR)

    # === Row 4: point-probe controls ===
    ax.text(20, 42, "Point-probe controls $\\bullet$ manual",
            ha="left", va="center", fontsize=8.5, fontweight="bold", color=C_CTL)
    rounded_box(ax, (20, 31), 22, 9, C_CTL,
                "window mean/std\n0.2 / 1.1 / 3.3 km", fontsize=7)
    rounded_box(ax, (45, 31), 22, 9, C_CTL,
                "8\u2013175 features\nSPEC / A+ctx / A+full", fontsize=7)
    arrow(ax, (42, 35.5), (45, 35.5))

    # === Input arrows ===
    arrow(ax, (15, 76), (20, 88), color=COLOR_A)        # DEM -> A
    arrow(ax, (15, 73), (20, 71.5), color=COLOR_B)      # DEM -> TM
    arrow(ax, (15, 34), (20, 53.5), color=C_PR)         # S2 -> Prithvi
    arrow(ax, (15, 31), (20, 35.5), color=C_CTL)        # S2 -> controls
    arrow(ax, (12, 66), (20, 37), color=C_CTL)          # DEM -> controls
    arrow(ax, (13, 40), (22, 67), color=COLOR_B)        # S2 -> TM (multimodal)

    # === Shared evaluation (right column) ===
    rounded_box(ax, (72, 62), 26, 12, COLOR_RF,
                "Random Forest\nsame hyperparams and\npoints for all pipelines",
                fontsize=7)
    rounded_box(ax, (72, 42), 26, 12, COLOR_EVAL,
                "5-fold spatial CV\n10 km blocks", fontsize=7.5)
    rounded_box(ax, (72, 22), 26, 12, COLOR_EVAL,
                "$\\Delta$AUC ROC / PR\nfold-level paired $t$\n+ pooled bootstrap",
                fontsize=7)
    for ysrc in (88.5, 71.5, 53.5, 35.5):
        arrow(ax, (67, ysrc) if ysrc in (71.5, 53.5) else ((44, ysrc) if ysrc == 88.5 else (67, ysrc)),
              (72, 68), color="#555555", lw=0.9)
    arrow(ax, (85.5, 62), (85.5, 54), color="black")
    arrow(ax, (85.5, 42), (85.5, 34), color="black")

    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"fig2_pipeline.{ext}", dpi=300,
                    bbox_inches="tight")
    print("  wrote fig2_pipeline.pdf + fig2_pipeline.png")


if __name__ == "__main__":
    main()
