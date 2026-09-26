"""Generate publication-quality figures for the paper.

Outputs go to paper/figures/ as both .pdf (vector, for LaTeX) and .png
(raster preview).

Style: single-column width ≈ 84 mm; full width ≈ 174 mm.
Uses matplotlib defaults adjusted for journal-style readability.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
from paths import RESULTS
from paths import FIGURES_DIR as OUT_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# Style — CAGEO friendly: clean, readable, B&W-compatible palette
# ----------------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Liberation Serif"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.titlesize": 10,
    "axes.linewidth": 0.7,
    "lines.linewidth": 1.2,
    "patch.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Width conversions
CM = 1 / 2.54
SINGLE_W = 8.4 * CM     # 84 mm
FULL_W = 17.4 * CM      # 174 mm
SHORT_NAMES = {
    "01_rio_lluta": "Lluta", "06_rio_huasco": "Huasco",
    "09_rio_maipo": "Maipo", "11_rio_maule": "Maule",
    "13_rio_bueno": "Bueno",
}
REGIME = {
    "06_rio_huasco": "Semi-arid",
    "09_rio_maipo": "Mediterranean",
    "11_rio_maule": "Temperate-humid",
}
COLOR_A = "#888888"     # gray for baseline manual features
COLOR_B = "#1b6f9e"     # blue for pretrained TerraMind
COLOR_B_RND = "#d65f3a" # orange for random-init TerraMind


def save_fig(fig, name):
    pdf = OUT_DIR / f"{name}.pdf"
    png = OUT_DIR / f"{name}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    print(f"  wrote {pdf.name} + {png.name}")


# ----------------------------------------------------------------------
# Figure 4 — Baseline vs TerraMind per basin (spatial holdout)
# ----------------------------------------------------------------------
def figure_4():
    basins = ["06_rio_huasco", "09_rio_maipo", "11_rio_maule"]
    data = {}
    for b in basins:
        d = json.loads((RESULTS / f"{b}_terramind_linprobe_spatial.json").read_text())
        data[b] = d

    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 8 * CM), constrained_layout=True)

    for j, (metric_key, ylabel, title) in enumerate([
        ("roc", "AUC ROC", "(a) AUC ROC"),
        ("pr",  "AUC PR",  "(b) AUC PR"),
    ]):
        ax = axes[j]
        x = np.arange(len(basins))
        a_means = [data[b][f"A_{metric_key}_mean"] for b in basins]
        a_stds  = [data[b][f"A_{metric_key}_std"]  for b in basins]
        b_means = [data[b][f"B_{metric_key}_mean"] for b in basins]
        b_stds  = [data[b][f"B_{metric_key}_std"]  for b in basins]

        w = 0.36
        ax.bar(x - w/2, a_means, w, yerr=a_stds, capsize=2.5,
               color=COLOR_A, edgecolor="black", linewidth=0.5,
               label="A: 17 geomorphometric features", zorder=2)
        ax.bar(x + w/2, b_means, w, yerr=b_stds, capsize=2.5,
               color=COLOR_B, edgecolor="black", linewidth=0.5,
               label="B: TerraMind+DEM (pretrained)", zorder=2)
        # annotate Δ
        for k, b in enumerate(basins):
            delta_mean = data[b][f"delta_{metric_key}_mean"]
            ci = data[b][f"delta_{metric_key}_ci_95"]
            sig = "*" if (ci[0] > 0) or (ci[1] < 0) else "n.s."
            y_top = max(a_means[k] + a_stds[k], b_means[k] + b_stds[k]) + 0.02
            ax.text(k, y_top, f"Δ={delta_mean:+.3f}\n{sig}",
                    ha="center", va="bottom", fontsize=7,
                    color=("black" if sig != "n.s." else "#666"))

        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{SHORT_NAMES[b]}\n({REGIME[b]})" for b in basins],
            fontsize=7.5,
        )
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontsize=9, fontweight="bold")
        ax.set_ylim(0.5, 1.05)
        ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6, zorder=0)
        if j == 0:
            ax.legend(loc="lower right", frameon=False, fontsize=7.5)

    save_fig(fig, "fig4_benchmark_per_basin")
    plt.close(fig)


# ----------------------------------------------------------------------
# Figure 5 — Random-init ablation per basin (3 bars: A, B-pre, B-rnd)
# ----------------------------------------------------------------------
def figure_5():
    basins = ["06_rio_huasco", "09_rio_maipo", "11_rio_maule"]
    data_pre = {b: json.loads((RESULTS / f"{b}_terramind_linprobe_spatial.json").read_text())
                for b in basins}
    data_rnd = {b: json.loads((RESULTS / f"{b}_terramind_linprobe_spatial_randinit.json").read_text())
                for b in basins}

    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 8 * CM), constrained_layout=True)

    for j, (metric_key, ylabel, title) in enumerate([
        ("roc", "AUC ROC", "(a) AUC ROC"),
        ("pr",  "AUC PR",  "(b) AUC PR"),
    ]):
        ax = axes[j]
        x = np.arange(len(basins))
        a_m = [data_pre[b][f"A_{metric_key}_mean"] for b in basins]
        a_s = [data_pre[b][f"A_{metric_key}_std"]  for b in basins]
        bp_m = [data_pre[b][f"B_{metric_key}_mean"] for b in basins]
        bp_s = [data_pre[b][f"B_{metric_key}_std"]  for b in basins]
        br_m = [data_rnd[b][f"B_{metric_key}_mean"] for b in basins]
        br_s = [data_rnd[b][f"B_{metric_key}_std"]  for b in basins]

        w = 0.26
        ax.bar(x - w, a_m,  w, yerr=a_s,  capsize=2.5,
               color=COLOR_A, edgecolor="black", linewidth=0.5,
               label="A: 17 features", zorder=2)
        ax.bar(x,     bp_m, w, yerr=bp_s, capsize=2.5,
               color=COLOR_B, edgecolor="black", linewidth=0.5,
               label="B: TerraMind pretrained", zorder=2)
        ax.bar(x + w, br_m, w, yerr=br_s, capsize=2.5,
               color=COLOR_B_RND, edgecolor="black", linewidth=0.5,
               label="B: TerraMind random-init", zorder=2)

        # annotate gap pretrained ↔ random
        for k, b in enumerate(basins):
            gap = bp_m[k] - br_m[k]
            y_top = max(a_m[k] + a_s[k], bp_m[k] + bp_s[k], br_m[k] + br_s[k]) + 0.02
            ax.text(k, y_top, f"gap={gap:+.3f}",
                    ha="center", va="bottom", fontsize=7,
                    color=("black" if abs(gap) > 0.02 else "#666"))

        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{SHORT_NAMES[b]}\n({REGIME[b]})" for b in basins],
            fontsize=7.5,
        )
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontsize=9, fontweight="bold")
        ax.set_ylim(0.5, 1.05)
        ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6, zorder=0)
        if j == 0:
            ax.legend(loc="lower right", frameon=False, fontsize=7.5)

    save_fig(fig, "fig5_random_init_ablation")
    plt.close(fig)


# ----------------------------------------------------------------------
# Figure 3 — Geomorphometric regimes across 5 basins
#   Panel a: morphotype shares heatmap (5 basins × 4 universal morphotypes)
#   Panel b: bootstrap ARI curve by basin
# ----------------------------------------------------------------------
def figure_3():
    syn = json.loads((RESULTS / "synthesis_phase1.json").read_text())
    components = syn["components"]
    basins = syn["basins"]
    SHORT_5 = {b: SHORT_NAMES[b] for b in basins}
    REGIME_5 = {
        "01_rio_lluta": "Hyperarid", "06_rio_huasco": "Semi-arid",
        "09_rio_maipo": "Mediterranean", "11_rio_maule": "Temperate-humid",
        "13_rio_bueno": "Temperate-rainy",
    }

    universal = [c for c in components if c["n_basins"] >= 5]
    sigs = {}
    for b in basins:
        d = json.loads((RESULTS / f"{b}_cluster_signatures.json").read_text())
        run = next(r for r in d["runs"] if r["k"] == 5)
        sigs[b] = {c["id"]: c.get("share", 0.0) for c in run["clusters"]}

    matrix = np.zeros((len(universal), len(basins)))
    for i, comp in enumerate(universal):
        for basin, cluster_id in comp["members"]:
            j = basins.index(basin)
            matrix[i, j] += sigs[basin].get(cluster_id, 0.0) * 100  # percent

    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 8 * CM),
                             gridspec_kw={"width_ratios": [1.0, 1.2]},
                             constrained_layout=True)
    # ---- Panel a: morphotype shares heatmap
    ax = axes[0]
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=65)
    ax.set_xticks(range(len(basins)))
    ax.set_xticklabels([SHORT_5[b] for b in basins], rotation=20)
    ax.set_yticks(range(len(universal)))
    ax.set_yticklabels([f"M{i + 1}" for i in range(len(universal))])
    for i in range(len(universal)):
        for j in range(len(basins)):
            v = matrix[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                    color="white" if v > 32 else "black", fontsize=7.5)
    ax.set_title("(a) Universal morphotype shares (%)",
                 loc="left", fontsize=9, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, label="share (%)")
    cbar.outline.set_linewidth(0.5)

    # ---- Panel b: bootstrap ARI curve
    ax = axes[1]
    colors = plt.colormaps["viridis"](np.linspace(0.05, 0.9, 5))
    K_VALUES = (3, 5, 7, 10, 12)
    for i, b in enumerate(basins):
        bs = json.loads((RESULTS / f"{b}_bootstrap_stability.json").read_text())
        means = {r["k"]: r["ari_mean"] for r in bs["runs"]}
        stds = {r["k"]: r["ari_std"] for r in bs["runs"]}
        ks = [k for k in K_VALUES if k in means]
        ys = [means[k] for k in ks]
        es = [stds[k] for k in ks]
        ax.errorbar(ks, ys, yerr=es, marker="o", markersize=4.5,
                    color=colors[i], capsize=2, lw=1.1, alpha=0.85,
                    label=f"{SHORT_5[b]} ({REGIME_5[b]})")

    ax.set_xlabel("K")
    ax.set_ylabel("Bootstrap ARI")
    ax.set_xticks(K_VALUES)
    ax.set_ylim(0.2, 1.0)
    ax.grid(linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_title("(b) Cluster stability across K", loc="left", fontsize=9, fontweight="bold")
    ax.legend(loc="upper right", frameon=False, fontsize=7)

    save_fig(fig, "fig3_geomorphometric_regimes")
    plt.close(fig)


# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("Generating Figure 4: benchmark per basin")
    figure_4()
    print("Generating Figure 5: random-init ablation")
    figure_5()
    print("Generating Figure 3: geomorphometric regimes")
    figure_3()
    print(f"\nAll figures saved to {OUT_DIR}")
