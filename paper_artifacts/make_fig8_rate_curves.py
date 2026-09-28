"""Figure 8 and Table 8 — the surfaces validated as maps.

Reads results/{basin}_surface_validation.json (src/surface_validation.py).

Figure: one panel per basin, cumulative share of the inventory against the
cumulative share of terrain ranked from most to least susceptible, both models
on the common 600 m support. Dashed lines are the success-rate curves of the
surfaces fitted on every point (the maps of Figure 7); solid lines are the
prediction-rate curves of the out-of-fold surfaces, the ones that carry
predictive meaning. The diagonal is a surface that ranks terrain at random.

Table: per basin and model, the areas under both curves, the share of events
captured by the top 10% of the terrain, and the density ratio (share of events
over share of area) per equal-area quintile class of the out-of-fold surface.
"""
from __future__ import annotations

import json

import matplotlib as mpl
import matplotlib.pyplot as plt

from paths import FIGURES_DIR, RESULTS, TABLES_DIR

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Liberation Serif"],
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "legend.fontsize": 7, "axes.linewidth": 0.6,
})
CM = 1 / 2.54
FULL_W = 16.0 * CM
BASINS = [("06_rio_huasco", "Huasco", "semi-arid"),
          ("09_rio_maipo", "Maipo", "mediterranean"),
          ("11_rio_maule", "Maule", "temperate-humid")]
MODELS = [("A", "Geomorphometric baseline (A)", "#404040"),
          ("PRITHVI", "Prithvi-EO-2.0-300M", "#d95f02")]
SUPPORT = "common"


def load(slug):
    return json.loads((RESULTS / f"{slug}_surface_validation.json").read_text())


def figure(data):
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 5.9 * CM),
                             constrained_layout=True, sharey=True)
    for ax, (slug, name, regime) in zip(axes, BASINS):
        d = data[slug][SUPPORT]
        ax.plot([0, 1], [0, 1], color="#999999", lw=0.6, ls=":")
        for key, label, color in MODELS:
            for kind, ls, lw in (("fit", "--", 0.8), ("oof", "-", 1.3)):
                r = d[f"{key}_{kind}"]
                ax.plot(r["curve"]["area_share"], r["curve"]["event_share"],
                        color=color, ls=ls, lw=lw)
        ax.set_title(f"{name} ({regime})", fontsize=8.5)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
        ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
        ax.set_aspect("equal")
        ax.text(0.97, 0.05,
                "prediction-rate AUC\n" + "\n".join(f"{short}: {d[k + '_oof']['auc']:.3f}"
                                        for (k, _, _), short in zip(MODELS, ("A", "Prithvi"))),
                transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5)
    axes[0].set_ylabel("Share of inventory captured")
    axes[1].set_xlabel("Share of terrain, ranked from most to least susceptible")
    handles = [plt.Line2D([], [], color=c, lw=1.3, label=l) for _, l, c in MODELS]
    handles += [plt.Line2D([], [], color="black", lw=1.3, label="prediction rate (out-of-fold)"),
                plt.Line2D([], [], color="black", lw=0.8, ls="--",
                           label="success rate (fitted on all points)")]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False)
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES_DIR / f"fig8_rate_curves.{ext}", dpi=300)
    print(f"[write] fig8_rate_curves.pdf")


def table(data):
    lines = [
        r"\begin{table*}",
        r"\centering",
        r"\caption{Susceptibility surfaces validated as maps, both models on the "
        r"common $600$~m support over cells whose Sentinel-2 patch is present "
        r"(Section~\ref{sec:methods:maps}). Success: area under the success-rate "
        r"curve of the surface fitted on every benchmark point. Prediction: area "
        r"under the prediction-rate curve of the out-of-fold surface, in which every cell is "
        r"predicted by the fold model that held its $10$~km block out. Top $10\%$: "
        r"share of the inventory falling in the tenth of the terrain the "
        r"out-of-fold surface ranks most susceptible. Density ratio: share of "
        r"events over share of area in each equal-area quintile class of the "
        r"out-of-fold surface, from very low (VL) to very high (VH); a value of "
        r"$1$ is what a random surface would give.}",
        r"\label{tab:ratecurves}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{llcccccccc}",
        r"\toprule",
        r" & & \multicolumn{2}{c}{Rate-curve AUC} & & \multicolumn{5}{c}{Density ratio (out-of-fold)} \\",
        r"\cmidrule(lr){3-4} \cmidrule(lr){6-10}",
        r"Basin & Model & Success & Prediction & Top $10\%$ & VL & L & M & H & VH \\",
        r"\midrule",
    ]
    for i, (slug, name, _regime) in enumerate(BASINS):
        d = data[slug][SUPPORT]
        n_ev = d["A_oof"]["n_events"]
        for j, (key, _label, _c) in enumerate(MODELS):
            fit, oof = d[f"{key}_fit"], d[f"{key}_oof"]
            dr = " & ".join(f"{c['density_ratio']:.2f}" for c in oof["classes"])
            first = f"{name} ($n = {n_ev}$)" if j == 0 else ""
            model = "A" if key == "A" else "Prithvi"
            lines.append(f"{first} & {model} & {fit['auc']:.3f} & {oof['auc']:.3f} & "
                         f"{oof['top10_capture'] * 100:.0f}\\% & {dr} \\\\")
        if i < len(BASINS) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (TABLES_DIR / "tab8_rate_curves.tex").write_text("\n".join(lines) + "\n")
    print("[write] tab8_rate_curves.tex")


def main():
    data = {slug: load(slug) for slug, _, _ in BASINS
            if (RESULTS / f"{slug}_surface_validation.json").exists()}
    missing = [s for s, _, _ in BASINS if s not in data]
    if missing:
        raise SystemExit(f"missing surface validation for {missing}")
    figure(data)
    table(data)


if __name__ == "__main__":
    main()
