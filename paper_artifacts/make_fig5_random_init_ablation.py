"""Figure 5 — Cross-FM random-initialization ablation.

Three panels (one per basin), each showing ΔROC of pretrained vs random-init
backbones for the three FM pipelines. Error bars: 95% bootstrap CI.
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


BASINS = [("06_rio_huasco", "Huasco", "Semi-arid"),
          ("09_rio_maipo",  "Maipo",  "Mediterranean"),
          ("11_rio_maule",  "Maule",  "Temperate-humid")]

FMS = [("TM+DEM",     "terramind",    "dem", "#1b6f9e"),
       ("TM+DEM+S2",  "terramind",    "mm",  "#0d3d5c"),
       ("Prithvi+S2", "prithvi-300m", "s2",  "#bb5500")]


def load(b, model, mod, init):
    init_s = "_randinit" if init == "rand" else ""
    if model == "terramind":
        mod_s = "" if mod == "dem" else "_dem+s2l2a"
        path = RESULTS / f"{b}_terramind_linprobe_spatial{init_s}{mod_s}.json"
    else:
        path = RESULTS / f"{b}_prithvi-300m_linprobe_spatial{init_s}.json"
    return json.loads(path.read_text())


def save_fig(fig, name):
    pdf = OUT_DIR / f"{name}.pdf"
    png = OUT_DIR / f"{name}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    print(f"  wrote {pdf.name} + {png.name}")


def main():
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 8 * CM),
                             sharey=True, constrained_layout=True)

    x = np.arange(len(FMS))
    w = 0.36

    for j, (b_slug, b_name, regime) in enumerate(BASINS):
        ax = axes[j]
        pretrained_means = []; pretrained_errs = []
        random_means = []; random_errs = []
        sigs_p = []; sigs_r = []; pt_gaps = []
        for label, model, mod, color in FMS:
            d_p = load(b_slug, model, mod, "pretr")
            d_r = load(b_slug, model, mod, "rand")
            m_p, plo, phi = fold_tci(d_p)
            m_r, rlo, rhi = fold_tci(d_r)
            pretrained_means.append(d_p["delta_roc_mean"])
            random_means.append(d_r["delta_roc_mean"])
            # Clip to non-negative: bootstrap CI is not always aligned with fold-mean
            pretrained_errs.append([
                m_p - plo,
                phi - m_p,
            ])
            random_errs.append([
                m_r - rlo,
                rhi - m_r,
            ])
            sigs_p.append("*" if not (plo <= 0 <= phi) else "")
            sigs_r.append("*" if not (rlo <= 0 <= rhi) else "")
            pt_gaps.append(d_p["delta_roc_mean"] - d_r["delta_roc_mean"])

        pretrained_errs = np.array(pretrained_errs).T
        random_errs     = np.array(random_errs).T

        # Bars: pretrained with FM color, random-init in lighter hatched fill
        for i, (label, model, mod, color) in enumerate(FMS):
            ax.bar(i - w / 2, pretrained_means[i], w,
                   yerr=[[pretrained_errs[0, i]], [pretrained_errs[1, i]]], capsize=2.5,
                   color=color, edgecolor="black", linewidth=0.5, zorder=2,
                   label="pretrained" if i == 0 else None)
            ax.bar(i + w / 2, random_means[i], w,
                   yerr=[[random_errs[0, i]], [random_errs[1, i]]], capsize=2.5,
                   color="white", edgecolor=color, linewidth=0.9,
                   hatch="//", zorder=2,
                   label="random-init" if i == 0 else None)
            # Sig marker
            ax.text(i - w / 2, pretrained_means[i] + pretrained_errs[1, i] + 0.005,
                    sigs_p[i], ha="center", va="bottom", fontsize=9, fontweight="bold")
            ax.text(i + w / 2, random_means[i] + random_errs[1, i] + 0.005,
                    sigs_r[i], ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.axhline(0, color="black", lw=0.6, zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{fm[0]}\nPT {g:+.3f}" for fm, g in zip(FMS, pt_gaps)],
            fontsize=7.0,
        )
        ax.set_title(f"({chr(97 + j)}) {b_name}\n({regime})",
                     loc="left", fontweight="bold")
        ax.set_ylim(-0.16, 0.18)
        ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6, zorder=0)
        if j == 0:
            ax.set_ylabel(r"$\Delta$ROC  (pipeline B $-$ baseline A)")
            ax.legend(loc="upper left", frameon=False, fontsize=7)

    save_fig(fig, "fig5_random_init_ablation")
    plt.close(fig)


if __name__ == "__main__":
    main()
