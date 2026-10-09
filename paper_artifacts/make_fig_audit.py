"""Figures for the audit: what each input correction changed, and the buffer sweep.

fig_snapshots.pdf  dROC of every benchmark cell at four stages of the input
                   pipeline, one panel per basin. Stage folders hold the result
                   files preserved before each correction:
                     _globaldraw      composite drawn per basin
                     _unstandardized  TerraMind before standardisation (per-tile composite)
                     _prithvi_offset  Prithvi before the BOA offset was removed
                     (top level)      both corrected
fig_buffers.pdf    Prithvi dROC and the coordinates-only floor as the
                   negative-exclusion window shrinks from 3.36 km to zero.
"""
from __future__ import annotations

import json
import math

import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import t as student_t

from paths import FIGURES_DIR, RESULTS

mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif", "Liberation Serif"],
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "legend.fontsize": 7, "axes.linewidth": 0.6,
})
CM = 1 / 2.54
FULL_W = 16.0 * CM
BASINS = [("06_rio_huasco", "Huasco", "semi-arid"),
          ("09_rio_maipo", "Maipo", "mediterranean"),
          ("11_rio_maule", "Maule", "temperate-humid")]
MODELS = [("terramind_linprobe_spatial", "TerraMind+DEM", "#4d8b55"),
          ("terramind_linprobe_spatial_dem+s2l2a", "TerraMind+DEM+S2", "#3f6fa8"),
          ("prithvi-300m_linprobe_spatial", "Prithvi-EO-2.0+S2", "#c0562b")]
STAGES = [("Per-basin scene draw", {"terramind": "_globaldraw", "prithvi": "_globaldraw"}),
          ("Per-tile scene draw", {"terramind": "_unstandardized", "prithvi": "_prithvi_offset"}),
          ("TerraMind standardised", {"terramind": "", "prithvi": "_prithvi_offset"}),
          ("Prithvi offset removed", {"terramind": "", "prithvi": ""})]


def tci(v):
    n = len(v)
    m = sum(v) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1))
    h = float(student_t.ppf(0.975, n - 1)) * sd / math.sqrt(n)
    return m, m - h, m + h


def deltas(path):
    return [f["delta_roc"] for f in json.loads(path.read_text())["fold_results"]]


def snapshots():
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 6.4 * CM), sharey=True,
                             constrained_layout=True)
    xs = range(len(STAGES))
    for ax, (slug, name, regime) in zip(axes, BASINS):
        ax.axhline(0, color="black", lw=0.7)
        for j, (stem, label, color) in enumerate(MODELS):
            sub_key = "prithvi" if "prithvi" in stem else "terramind"
            pts = []
            for i, (_, src) in enumerate(STAGES):
                sub = src[sub_key]
                p = (RESULTS / sub / f"{slug}_{stem}.json") if sub else (RESULTS / f"{slug}_{stem}.json")
                m, lo, hi = tci(deltas(p))
                pts.append(m)
                x = i + (j - 1) * 0.16
                sig = lo > 0 or hi < 0
                ax.plot([x, x], [lo, hi], color=color, lw=0.8, alpha=0.7)
                ax.plot(x, m, "o", ms=4, mec=color, mfc=color if sig else "white", mew=1.1)
            ax.plot([i + (j - 1) * 0.16 for i in xs], pts, color=color, lw=0.7, alpha=0.5)
        ax.set_title(f"{name} ({regime})", fontsize=8.5)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([f"S{i + 1}" for i in xs], fontsize=7)
        ax.set_xlim(-0.5, len(STAGES) - 0.5)
    axes[0].set_ylabel(r"$\Delta$ROC vs. baseline")
    handles = [plt.Line2D([], [], color=c, marker="o", lw=0.8, ms=4, label=l) for _, l, c in MODELS]
    handles.append(plt.Line2D([], [], color="grey", marker="o", mfc="white", lw=0, ms=4,
                              label="open: fold-level 95% CI includes zero"))
    axes[1].set_xlabel("input-pipeline stage (S1 per-basin draw, S2 per-tile draw, "
                       "S3 TerraMind standardised, S4 Prithvi offset removed)", fontsize=6.5)
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False)
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES_DIR / f"fig_snapshots.{ext}", dpi=300)
    print("[write] fig_snapshots.pdf")


def buffers():
    s = json.loads((RESULTS / "buffer_sensitivity_summary.json").read_text())
    bufs = [("112", "3.36 km"), ("33", "1 km"), ("17", "0.5 km"), ("0", "none")]
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 5.6 * CM), sharey=True,
                             constrained_layout=True)
    for ax, (slug, name, regime) in zip(axes, BASINS):
        ax.axhline(0, color="black", lw=0.7)
        ms = []
        for i, (b, _lab) in enumerate(bufs):
            m, lo, hi = s[slug][b]["PRITHVI"]
            sig = lo > 0 or hi < 0
            ax.plot([i, i], [lo, hi], color="#c0562b", lw=0.8, alpha=0.7)
            ax.plot(i, m, "o", ms=4, mec="#c0562b", mfc="#c0562b" if sig else "white", mew=1.1)
            ms.append(m)
            ax.text(i, -0.235, f"{s[slug][b]['coords10']:.2f}", ha="center", fontsize=6.5, color="#3f6fa8")
        ax.plot(range(len(bufs)), ms, color="#c0562b", lw=0.7, alpha=0.6)
        ax.set_xticks(range(len(bufs)))
        ax.set_xticklabels([l for _, l in bufs], fontsize=6.5)
        ax.set_xlim(-0.5, len(bufs) - 0.5)
        ax.set_ylim(-0.26, 0.26)
        ax.set_title(f"{name} ({regime})", fontsize=8.5)
        ax.set_xlabel("negative-exclusion window", fontsize=7)
    axes[0].set_ylabel(r"Prithvi $\Delta$ROC vs. baseline")
    axes[0].text(-0.45, -0.205, "coordinates-only ROC:", fontsize=6, color="#3f6fa8")
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES_DIR / f"fig_buffers.{ext}", dpi=300)
    print("[write] fig_buffers.pdf")


if __name__ == "__main__":
    snapshots()
    buffers()
