"""Figure 1 — Map of Chile with the five pilot basins + climatic profile.

Left panel: all BNA basins of Chile (as country background) with the five
pilot basins highlighted in distinct colors. Right panel: latitudinal profile
of mean annual precipitation and temperature for the five basins.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from paths import BASIN_POLY_DIR as POLY_DIR
from paths import COUNTRY_SHP
from paths import FIGURES_DIR as OUT_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Style — CAGEO friendly
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Liberation Serif"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.7,
    "lines.linewidth": 1.2,
    "patch.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

CM = 1 / 2.54
FULL_W = 17.4 * CM
SHORT_NAMES = {
    "01_rio_lluta": "Lluta", "06_rio_huasco": "Huasco",
    "09_rio_maipo": "Maipo", "11_rio_maule": "Maule",
    "13_rio_bueno": "Bueno",
}
REGIME = {
    "01_rio_lluta": "Hyperarid", "06_rio_huasco": "Semi-arid",
    "09_rio_maipo": "Mediterranean", "11_rio_maule": "Temperate-humid",
    "13_rio_bueno": "Temperate-rainy",
}
# Climatic data (from Table 1, computed in make_basin_table.py)
# Climatology is computed from the same rasters and the same DEM>0 mask that
# Table 1 uses, by calling that table's own routine. These numbers were
# previously hardcoded here and had drifted from the table in four of the five
# basins (Maule read 882 mm against the table's 920), which is the failure mode
# a figure and a table sharing no source will always eventually reach.
from make_basin_table import BASINS as _TABLE_BASINS
from make_basin_table import basin_stats as _basin_stats

BASIN_DATA = {
    bid: {"lat": s["lat"], "P": s["p_mean"], "T": s["t_mean"]}
    for bid, s in ((b[0], _basin_stats(b[0])) for b in _TABLE_BASINS)
}
COLORS = {
    "01_rio_lluta": "#e41a1c",     # red — hyperarid
    "06_rio_huasco": "#ff7f00",    # orange — semi-arid
    "09_rio_maipo": "#984ea3",     # purple — mediterranean
    "11_rio_maule": "#377eb8",     # blue — temperate-humid
    "13_rio_bueno": "#1b9e77",     # green — temperate-rainy
}
ORDER = list(BASIN_DATA.keys())


def main():
    # Load country background: all BNA basins
    country = gpd.read_file(COUNTRY_SHP)
    if country.crs is None or country.crs.to_epsg() not in (4326, 4269):
        country = country.to_crs(epsg=4326)

    # Load each pilot basin
    pilot = {}
    for b in ORDER:
        gdf = gpd.read_file(POLY_DIR / f"{b}.geojson")
        if gdf.crs is None or gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)
        pilot[b] = gdf

    fig = plt.figure(figsize=(FULL_W, 13 * CM), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[0.65, 1.0])

    # === Left panel: map ===
    ax = fig.add_subplot(gs[0, 0])
    country.plot(ax=ax, color="#eeeeee", edgecolor="#888888",
                 linewidth=0.25, zorder=1)
    for b in ORDER:
        pilot[b].plot(ax=ax, color=COLORS[b], edgecolor="black",
                      linewidth=0.5, alpha=0.85, zorder=2)
        # Centroid label
        centroid = pilot[b].geometry.unary_union.centroid
        ax.annotate(
            SHORT_NAMES[b],
            xy=(centroid.x, centroid.y),
            xytext=(10, 0), textcoords="offset points",
            fontsize=8, fontweight="bold",
            ha="left", va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=COLORS[b],
                      lw=0.6, alpha=0.9),
            zorder=3,
        )
    ax.set_xlabel("Longitude ($^{\\circ}$E)")
    ax.set_ylabel("Latitude ($^{\\circ}$N)")
    ax.set_title("(a) Five-basin transect across Chile",
                 loc="left", fontsize=9, fontweight="bold")
    ax.set_aspect("equal")
    # Zoom on continental Chile (ignore territorios australes outliers)
    ax.set_xlim(-76, -65)
    ax.set_ylim(-46, -16)
    ax.grid(alpha=0.25, linestyle=":", linewidth=0.4)

    # === Right panel: climatic profile ===
    ax2 = fig.add_subplot(gs[0, 1])
    lats = np.array([BASIN_DATA[b]["lat"] for b in ORDER])
    Ps = np.array([BASIN_DATA[b]["P"] for b in ORDER])
    Ts = np.array([BASIN_DATA[b]["T"] for b in ORDER])

    # Bar plot of precipitation, with temperature overlaid as twinned line
    x = np.arange(len(ORDER))
    bar_colors = [COLORS[b] for b in ORDER]
    bars = ax2.bar(x, Ps, color=bar_colors, edgecolor="black",
                   linewidth=0.5, alpha=0.85, label="Annual precip.")
    for b, p in zip(bars, Ps):
        ax2.text(b.get_x() + b.get_width() / 2, p + 30,
                 f"{p:,.0f}", ha="center", va="bottom", fontsize=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(
        [f"{SHORT_NAMES[b]}\n({REGIME[b]})\n${BASIN_DATA[b]['lat']:.1f}^\\circ$S"
         for b in ORDER],
        fontsize=7, rotation=15, ha="right",
    )
    ax2.set_ylabel("Mean annual precipitation (mm/yr)")
    ax2.set_ylim(0, 1900)
    ax2.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    ax2.set_title("(b) Climate gradient: precipitation and temperature",
                  loc="left", fontsize=9, fontweight="bold")

    # Twin axis for temperature
    ax3 = ax2.twinx()
    ax3.plot(x, Ts, marker="o", color="black", lw=1.4,
             markersize=6, label="Mean annual temp.", zorder=10)
    for xi, t in zip(x, Ts):
        ax3.annotate(f"{t:+.1f}$^\\circ$C", xy=(xi, t),
                     xytext=(0, -14), textcoords="offset points",
                     ha="center", va="top", fontsize=7.5, color="black")
    ax3.set_ylabel("Mean annual temperature ($^{\\circ}$C)")
    ax3.set_ylim(0, 16)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(True)
    ax3.spines["right"].set_linewidth(0.7)

    # Legend combined
    handles_left, labels_left = ax2.get_legend_handles_labels()
    handles_right, labels_right = ax3.get_legend_handles_labels()
    ax2.legend(handles_left + handles_right, labels_left + labels_right,
               loc="upper left", frameon=False, fontsize=7.5)

    out_pdf = OUT_DIR / "fig1_map.pdf"
    out_png = OUT_DIR / "fig1_map.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"  wrote {out_pdf}")
    print(f"  wrote {out_png}")


if __name__ == "__main__":
    main()
