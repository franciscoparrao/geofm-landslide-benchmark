"""Figure 7 — susceptibility surfaces for the three benchmarked basins.

The benchmark reports differences in AUC. Steger et al. (2016) show that a model
can hold a high AUC and still produce a map a geomorphologist would reject,
because the errors of a susceptibility model are spatially organised and a
scalar average is blind to their organisation. This figure puts the surfaces in
the paper so the reported differences can be looked at.

One row per basin, in the order of the transect. Columns: (a) the seventeen-layer
geomorphometric baseline at the native 30 m grid, with the inventory overlaid;
(b) Prithvi-EO-2.0-300M on the 600 m grid at which one 224 x 224 patch per cell
is computable over basins of this size; (c) their difference, with the baseline
averaged to the same grid and restricted to cells where the imagery is present.

Rendering all three basins rather than only the one that carries the headline is
deliberate: the two drier basins are where the foundation models win, and a
diagnostic applied only where it confirms the argument is not a diagnostic.

The Sentinel-2 composites are masked for snow and cloud, and that mask is
altitude-dependent. Where a cell's patch is mostly empty, Prithvi returns an
embedding the classifier reads as highly susceptible, so the gap outline is drawn
on (b) and (c) is restricted to cells where both models have their inputs.
"""
from __future__ import annotations

import csv

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.lines import Line2D

from paths import FIGURES_DIR as OUT_DIR
from paths import RESULTS, S2_COMPOSITE_DIR

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Liberation Serif"],
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "legend.fontsize": 7, "axes.linewidth": 0.6,
})
CM = 1 / 2.54
FULL_W = 16.0 * CM          # matches \textwidth under geometry{margin=2.5cm}
BASINS = [("06_rio_huasco", "Huasco", "semi-arid"),
          ("09_rio_maipo", "Maipo", "mediterranean"),
          ("11_rio_maule", "Maule", "temperate-humid")]
STRIDE = 20
MAPS = RESULTS / "susceptibility_maps"
DISPLAY_FACTOR = 4          # 30 m -> 120 m for column (a); keeps the PDF light
HLS_BAND = 2                # the composite's six HLS bands share one mask
GAP_THRESHOLD = 0.8
DIFF_CLIP = 99              # percentile of |difference| used for the colour range
INV_COLOR = "#1b9e77"
GAP_COLOR = "#3f6fd1"       # deliberately not INV_COLOR: they co-occur in one row


def block_reduce(a, f):
    h, w = a.shape[0] // f * f, a.shape[1] // f * f
    b = a[:h, :w].reshape(h // f, f, w // f, f)
    m = np.where(b >= 0, b, np.nan)
    with np.errstate(invalid="ignore"):
        return np.nanmean(m, axis=(1, 3))


def patch_gap_fraction(basin, rows, cols, half=112):
    """Share of each cell's 224 x 224 Sentinel-2 patch that is nodata.

    The composite declares nodata = 0 and the six HLS-equivalent bands share the
    mask exactly, so one band settles it. An integral image turns tens of
    thousands of window sums into two cumulative sums.
    """
    with rasterio.open(S2_COMPOSITE_DIR / f"{basin}_s2l2a_2023.tif") as src:
        z = (src.read(HLS_BAND) == 0).astype(np.float32)
    ii = np.pad(z, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    r0 = np.clip(rows - half, 0, z.shape[0]); r1 = np.clip(rows + half, 0, z.shape[0])
    c0 = np.clip(cols - half, 0, z.shape[1]); c1 = np.clip(cols + half, 0, z.shape[1])
    area = np.maximum((r1 - r0) * (c1 - c0), 1)
    return (ii[r1, c1] - ii[r0, c1] - ii[r1, c0] + ii[r0, c0]) / area


def load_inventory(basin):
    r, c, lab = [], [], []
    with open(RESULTS / "benchmark_points" / f"{basin}_points_uniform.csv") as f:
        for row in csv.DictReader(f):
            r.append(int(row["row"])); c.append(int(row["col"]))
            lab.append(int(row["label"]))
    r, c, lab = np.array(r), np.array(c), np.array(lab)
    return r[lab == 1], c[lab == 1]


def scalebar(ax, px_per_unit, km, label, n_cols):
    """Draw a scale bar of `km` kilometres in axes-fraction coordinates."""
    frac = (km * 1000 / 30 / px_per_unit) / n_cols
    x0, y0 = 0.06, 0.06
    ax.plot([x0, x0 + frac], [y0, y0], transform=ax.transAxes,
            color="black", lw=1.4, solid_capstyle="butt")
    ax.text(x0 + frac / 2, y0 + 0.03, label, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=6)


def load_basin(basin):
    with rasterio.open(MAPS / f"{basin}_susceptibility_A.tif") as src:
        A = src.read(1)
    with rasterio.open(MAPS / f"{basin}_susceptibility_PRITHVI_s{STRIDE}.tif") as src:
        P = src.read(1)
    grid = np.load(MAPS / f"{basin}_grid_s{STRIDE}.npz")
    gr, gc = grid["grid_rows"], grid["grid_cols"]
    half = STRIDE // 2

    # The baseline is averaged over exactly the STRIDE x STRIDE block each grid
    # cell represents, so the two surfaces are differenced at a common support.
    A_coarse = np.full(P.shape, -1.0, dtype=np.float32)
    for i, rr in enumerate(gr):
        block = A[max(0, rr - half):rr + half]
        for j, cc in enumerate(gc):
            w = block[:, max(0, cc - half):cc + half]
            ok = w[w >= 0]
            if ok.size:
                A_coarse[i, j] = ok.mean()

    gapf = patch_gap_fraction(basin, grid["rows"], grid["cols"])
    gap = np.full(P.shape, np.nan, dtype=np.float32)
    gap[np.searchsorted(gr, grid["rows"]), np.searchsorted(gc, grid["cols"])] = gapf
    gap_mask = gap > GAP_THRESHOLD

    valid = (P >= 0) & (A_coarse >= 0)
    diff = np.where(valid & ~gap_mask, P - A_coarse, np.nan)
    return A, P, A_coarse, gap_mask, diff, gapf


def main():
    data = {b: load_basin(b) for b, _, _ in BASINS}
    lim = max(float(np.nanpercentile(np.abs(d[4]), DIFF_CLIP)) for d in data.values())

    fig, axes = plt.subplots(3, 3, figsize=(FULL_W, 15.6 * CM),
                             constrained_layout=True)
    im_s = im_d = None
    for row, (slug, name, regime) in enumerate(BASINS):
        A, P, A_coarse, gap_mask, diff, gapf = data[slug]
        Ad = block_reduce(A, DISPLAY_FACTOR)
        Ad = np.where(Ad >= 0, Ad, np.nan)
        Pm = np.where(P >= 0, P, np.nan)
        pr, pc = load_inventory(slug)

        ax = axes[row][0]
        im_s = ax.imshow(Ad, cmap="magma_r", vmin=0, vmax=1, interpolation="nearest")
        ax.scatter(pc / DISPLAY_FACTOR, pr / DISPLAY_FACTOR, s=0.8, c=INV_COLOR,
                   linewidths=0, alpha=0.85)
        ax.set_ylabel(f"{name}\n({regime})", fontsize=8.5, fontweight="bold")
        scalebar(ax, DISPLAY_FACTOR, 50, "50 km", Ad.shape[1])

        ax = axes[row][1]
        ax.imshow(Pm, cmap="magma_r", vmin=0, vmax=1, interpolation="nearest")
        if np.any(gap_mask):
            ax.contour(gap_mask.astype(float), levels=[0.5],
                       colors=GAP_COLOR, linewidths=0.6)
        scalebar(ax, STRIDE, 50, "50 km", Pm.shape[1])

        ax = axes[row][2]
        im_d = ax.imshow(diff, cmap="RdBu_r", vmin=-lim, vmax=lim,
                         interpolation="nearest")
        scalebar(ax, STRIDE, 50, "50 km", diff.shape[1])

        for ax in axes[row]:
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.4)

    for col, title in enumerate(("(a) Baseline A, 30 m",
                                 "(b) Prithvi-EO-2.0, 600 m",
                                 "(c) Prithvi $-$ baseline")):
        axes[0][col].set_title(title, loc="left", fontsize=8.5, fontweight="bold")

    cb = fig.colorbar(im_s, ax=[axes[r][c] for r in range(3) for c in (0, 1)],
                      fraction=0.035, pad=0.01, location="bottom", shrink=0.55)
    cb.set_label("susceptibility", fontsize=8)
    cb2 = fig.colorbar(im_d, ax=[axes[r][2] for r in range(3)],
                       fraction=0.035, pad=0.01, location="bottom", shrink=0.9)
    cb2.set_label(f"Prithvi $-$ baseline (clipped at $\\pm${lim:.2f})", fontsize=8)

    handles = [Line2D([], [], marker="o", ls="", ms=3, color=INV_COLOR,
                      label="inventoried landslide"),
               Line2D([], [], color=GAP_COLOR, lw=1.0,
                      label="patch $>80\\%$ Sentinel-2 NoData")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.02))

    for ext in ("pdf", "png"):
        p = OUT_DIR / f"fig7_susceptibility_maps.{ext}"
        fig.savefig(p, dpi=400 if ext == "png" else None, bbox_inches="tight")
        print(f"  wrote {p.name}")
    plt.close(fig)

    print(f"\n[stats] difference colour range clipped at +/-{lim:.3f} "
          f"({DIFF_CLIP}th percentile of |difference|)")
    for slug, name, _ in BASINS:
        A, P, A_coarse, gap_mask, diff, gapf = data[slug]
        ok = np.isfinite(diff)
        n_gap = int(np.nansum(gap_mask))
        print(f"\n[{name}] grid cells {len(gapf):,}; gap cells {n_gap:,} "
              f"({n_gap / len(gapf) * 100:.0f}%)")
        if n_gap:
            print(f"[{name}] mean Prithvi in gaps {np.nanmean(P[gap_mask]):.3f} "
                  f"vs {np.nanmean(P[~gap_mask & (P >= 0)]):.3f} where imagery present")
        print(f"[{name}] compared {ok.sum():,}; mean diff {np.nanmean(diff):+.3f}; "
              f"Prithvi higher in {(diff[ok] > 0).mean() * 100:.0f}%")
        for lo, hi, lab in ((0.0, 0.2, "low"), (0.2, 0.5, "moderate"),
                            (0.5, 1.01, "high")):
            m = ok & (A_coarse >= lo) & (A_coarse < hi)
            if m.sum():
                print(f"[{name}]   baseline {lab:<9} ({m.sum():>6,} cells): "
                      f"mean diff {np.nanmean(diff[m]):+.3f}")


if __name__ == "__main__":
    main()
