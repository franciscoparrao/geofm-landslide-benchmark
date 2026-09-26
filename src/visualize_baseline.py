"""Side-by-side maps: DEM hillshade, slope, and cluster maps for K=5,10,15."""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import LightSource, ListedColormap

from config import BASINS, DEFAULT_BASIN, RESULTS, basin_dir

DOWNSAMPLE = 4
K_VALUES = (5, 10, 15)


def downsampled_read(path, factor=DOWNSAMPLE):
    with rasterio.open(path) as src:
        h = src.height // factor
        w = src.width // factor
        arr = src.read(1, out_shape=(h, w), masked=True)
        nodata = src.nodata
    return arr, nodata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--basin", default=DEFAULT_BASIN, choices=BASINS)
    args = parser.parse_args()
    basin = args.basin
    bd = basin_dir(basin)

    dem, _ = downsampled_read(bd / "dem_30m.tif")
    slope, _ = downsampled_read(bd / "terrain" / "slope.tif")

    cluster_rasters = {}
    for k in K_VALUES:
        cl, _ = downsampled_read(RESULTS / f"{basin}_kmeans_K{k:02d}.tif")
        arr = np.ma.getdata(cl).astype(np.float32)
        arr[arr == -1] = np.nan
        if np.ma.is_masked(cl):
            arr[np.ma.getmaskarray(cl)] = np.nan
        cluster_rasters[k] = arr

    fig, axes = plt.subplots(2, 3, figsize=(18, 12), constrained_layout=True)

    ls = LightSource(azdeg=315, altdeg=45)
    dem_filled = dem.filled(np.nan)
    finite = np.isfinite(dem_filled)
    if finite.any():
        shade = ls.hillshade(np.where(finite, dem_filled, 0), vert_exag=2)
        shade = np.where(finite, shade, np.nan)
    else:
        shade = dem_filled
    ax = axes[0, 0]
    im = ax.imshow(dem_filled, cmap="terrain")
    ax.imshow(shade, cmap="gray", alpha=0.35)
    ax.set_title(f"DEM 30m + hillshade — {basin}")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.04, label="elev (m)")

    ax = axes[0, 1]
    slope_filled = slope.filled(np.nan)
    im = ax.imshow(slope_filled, cmap="magma", vmin=0, vmax=np.nanpercentile(slope_filled, 98))
    ax.set_title("Slope (deg)")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.04, label="deg")

    axes[0, 2].axis("off")

    panels = [(1, 0, 5), (1, 1, 10), (1, 2, 15)]
    for r, c, k in panels:
        ax = axes[r, c]
        cmap = ListedColormap(plt.colormaps["tab20"](np.linspace(0, 1, k)))
        ax.imshow(cluster_rasters[k], cmap=cmap, vmin=-0.5, vmax=k - 0.5,
                  interpolation="nearest")
        ax.set_title(f"K-means K={k}")
        ax.axis("off")

    fig.suptitle(f"PCA(10)+MiniBatchKMeans baseline — {basin}", fontsize=14)
    out = RESULTS / f"{basin}_baseline_overview.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[viz] wrote {out}")


if __name__ == "__main__":
    main()
