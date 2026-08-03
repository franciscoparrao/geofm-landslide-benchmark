"""Build feature-pixel matrix for one basin.

Reads numeric terrain + hydrology layers, verifies raster alignment, builds a
masked feature-pixel matrix and persists it together with georeferencing
metadata so downstream clustering can rasterize results back.
"""
from __future__ import annotations

import argparse

import numpy as np
import rasterio
from rasterio.errors import RasterioIOError

from config import (
    BASINS,
    DEFAULT_BASIN,
    LOG_TRANSFORM,
    RESULTS,
    feature_names,
    feature_paths,
)


def load_layer(path):
    with rasterio.open(path) as src:
        data = src.read(1, masked=True)
        profile = {
            "shape": (src.height, src.width),
            "transform": src.transform,
            "crs": src.crs,
            "nodata": src.nodata,
        }
    return data, profile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--basin", default=DEFAULT_BASIN, choices=BASINS)
    args = parser.parse_args()
    basin = args.basin

    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / f"{basin}_stack.npz"
    paths = feature_paths(basin)
    names = feature_names(basin)

    print(f"[build_stack] basin={basin} features={len(names)}")

    layers = {}
    reference_profile = None
    for name in names:
        path = paths[name]
        try:
            data, profile = load_layer(path)
        except RasterioIOError as exc:
            raise SystemExit(f"failed to read {path}: {exc}")

        if reference_profile is None:
            reference_profile = profile
            print(
                f"  reference shape={profile['shape']} crs={profile['crs']} "
                f"transform={profile['transform']!r}"
            )
        else:
            if profile["shape"] != reference_profile["shape"]:
                raise SystemExit(
                    f"shape mismatch {name}: {profile['shape']} vs "
                    f"{reference_profile['shape']}"
                )
            if profile["transform"] != reference_profile["transform"]:
                raise SystemExit(f"transform mismatch on {name}")
            if profile["crs"] != reference_profile["crs"]:
                print(
                    f"  WARN CRS mismatch on {name}: {profile['crs']} "
                    f"(grid aligned, continuing)"
                )

        if name in LOG_TRANSFORM:
            arr = data.filled(np.nan).astype(np.float32)
            arr = np.where(np.isfinite(arr) & (arr >= 0), np.log1p(arr), np.nan)
            data = np.ma.masked_invalid(arr)

        layers[name] = data
        finite = int(np.isfinite(data.filled(np.nan)).sum())
        print(f"  loaded {name:<24} valid_pixels={finite}")

    height, width = reference_profile["shape"]
    n_pixels = height * width

    valid_mask = np.ones(n_pixels, dtype=bool)
    feature_arrays = []
    for name in names:
        arr = layers[name].filled(np.nan).astype(np.float32).reshape(-1)
        valid_mask &= np.isfinite(arr)
        feature_arrays.append(arr)

    n_valid = int(valid_mask.sum())
    print(
        f"[build_stack] valid_pixels={n_valid:,} / {n_pixels:,} "
        f"({100 * n_valid / n_pixels:.1f}%)"
    )
    if n_valid == 0:
        raise SystemExit("no valid pixels after masking — check nodata handling")

    X = np.column_stack([arr[valid_mask] for arr in feature_arrays]).astype(np.float32)
    valid_idx = np.flatnonzero(valid_mask).astype(np.int64)

    transform = reference_profile["transform"]
    transform_arr = np.array(
        [transform.a, transform.b, transform.c, transform.d, transform.e, transform.f],
        dtype=np.float64,
    )
    crs_wkt = reference_profile["crs"].to_wkt() if reference_profile["crs"] else ""

    np.savez_compressed(
        out_path,
        X=X,
        valid_idx=valid_idx,
        feature_names=np.array(names),
        height=np.int64(height),
        width=np.int64(width),
        transform=transform_arr,
        crs_wkt=np.array(crs_wkt),
    )
    size_mb = out_path.stat().st_size / 1024 ** 2
    print(f"[build_stack] wrote {out_path} ({size_mb:.1f} MB) X.shape={X.shape}")


if __name__ == "__main__":
    main()
