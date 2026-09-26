"""Build one composite in scene batches, re-signing the STAC tokens per batch.

Why this exists: Planetary Computer's SAS tokens live about seventy minutes and
are stamped into the asset URLs when stackstac builds the stack. Huasco's
season-matched 2023 draw is 180 scenes across six MGRS tiles, and the mean takes
longer than that, so the single-shot path dies partway through with HTTP 403.
Parallelism does not help: at four, eight and twelve dask workers the run failed
the same way, and CPU never rose above 191% of the 1200% available, because the
cost is per-chunk I/O latency.

Splitting the raster into row blocks does not help either, and this was measured
rather than assumed: each block re-opens all 312 of the 2016 draw's files and
pays the full open overhead, so one quarter of the rows cost 515s and 629s
against 541s for the whole thing in one pass. The work does not divide along
space.

It divides along time. A skipna mean is a sum over the non-null values divided
by their count, and both are associative over the scene axis:

    mean = (Σ_batches Σ_scenes x) / (Σ_batches count)

So each batch opens only its own scenes, the total read is the same as one pass,
and every compute() sits well inside the token lifetime. The scene set is pinned
by ID from the first search and re-checked before each batch, so a re-search
cannot silently change which scenes contribute.

Everything else is download_s2_composite.py unchanged: same per-tile draw, same
stack parameters, same harmonisation, same nan->0 uint16 cast. The sums are
exact in float32 here (the values are integers below 2**14 DN and at most a few
hundred scenes contribute, so the total stays far below the 2**24 where float32
stops representing integers exactly), so the product is identical to what the
single-shot path would have written had it finished.

Usage mirrors the original, plus --batches:

    python3 composite_batched.py --basin 06_rio_huasco --year 2023 \
        --months 1-6 --harmonize --suffix _matched --batches 4
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import io
from time import perf_counter

import numpy as np
import rasterio
import stackstac
import xarray as xr
from rasterio.transform import from_origin

from download_s2_composite import (
    CLOUD_THRESHOLD,
    OUT_DIR,
    S2_BANDS,
    baseline_offset,
    get_stac_client,
    load_basin_geom,
    load_dem_grid,
    search_items,
)


def _pin_hash(ids):
    """Identify a scene set, so a checkpoint cannot be reused across draws."""
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()[:16]


def _draw(bbox_wgs84, max_items, year, months, per_tile, quiet=False):
    """Run the STAC search and return freshly signed items."""
    client = get_stac_client()
    if quiet:
        with contextlib.redirect_stdout(io.StringIO()):
            return search_items(client, bbox_wgs84, max_items=max_items,
                                year=year, months=months, per_tile=per_tile)
    return search_items(client, bbox_wgs84, max_items=max_items, year=year,
                        months=months, per_tile=per_tile)


def _stack(items, dem_bounds, epsg, stack_dtype):
    bbox = (dem_bounds.left, dem_bounds.bottom, dem_bounds.right, dem_bounds.top)
    return stackstac.stack(
        items, epsg=epsg, bounds=bbox, resolution=30, assets=S2_BANDS,
        chunksize=1024, rescale=False, dtype=stack_dtype,
        fill_value=np.dtype(stack_dtype).type(np.nan),
    )


def build_batched(basin_id, year, months, max_items, harmonize, suffix,
                  per_tile, stack_dtype, batches):
    print(f"\n=== {basin_id} (batched, {batches} scene batches) ===")
    t0 = perf_counter()

    _gdf, bbox_wgs84 = load_basin_geom(basin_id)
    _dem_transform, dem_crs, dem_h, dem_w, dem_bounds = load_dem_grid(basin_id)
    print(f"  DEM grid: {dem_w}x{dem_h} px, CRS={dem_crs}")

    items = _draw(bbox_wgs84, max_items, year, months, per_tile)
    pinned = [it.id for it in items]
    n_tiles = len({it.properties.get("s2:mgrs_tile", "?") for it in items})
    print(f"  Using {len(items)} S2 L2A items with cloud<{CLOUD_THRESHOLD}% in "
          f"{year} across {n_tiles} MGRS tiles")
    if not items:
        raise SystemExit("No items found")

    probe = _stack(items, dem_bounds, dem_crs.to_epsg(), stack_dtype)
    x_coords, y_coords = probe.x.values, probe.y.values
    height, width = probe.sizes["y"], probe.sizes["x"]
    res_x = abs(x_coords[1] - x_coords[0])
    res_y = abs(y_coords[1] - y_coords[0])
    transform = from_origin(x_coords[0] - res_x / 2, y_coords[0] + res_y / 2,
                            res_x, res_y)
    del probe

    n_bands = len(S2_BANDS)
    total_sum = np.zeros((n_bands, height, width), dtype="float32")
    total_cnt = np.zeros((n_bands, height, width), dtype="int16")

    groups = [list(g) for g in np.array_split(np.arange(len(pinned)), batches)]
    print(f"  Accumulating over {batches} batches of "
          f"{[len(g) for g in groups]} scenes")

    ckpt = OUT_DIR / f".{basin_id}_s2l2a_{year}{suffix}_ckpt.npz"
    first = 0
    if ckpt.exists():
        with np.load(ckpt) as z:
            if int(z["batches"]) == batches and str(z["pinned_hash"]) == _pin_hash(pinned):
                total_sum = z["s"].astype("float32")
                total_cnt = z["c"].astype("int16")
                first = int(z["next"])
                print(f"  Resuming from checkpoint at batch {first + 1}/{batches}")
            else:
                print("  Ignoring checkpoint: it describes a different run")

    for b, idx in enumerate(groups):
        if b < first:
            continue
        tb = perf_counter()
        # Transient reads fail often enough over a multi-hour run (and GDAL's
        # own retries do not cover every case), so a failed batch is retried
        # with freshly signed tokens rather than losing the whole composite.
        for attempt in range(1, 4):
            try:
                items_b = _draw(bbox_wgs84, max_items, year, months, per_tile,
                                quiet=True)
                if [it.id for it in items_b] != pinned:
                    raise SystemExit(
                        f"batch {b}: STAC re-search returned a different scene "
                        "set; aborting rather than mixing draws")

                subset = [items_b[i] for i in idx]
                sub = _stack(subset, dem_bounds, dem_crs.to_epsg(), stack_dtype)
                if harmonize:
                    offsets = np.array([baseline_offset(it) for it in subset],
                                       dtype="float64")
                    off_da = xr.DataArray(offsets, dims=["time"],
                                          coords={"time": sub.time})
                    sub = (sub - off_da).clip(min=0)

                # One compute() for both reductions: computing them separately
                # makes dask read every scene twice, which doubled the batch
                # time in testing.
                part = xr.Dataset({"s": sub.sum(dim="time", skipna=True),
                                   "c": sub.count(dim="time")}).compute()
                break
            except SystemExit:
                raise
            except Exception as exc:
                if attempt == 3:
                    raise
                print(f"    batch {b + 1}: attempt {attempt} failed "
                      f"({type(exc).__name__}), retrying with fresh tokens")

        total_sum += part["s"].values.astype("float32")
        total_cnt += part["c"].values.astype("int16")
        del part, sub
        # Batches die on the second one without this: dask's graph and the
        # per-batch arrays are both large, and CPython does not return the
        # arenas promptly enough on its own.
        gc.collect()
        np.savez(ckpt, s=total_sum, c=total_cnt, next=b + 1, batches=batches,
                 pinned_hash=_pin_hash(pinned))
        print(f"    batch {b + 1}/{batches}: {len(idx)} scenes "
              f"({perf_counter() - tb:.0f}s)")

    with np.errstate(invalid="ignore", divide="ignore"):
        composite = np.where(total_cnt > 0, total_sum / total_cnt, np.nan)

    out_path = OUT_DIR / f"{basin_id}_s2l2a_{year}{suffix}.tif"
    profile = {
        "driver": "GTiff", "height": height, "width": width, "count": n_bands,
        "dtype": "uint16", "crs": dem_crs, "transform": transform,
        "compress": "deflate", "tiled": True, "blockxsize": 512,
        "blockysize": 512, "nodata": 0,
    }
    print(f"  Writing {out_path.name} ({width}x{height}, {n_bands} bands)")
    with rasterio.open(out_path, "w", **profile) as dst:
        for i in range(n_bands):
            band = np.nan_to_num(composite[i], nan=0).astype("uint16")
            dst.write(band, i + 1)
            dst.set_band_description(i + 1, S2_BANDS[i])

    ckpt.unlink(missing_ok=True)
    size_mb = out_path.stat().st_size / (1024 ** 2)
    print(f"  ✓ {out_path.name} ({size_mb:.1f} MB) "
          f"elapsed={perf_counter() - t0:.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--basin", default="06_rio_huasco")
    ap.add_argument("--year", type=int, default=2023)
    ap.add_argument("--months", default=None)
    ap.add_argument("--max-items", type=int, default=30)
    ap.add_argument("--harmonize", action="store_true")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--global-draw", action="store_true")
    ap.add_argument("--float64", action="store_true",
                    help="build the stack in float64; float32 is the default "
                         "here because this path exists for large draws")
    ap.add_argument("--batches", type=int, default=4,
                    help="number of scene batches; each one re-signs the tokens")
    a = ap.parse_args()
    months = tuple(int(x) for x in a.months.split("-")) if a.months else None
    build_batched(a.basin, a.year, months, a.max_items, a.harmonize, a.suffix,
                  not a.global_draw, "float64" if a.float64 else "float32",
                  a.batches)


if __name__ == "__main__":
    main()
