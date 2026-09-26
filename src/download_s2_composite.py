"""Download per-basin Sentinel-2 L2A annual composite via Planetary Computer.

Targets the 12 bands expected by TerraMind v1 (PRETRAINED_BANDS["untok_sen2l2a@224"]):
  B01 (Coastal aerosol), B02 (Blue), B03 (Green), B04 (Red),
  B05/B06/B07 (Red edge), B08 (NIR broad), B8A (NIR narrow),
  B09 (Water vapor), B11/B12 (SWIR).

Output: paper/data/s2_composites/{basin}_s2l2a_{year}.tif
  12-band GeoTIFF, uint16 (raw S2 reflectance × 10000), aligned to basin DEM grid.

Implementation:
  - Search STAC for Sentinel-2 L2A scenes intersecting basin polygon
    over 2023, with eo:cloud_cover < CLOUD_THRESHOLD.
  - Lazy stackstac stack, then mean across time (see note below: median needs the full timeseries in memory).
  - Reproject/clip to match the basin DEM grid exactly (same CRS, transform).

This script reuses the basin_polygons of the postdoc paper1 pipeline
(input shared, not output coupled).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import planetary_computer as pc
import pystac_client
import rasterio
import stackstac
import xarray as xr
from rasterio.warp import Resampling, calculate_default_transform, reproject

from paths import BASIN_POLY_DIR as POLY_DIR
from paths import BASIN_DATA_DIR as DEM_DIR_BASE
from paths import S2_COMPOSITE_DIR as OUT_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
DEFAULT_YEAR = 2023
CLOUD_THRESHOLD = 20

# TerraMind expects S2L2A in this band order (PRETRAINED_BANDS)
S2_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07",
            "B08", "B8A", "B09", "B11", "B12"]


def get_stac_client():
    return pystac_client.Client.open(STAC_URL, modifier=pc.sign_inplace)


def load_basin_geom(basin_id):
    """Return (basin_gdf, basin_bbox_wgs84) for STAC query.

    Use bbox in WGS84 for STAC intersects (less precise but robust to
    numpy 2.x / geopandas to_json incompat); the per-pixel clipping is
    done later via reproject to the DEM grid.
    """
    import geopandas as gpd
    gdf = gpd.read_file(POLY_DIR / f"{basin_id}.geojson")
    gdf_wgs84 = gdf.to_crs("EPSG:4326") if gdf.crs.to_epsg() != 4326 else gdf
    total_bounds = gdf_wgs84.total_bounds  # (minx, miny, maxx, maxy)
    return gdf, tuple(float(v) for v in total_bounds)


def load_dem_grid(basin_id):
    """Return (transform, crs, height, width) of basin DEM (target grid)."""
    dem_path = DEM_DIR_BASE / basin_id / "dem_30m.tif"
    with rasterio.open(dem_path) as src:
        return src.transform, src.crs, src.height, src.width, src.bounds


def search_items(client, bbox_wgs84, max_items=30, year=DEFAULT_YEAR, months=None,
                 per_tile=True):
    """Return least-cloudy S2 L2A items intersecting bbox, balanced across tiles.

    `months` restricts acquisitions to a (first, last) inclusive month range, so
    two composites from different years can be matched seasonally.

    `per_tile` takes max_items from EACH Sentinel-2 MGRS tile rather than
    max_items from the basin as a whole. Taking them globally is a bug when a
    basin spans several tiles, because eo:cloud_cover is a per-scene property and
    the clearest scenes concentrate in the driest tiles: sorting the pooled list
    and truncating it leaves entire tiles with no contributing scene, and the
    temporal mean is then empty across them. In Maule this emptied four of
    eleven tiles, one of which (19HCA) had 150 qualifying scenes and contributed
    none, removing 27% of the basin along straight MGRS boundaries. Balancing the
    draw per tile costs nothing and is what the composite was meant to do.
    """
    search = client.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox_wgs84,
        datetime=f"{year}-01-01/{year}-12-31",
        query={"eo:cloud_cover": {"lt": CLOUD_THRESHOLD}},
    )
    items = list(search.items())
    if months:
        lo, hi = months
        items = [it for it in items if lo <= it.datetime.month <= hi]
    cloud = lambda it: it.properties.get("eo:cloud_cover", 100.0)
    if not per_tile:
        return sorted(items, key=cloud)[:max_items]

    by_tile = {}
    for it in items:
        by_tile.setdefault(it.properties.get("s2:mgrs_tile", "?"), []).append(it)
    out = []
    for tile in sorted(by_tile):
        picked = sorted(by_tile[tile], key=cloud)[:max_items]
        out.extend(picked)
        print(f"    tile {tile}: {len(by_tile[tile]):>4} qualifying, {len(picked):>3} used "
              f"(cloud {cloud(picked[0]):.1f}-{cloud(picked[-1]):.1f}%)")
    return out


def baseline_offset(item):
    """BOA_ADD_OFFSET introduced by S2 processing baseline 04.00 (Jan 2022).

    Products from baseline >= 04.00 carry DN = reflectance*10000 + 1000, while
    earlier products (and the HLS convention Prithvi was pretrained on) do not.
    Compositing across the boundary without correcting this shifts radiometry by
    a constant 1000 DN.
    """
    try:
        pb = float(str(item.properties.get("s2:processing_baseline", "0")))
    except ValueError:
        pb = 0.0
    return 1000.0 if pb >= 4.0 else 0.0


def build_composite(basin_id, year=DEFAULT_YEAR, months=None,
                    max_items=30, harmonize=False, suffix="", per_tile=True,
                    stack_dtype="float64"):
    """Build one per-basin annual composite.

    stack_dtype bounds peak memory: the stack is (time, band, y, x) and float32
    halves it against the stackstac default. The output is uint16 either way,
    and the mean of at most a few dozen values below 2**14 DN is exact in
    float32 (integers are exact to 2**24, and the division costs at most one
    ulp, about 0.01 DN), so the written product is unchanged. Basins spanning
    many MGRS tiles need it: Maule draws from eleven tiles and does not fit in
    the headroom this machine has under float64.
    """
    print(f"\n=== {basin_id} ===")
    t0 = perf_counter()

    basin_gdf, bbox_wgs84 = load_basin_geom(basin_id)
    dem_transform, dem_crs, dem_h, dem_w, dem_bounds = load_dem_grid(basin_id)
    print(f"  DEM grid: {dem_w}x{dem_h} px, CRS={dem_crs}, "
          f"bbox={tuple(round(x, 1) for x in dem_bounds)}")
    print(f"  Basin bbox WGS84: {tuple(round(x, 3) for x in bbox_wgs84)}")

    client = get_stac_client()
    items = search_items(client, bbox_wgs84, max_items=max_items, year=year,
                         months=months, per_tile=per_tile)
    n_tiles = len({it.properties.get("s2:mgrs_tile", "?") for it in items})
    print(f"  Using {len(items)} S2 L2A items with cloud<{CLOUD_THRESHOLD}% in {year} "
          f"across {n_tiles} MGRS tiles")
    if not items:
        raise SystemExit("No items found")

    # Build stack — use stackstac at the basin's DEM CRS/grid directly
    bbox = (dem_bounds.left, dem_bounds.bottom, dem_bounds.right, dem_bounds.top)
    epsg = dem_crs.to_epsg()
    stack = stackstac.stack(
        items,
        epsg=epsg,
        bounds=bbox,
        resolution=30,
        assets=S2_BANDS,
        chunksize=1024,
        rescale=False,    # keep raw S2 reflectance scale (×10000)
        dtype=stack_dtype,
        # stackstac validates with np.can_cast(type(fill_value), dtype), which
        # inspects the Python type rather than the array dtype: a plain float NaN
        # is float64 and a 0-d array is ndarray, and both fail against float32.
        # It has to be a numpy scalar of the stack's own dtype.
        fill_value=np.dtype(stack_dtype).type(np.nan),
    )
    print(f"  Stack shape (time, band, y, x): {stack.shape}")
    print(f"  Stack dtype: {stack.dtype}")

    if harmonize:
        import xarray as _xr
        offsets = np.array([baseline_offset(it) for it in items], dtype="float64")
        n_off = int((offsets > 0).sum())
        print(f"  Harmonizing radiometry: subtracting BOA_ADD_OFFSET from "
              f"{n_off}/{len(items)} items (baseline >= 04.00)")
        off_da = _xr.DataArray(offsets, dims=["time"], coords={"time": stack.time})
        stack = (stack - off_da).clip(min=0)

    # Mean across time is chunk-friendly (median would require full timeseries
    # per pixel in memory). With scene-level cloud filter < 20% and ≤30
    # least-cloudy items, mean is a reasonable cloud-robust composite.
    composite = stack.mean(dim="time", skipna=True)

    print("  Computing mean composite over time (chunk-friendly)...")
    t1 = perf_counter()
    composite_np = composite.compute()
    print(f"  Composite computed in {perf_counter() - t1:.1f}s")

    # Write
    out_path = OUT_DIR / f"{basin_id}_s2l2a_{year}{suffix}.tif"
    profile = {
        "driver": "GTiff",
        "height": composite_np.shape[1],
        "width": composite_np.shape[2],
        "count": len(S2_BANDS),
        "dtype": "uint16",
        "crs": dem_crs,
        "transform": composite_np.transform if hasattr(composite_np, "transform")
                     else dem_transform,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "nodata": 0,
    }
    # stackstac returns DataArray with transform attribute via x/y coords
    # Build transform from coords
    x_coords = composite_np.x.values
    y_coords = composite_np.y.values
    res_x = abs(x_coords[1] - x_coords[0])
    res_y = abs(y_coords[1] - y_coords[0])
    from rasterio.transform import from_origin
    transform = from_origin(
        x_coords[0] - res_x / 2,
        y_coords[0] + res_y / 2,
        res_x, res_y,
    )
    profile["transform"] = transform
    profile["height"] = composite_np.shape[1]
    profile["width"] = composite_np.shape[2]

    print(f"  Writing {out_path.name} ({profile['width']}x{profile['height']}, "
          f"{profile['count']} bands)")
    with rasterio.open(out_path, "w", **profile) as dst:
        for i in range(len(S2_BANDS)):
            band_data = composite_np.values[i]
            band_data = np.nan_to_num(band_data, nan=0).astype("uint16")
            dst.write(band_data, i + 1)
            dst.set_band_description(i + 1, S2_BANDS[i])

    size_mb = out_path.stat().st_size / (1024 ** 2)
    print(f"  ✓ {out_path.name} ({size_mb:.1f} MB) elapsed={perf_counter() - t0:.1f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--basin", default="06_rio_huasco")
    parser.add_argument("--year", type=int, default=DEFAULT_YEAR,
                        help="Calendar year of the composite (e.g. 2016 for a "
                             "pre-event composite in Huasco).")
    parser.add_argument("--months", default=None,
                        help="Inclusive month range 'lo-hi' (e.g. '1-6') to match "
                             "two composites seasonally.")
    parser.add_argument("--max-items", type=int, default=30)
    parser.add_argument("--float32", action="store_true",
                        help="build the stack in float32 to halve peak memory; "
                             "the uint16 output is unchanged")
    parser.add_argument("--global-draw", action="store_true",
                        help="reproduce the pre-fix behaviour: take max-items from the\n                              basin as a whole instead of from each MGRS tile")
    parser.add_argument("--harmonize", action="store_true",
                        help="Subtract the baseline-04.00 BOA_ADD_OFFSET so that "
                             "pre- and post-2022 acquisitions share a radiometric scale.")
    parser.add_argument("--suffix", default="",
                        help="Appended to the output filename.")
    args = parser.parse_args()
    months = tuple(int(x) for x in args.months.split("-")) if args.months else None
    build_composite(args.basin, year=args.year, months=months,
                    max_items=args.max_items, harmonize=args.harmonize,
                    per_tile=not args.global_draw,
                    stack_dtype="float32" if args.float32 else "float64",
                    suffix=args.suffix)


if __name__ == "__main__":
    main()
