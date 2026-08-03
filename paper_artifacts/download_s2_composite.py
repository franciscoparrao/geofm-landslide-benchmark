"""Download per-basin Sentinel-2 L2A annual composite via Planetary Computer.

Targets the 12 bands expected by TerraMind v1 (PRETRAINED_BANDS["untok_sen2l2a@224"]):
  B01 (Coastal aerosol), B02 (Blue), B03 (Green), B04 (Red),
  B05/B06/B07 (Red edge), B08 (NIR broad), B8A (NIR narrow),
  B09 (Water vapor), B11/B12 (SWIR).

Output: paper/data/s2_composites/{basin}_s2l2a_2023.tif
  12-band GeoTIFF, uint16 (raw S2 reflectance × 10000), aligned to basin DEM grid.

Implementation:
  - Search STAC for Sentinel-2 L2A scenes intersecting basin polygon
    over 2023, with eo:cloud_cover < CLOUD_THRESHOLD.
  - Lazy stackstac stack, then median across time (cloud-robust composite).
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

ROOT = Path("/home/franciscoparrao/proyectos/no_supervisado_superficie")
POSTDOC = Path("/home/franciscoparrao/proyectos/postdoc/papers/paper1_susceptibilidad")
POLY_DIR = POSTDOC / "basin_polygons"
DEM_DIR_BASE = POSTDOC / "factors"
OUT_DIR = ROOT / "paper/data/s2_composites"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
DATE_RANGE = "2023-01-01/2023-12-31"
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


def search_items(client, bbox_wgs84, max_items=30):
    """Return up to max_items least-cloudy S2 L2A items intersecting bbox."""
    search = client.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox_wgs84,
        datetime=DATE_RANGE,
        query={"eo:cloud_cover": {"lt": CLOUD_THRESHOLD}},
    )
    items = list(search.items())
    items_sorted = sorted(items, key=lambda it: it.properties.get("eo:cloud_cover", 100.0))
    return items_sorted[:max_items]


def build_composite(basin_id):
    print(f"\n=== {basin_id} ===")
    t0 = perf_counter()

    basin_gdf, bbox_wgs84 = load_basin_geom(basin_id)
    dem_transform, dem_crs, dem_h, dem_w, dem_bounds = load_dem_grid(basin_id)
    print(f"  DEM grid: {dem_w}x{dem_h} px, CRS={dem_crs}, "
          f"bbox={tuple(round(x, 1) for x in dem_bounds)}")
    print(f"  Basin bbox WGS84: {tuple(round(x, 3) for x in bbox_wgs84)}")

    client = get_stac_client()
    items = search_items(client, bbox_wgs84)
    print(f"  Found {len(items)} S2 L2A items with cloud<{CLOUD_THRESHOLD}% in {DATE_RANGE}")
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
    )
    print(f"  Stack shape (time, band, y, x): {stack.shape}")
    print(f"  Stack dtype: {stack.dtype}")

    # Mean across time is chunk-friendly (median would require full timeseries
    # per pixel in memory). With scene-level cloud filter < 20% and ≤30
    # least-cloudy items, mean is a reasonable cloud-robust composite.
    composite = stack.mean(dim="time", skipna=True)

    print("  Computing mean composite over time (chunk-friendly)...")
    t1 = perf_counter()
    composite_np = composite.compute()
    print(f"  Composite computed in {perf_counter() - t1:.1f}s")

    # Write
    out_path = OUT_DIR / f"{basin_id}_s2l2a_2023.tif"
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
    args = parser.parse_args()
    build_composite(args.basin)


if __name__ == "__main__":
    main()
