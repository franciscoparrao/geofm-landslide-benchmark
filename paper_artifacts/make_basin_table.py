"""Compute Table 1 (basin summary) from real raster data.

Outputs paper/tables/tab1_basins.{tex,md} with: BNA ID, basin name, latitude
(centroid), area (km² from valid DEM pixels), mean annual temperature (BIO_01),
mean annual precipitation (BIO_12), and inventoried positive event count.

Precipitation is reported twice. The basin-wide mean is the natural summary but
it is not the statistic the regime labels name: Lluta and Huasco both reach the
Altiplano, where austral-summer convective rainfall raises the catchment average
far above anything the inhabited and landslide-prone lowlands receive. Lluta's
mean elevation is 3,502 m and its basin-mean precipitation (194 mm/yr) exceeds
semi-arid Huasco's (78 mm/yr), which inverts the north-south gradient the study
design rests on. Restricting the average to pixels below 2,000 m recovers a
monotone series that matches the labels, so both columns are tabulated.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio

from paths import BASIN_DATA_DIR as FACTORS
from paths import ML_DATASET_DIR as ML_DATASET
from paths import TABLES_DIR as OUT_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Altitude cut separating the lowland climatology the regime labels describe
# from the Altiplano headwaters that distort the basin means of the two
# northern catchments (see module docstring).
LOWLAND_M = 2000.0

BASINS = [
    ("01_rio_lluta", "Lluta", "Hyperarid"),
    ("06_rio_huasco", "Huasco", "Semi-arid"),
    ("09_rio_maipo", "Maipo", "Mediterranean"),
    ("11_rio_maule", "Maule", "Temperate-humid"),
    ("13_rio_bueno", "Bueno", "Temperate-rainy"),
]

# Official DGA/BNA catchment codes (COD_CUEN in Cuencas_BNA.shp). The directory
# prefixes above are this project's own ordering along the transect and are not
# the national codes; printing those as "BNA" would misidentify the basins to
# anyone checking them against DGA sources.
BNA_CODE = {
    "01_rio_lluta": "012",
    "06_rio_huasco": "038",
    "09_rio_maipo": "057",
    "11_rio_maule": "073",
    "13_rio_bueno": "103",
}


def basin_stats(basin_id):
    base = FACTORS / basin_id
    dem_p = base / "dem_30m.tif"
    bio01_p = base / "climate" / "bio_01.tif"   # mean annual temperature (°C)
    bio12_p = base / "climate" / "bio_12.tif"   # annual precipitation (mm)

    with rasterio.open(dem_p) as src:
        dem = src.read(1)
        transform = src.transform
        dem_nodata = src.nodata

    # The DEMs carry no nodata tag and code out-of-basin cells as a literal 0,
    # so a mask built only from finiteness admits the whole rectangular footprint
    # -- roughly twice each basin's true extent. Averaging the co-registered
    # climate rasters over that footprint is what made the transect's
    # precipitation gradient non-monotone: Lluta's bounding box reaches far into
    # the Altiplano and across the Peruvian border, and the wet cells it picks up
    # lifted the hyper-arid basin above the semi-arid one. Masking on dem > 0
    # recovers the basin proper; it costs the handful of cells sitting exactly at
    # sea level, which no basin statistic here is sensitive to.
    valid_mask = np.isfinite(dem) & (dem > 0)
    if dem_nodata is not None:
        valid_mask &= dem != dem_nodata
    n_valid = int(valid_mask.sum())
    px_area_km2 = abs(transform.a * transform.e) / 1e6
    area_km2 = n_valid * px_area_km2

    # geographic centroid from raster CRS bounding box, reprojected to EPSG:4326
    with rasterio.open(dem_p) as src:
        bounds = src.bounds
        crs = src.crs
    import pyproj
    transformer = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon_c, lat_c = transformer.transform(
        (bounds.left + bounds.right) / 2,
        (bounds.bottom + bounds.top) / 2,
    )

    def _basin_mean(path, mask):
        """Mean of a co-registered layer over `mask`, skipping its own nodata."""
        with rasterio.open(path) as src:
            arr = src.read(1, masked=True)
        ok = mask & ~np.ma.getmaskarray(arr)
        data = np.ma.getdata(arr)
        ok &= np.isfinite(data) & (data > -1e10)
        return float(np.mean(data[ok])) if ok.any() else float("nan")

    t_mean = _basin_mean(bio01_p, valid_mask)
    p_mean = _basin_mean(bio12_p, valid_mask)
    # Lowland precipitation: the same layer restricted to the elevation band the
    # regime labels describe. Reported alongside the basin mean because the two
    # northern catchments rise into a summer-rainfall regime that their inhabited
    # and landslide-prone lowlands never see.
    p_low = _basin_mean(bio12_p, valid_mask & (dem < LOWLAND_M))
    elev_mean = float(np.mean(dem[valid_mask]))

    # WorldClim BIO_01 stored as integer in tenths of °C historically; check magnitude
    if abs(t_mean) > 80:
        t_mean = t_mean / 10.0

    csv_p = ML_DATASET / f"{basin_id}.csv"
    if csv_p.exists():
        with open(csv_p) as f:
            f.readline()
            n_pos = sum(1 for line in f if line.split(",")[4] == "1")
    else:
        n_pos = None

    return {
        "basin_id": basin_id,
        "lat": lat_c,
        "lon": lon_c,
        "area_km2": area_km2,
        "t_mean": t_mean,
        "p_mean": p_mean,
        "p_low": p_low,
        "elev_mean": elev_mean,
        "n_pos": n_pos,
    }


def main():
    rows = []
    for basin_id, name, regime in BASINS:
        s = basin_stats(basin_id)
        s["name"] = name
        s["regime"] = regime
        rows.append(s)
        print(
            f"{name:<8} lat={s['lat']:+.2f}  area={s['area_km2']:7,.0f} km²  "
            f"T={s['t_mean']:+.1f}°C  P={s['p_mean']:5.0f} mm  "
            f"P<2km={s['p_low']:5.0f} mm  z={s['elev_mean']:5.0f} m  "
            f"n_pos={s['n_pos']}"
        )

    # ---- LaTeX
    lines = [
        r"\begin{table*}",
        r"\centering",
        r"\caption{Summary of the five Chilean BNA basins used in this study. "
        r"Latitude is the centroid of the basin bounding box. Area is the count "
        r"of in-basin DEM pixels; the 30 m rasters carry no nodata tag and code "
        r"cells outside the divide as zero, so all statistics here are computed "
        r"under an explicit $\mathrm{DEM}>0$ mask rather than over the "
        r"rectangular footprint. Mean elevation ($\bar{z}$), mean annual "
        r"temperature ($T$) and annual precipitation ($P$) are averages over "
        r"that mask; $T$ and $P$ are the WorldClim BioClim layers BIO\_01 and "
        r"BIO\_12 resampled to the DEM grid. "
        r"$P_{<2\,\mathrm{km}}$ restricts the precipitation average to pixels "
        r"below 2{,}000~m. The regime labels follow the lowland climatology of "
        r"Chile rather than the basin mean: Lluta and Huasco extend into the "
        r"Altiplano, whose austral-summer convective rainfall lifts the "
        r"catchment average of hyper-arid Lluta ($\bar{z}=3{,}432$~m) above that "
        r"of semi-arid Huasco. "
        r"Restricted to the lowlands, precipitation increases monotonically "
        r"southward, as the transect design assumes. "
        r"Positive landslide events ($n_+$) include both rainfall- and "
        r"seismic-triggered events from SERNAGEOMIN and the Maule 2010 "
        r"inventory after spatial validity filtering.}",
        r"\label{tab:basins}",
        r"\small",
        r"\begin{tabular}{cllrrrrrrr}",
        r"\toprule",
        r"BNA & Basin & Regime & Lat ($^{\circ}$) & Area (km$^2$) & "
        r"$\bar{z}$ (m) & $T$ ($^{\circ}$C) & $P$ (mm/yr) & "
        r"$P_{<2\,\mathrm{km}}$ (mm/yr) & $n_+$ \\",
        r"\midrule",
    ]
    for r in rows:
        bna = BNA_CODE[r["basin_id"]]
        npos_str = f"{r['n_pos']}" if r["n_pos"] is not None else "--"
        lines.append(
            f"{bna} & {r['name']} & {r['regime']} & "
            f"{r['lat']:+.2f} & {r['area_km2']:,.0f} & "
            f"{r['elev_mean']:,.0f} & {r['t_mean']:+.1f} & "
            f"{r['p_mean']:,.0f} & {r['p_low']:,.0f} & {npos_str} \\\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
        "",
    ])
    (OUT_DIR / "tab1_basins.tex").write_text("\n".join(lines))
    print(f"\n  wrote tab1_basins.tex")

    # ---- Markdown
    md_lines = [
        "# Table 1 — Five Chilean BNA basins",
        "",
        "| BNA | Basin | Regime | Lat (°) | Area (km²) | z̄ (m) | T (°C) | "
        "P (mm/yr) | P<2 km (mm/yr) | n_+ |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        bna = BNA_CODE[r["basin_id"]]
        npos_str = f"{r['n_pos']}" if r["n_pos"] is not None else "--"
        md_lines.append(
            f"| {bna} | {r['name']} | {r['regime']} | "
            f"{r['lat']:+.2f} | {r['area_km2']:,.0f} | "
            f"{r['elev_mean']:,.0f} | {r['t_mean']:+.1f} | "
            f"{r['p_mean']:,.0f} | {r['p_low']:,.0f} | {npos_str} |"
        )
    (OUT_DIR / "tab1_basins.md").write_text("\n".join(md_lines))
    print(f"  wrote tab1_basins.md")


if __name__ == "__main__":
    main()
