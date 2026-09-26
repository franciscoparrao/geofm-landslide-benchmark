# Data inputs

Nothing in this directory is redistributed; all inputs are open or publicly
accessible from their original providers.

| Input | Source | Access |
|---|---|---|
| DEM 30 m (per basin) | Copernicus GLO-30 | Microsoft Planetary Computer STAC, collection `cop-dem-glo-30` |
| Sentinel-2 L2A scenes (2023 composites) | ESA / Planetary Computer | `paper_artifacts/download_s2_composite.py` |
| 17-layer geomorphometric stack | derived from the DEM | `src/build_stack.py` (terrain/hydrology layers computed with SurtGIS/SAGA-compatible algorithms) |
| Landslide inventory (national) | SERNAGEOMIN catalog of mass-wasting events | public, request via SERNAGEOMIN portal; keep only records with "exact coordinates" precision |
| Maule 2010 coseismic inventory | Serey et al. (2019), Landslides 16:1153-1165 | published supplementary data |
| WorldClim BioClim normals | Fick & Hijmans (2017) | worldclim.org |

## Expected layout

Everything defaults to a location inside the package, so putting the inputs
here — or symlinking them here, which is what we do, since these datasets are
large and are never copied into the repository — requires no code changes:

```
data/{basin}/dem_30m.tif
data/{basin}/terrain/{feature}.tif
data/{basin}/hydrology/{feature}.tif
data/s2_composites/{basin}_s2l2a_2023.tif
data/basin_inventory/{basin}.csv    # id,type,trigger,year,lat,lon
data/ml_dataset/{basin}.csv         # event coordinates used by the probes
data/basin_polygons/{basin}.*       # basin outlines, for the composite builder
data/models/prithvi-300m/           # Prithvi-EO-2.0-300M weights + config.json
```

## Overriding the locations

Each default can be redirected with an environment variable instead of editing
the source:

| Variable | Overrides |
|---|---|
| `GEOFM_BASIN_DATA_DIR` | `data/` (per-basin DEM and feature rasters) |
| `GEOFM_INVENTORY_DIR` | `data/basin_inventory/` |
| `GEOFM_ML_DATASET_DIR` | `data/ml_dataset/` |
| `GEOFM_S2_DIR` | `data/s2_composites/` |
| `GEOFM_BASIN_POLYGONS_DIR` | `data/basin_polygons/` |
| `GEOFM_PRITHVI_DIR` | `data/models/prithvi-300m/` (or pass `--prithvi-path`) |
| `GEOFM_RESULTS_DIR` | `results/` |
| `GEOFM_TABLES_DIR`, `GEOFM_FIGURES_DIR` | `tables/`, `figures/` |
| `GEOFM_COUNTRY_SHP` | national basin boundaries, locator map only |
