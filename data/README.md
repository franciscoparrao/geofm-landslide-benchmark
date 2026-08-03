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

Expected layout per basin (paths configured in `src/config.py`):

```
data/{basin}/dem_30m.tif
data/{basin}/terrain/{feature}.tif
data/{basin}/hydrology/{feature}.tif
s2_composites/{basin}_s2l2a_2023.tif
basin_inventory/{basin}.csv   # id,type,trigger,year,lat,lon
```
