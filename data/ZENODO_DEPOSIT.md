# Zenodo deposit: derived datasets for geofm-landslide-benchmark v1.4.0

The deposit holds the code of tag `v1.4.0` and the derived artifacts that the
per-fold result files in `results/` summarise. Nothing here is needed to
regenerate a table or figure (the committed JSON suffice); these files let a
reader re-run the pipeline without rebuilding the inputs.

All rasters and pixel indices share the basin's 30 m DEM grid (Copernicus
GLO-30), projected in UTM 19S (EPSG:32719). Basin slugs follow the Chilean
national basin codes: `01_rio_lluta`, `06_rio_huasco`, `09_rio_maipo`,
`11_rio_maule`, `13_rio_bueno`.

## Files

| File | Content |
|---|---|
| `geofm-landslide-benchmark-v1.4.0.zip` | the repository at tag `v1.4.0` |
| `{basin}_stack.npz` (5 basins) | 17-layer geomorphometric feature stack |
| `{basin}_s2l2a_2023.tif` (Huasco, Maipo, Maule) | annual 2023 Sentinel-2 L2A composite |
| `{basin}_points_uniform.csv`, `{basin}_points_constrained.csv` | benchmark points |
| `{basin}_prithvi_patches.u16` + `.npz` | Prithvi input patches of the benchmark points |
| `11_rio_maule_folds.npz` | Maule benchmark folds as index arrays |

### Feature stacks (`{basin}_stack.npz`)

`X` is float32, shape (n valid pixels, 17), columns named in `feature_names`;
`valid_idx` gives each row's flat index into the `height` x `width` grid, with
`transform` and `crs_wkt` georeferencing it. Layers: convergence, curvature,
dev, eastness (sin of aspect), mrrtf, mrvbf, northness (cos of aspect),
openness_negative, openness_positive, slope, svf, tpi, tri, vrm, twi, hand,
log1p_flow_accumulation_mfd. Built by `src/build_stack.py`.

### Sentinel-2 composites (`{basin}_s2l2a_2023.tif`)

Twelve uint16 bands in the order B01, B02, B03, B04, B05, B06, B07, B08, B8A,
B09, B11, B12 (band descriptions carry the names). Per-pixel mean of up to thirty
least-cloudy 2023 scenes (cloud cover below 20 %) drawn per MGRS tile from the
Microsoft Planetary Computer, reprojected to the DEM grid by nearest neighbour.
No per-pixel cloud or snow mask is applied. Values are raw L2A digital numbers:
scenes from processing baseline 04.00 onward carry `BOA_ADD_OFFSET = 1000`, which
the pipeline subtracts at encoding time (`TERRAMIND_INPUT`, `PRITHVI_INPUT` in
`src/terramind_linprobe.py`). Built by `paper_artifacts/download_s2_composite.py`.

### Benchmark points (`*_points_uniform.csv`, `*_points_constrained.csv`)

Columns `row, col` (pixel indices on the DEM grid), `x, y_coord` (EPSG:32719),
`label` (1 event, 0 background), `spatial_block_id` (10 km block). `uniform` is
the benchmark draw (seed 42); `constrained` restricts negatives to the
positives' slope envelope (Table 7). Both match `point_probes.build_dataset()`
exactly in v1.4.0.

### Prithvi patches (`{basin}_prithvi_patches.u16`, `.npz`)

A raw memmap of uint16, shape `(n, 6, 224, 224)` as stored in the `.npz`
(`shape`), holding the six HLS-equivalent bands (B02, B03, B04, B8A, B11, B12)
around each benchmark point, in the order of the `.npz` `rows`, `cols`, `y`. The
values are raw L2A digital numbers: subtract the 1000 DN offset from valid
pixels before Prithvi's mean/std normalisation. The `.npz` `fingerprint` was
computed before the offset entered the embedding fingerprint, so it does not
match a v1.4.0 cache key; the point coordinates do. Built by
`src/export_patches.py` for runs on hosts without the basin rasters.

### Maule folds (`11_rio_maule_folds.npz`)

`train_k`, `test_k` index arrays (k = 0..4) of the benchmark's `GroupKFold`
over 10 km blocks, aligned with the point order above.

## Not included

The season-matched 2016/2023 composites of the pre-/post-event test, embedding
caches and susceptibility rasters. The pre-/post-event test, the
transformer-block sweep, low-rank adaptation and Prithvi's susceptibility
surfaces predate the Prithvi input-offset correction and are not reported in
the v1.4.0 manuscript.
