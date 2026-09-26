r"""Render susceptibility surfaces for a basin from the benchmarked pipelines.

The benchmark reports differences in AUC. Those are summary statistics, and
\citet{Steger2016} shows that a model can hold a high AUC while producing a map
a geomorphologist would reject, because the error structure is spatial and a
scalar hides it. This script produces the missing object: the actual surfaces,
so the reported difference can be looked at rather than only read.

Two surfaces per basin:

  A        the seventeen-layer geomorphometric baseline, at the native 30 m
           grid, over every pixel with finite values in all layers;
  PRITHVI  Prithvi-EO-2.0-300M embeddings, on a strided grid, because one
           224 x 224 patch per pixel is not computable at 30 m over a basin of
           twenty thousand square kilometres.

Both are fitted on the full benchmark point set rather than fold by fold. That
is the usual convention for a published susceptibility map and it avoids
stitching one map out of five models, which would put seams at block
boundaries and invite exactly the kind of spatial artefact the figure exists to
expose. The honest validation of these models is the spatial-block
cross-validation of the benchmark, not the map.

Stage 1 (--stage baseline) runs locally. Stage 2 needs a GPU: it writes the
grid coordinates for a remote encoder (--stage grid), and consumes the returned
embeddings (--stage fm).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import rasterio
from sklearn.ensemble import RandomForestClassifier

from config import BASINS, RESULTS, basin_dir
from point_probes import build_dataset
from terramind_linprobe import N_TREES, PATCH_SIZE, SEED

MAP_DIR = RESULTS / "susceptibility_maps"
CHUNK = 2_000_000


def rf(seed_offset: int = 0) -> RandomForestClassifier:
    """The benchmark's classifier, with its settings unchanged."""
    return RandomForestClassifier(
        n_estimators=N_TREES, min_samples_leaf=5, n_jobs=-1,
        random_state=SEED + seed_offset, class_weight="balanced",
    )


def write_raster(path: Path, arr: np.ndarray, profile: dict) -> None:
    prof = dict(profile)
    prof.update(driver="GTiff", dtype="float32", count=1, nodata=np.float32(-1),
                compress="deflate", predictor=2, tiled=True,
                blockxsize=256, blockysize=256)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr.astype(np.float32), 1)
    print(f"[write] {path.name}  {arr.shape}  {path.stat().st_size / 1e6:.1f} MB")


# ---------------------------------------------------------------- stage 1
def stage_baseline(basin: str) -> None:
    rows, cols, y, feats, _blk = build_dataset(basin, negatives="uniform")
    clf = rf()
    t0 = perf_counter()
    clf.fit(feats, y)
    print(f"[fit] A on {len(y)} points ({int(y.sum())} pos) in {perf_counter()-t0:.0f}s")

    with rasterio.open(basin_dir(basin) / "dem_30m.tif") as src:
        profile, height, width = src.profile, src.height, src.width

    with np.load(RESULTS / f"{basin}_stack.npz", allow_pickle=False) as npz:
        X_all = npz["X"].astype(np.float32, copy=False)
        valid_idx = npz["valid_idx"]

    surface = np.full(height * width, -1.0, dtype=np.float32)
    t0 = perf_counter()
    for i in range(0, len(valid_idx), CHUNK):
        sl = slice(i, min(i + CHUNK, len(valid_idx)))
        surface[valid_idx[sl]] = clf.predict_proba(X_all[sl])[:, 1]
        done = min(i + CHUNK, len(valid_idx))
        print(f"  [predict] {done:,}/{len(valid_idx):,}  "
              f"elapsed={perf_counter()-t0:.0f}s", flush=True)
    del X_all, valid_idx

    MAP_DIR.mkdir(parents=True, exist_ok=True)
    write_raster(MAP_DIR / f"{basin}_susceptibility_A.tif",
                 surface.reshape(height, width), profile)


# ---------------------------------------------------------------- stage 2a
def stage_grid(basin: str, stride: int) -> None:
    """In-basin grid centres whose 224 x 224 patch fits inside the raster."""
    with rasterio.open(basin_dir(basin) / "dem_30m.tif") as src:
        dem = src.read(1)
        height, width = src.height, src.width
    half = PATCH_SIZE // 2
    rr = np.arange(half, height - half, stride)
    cc = np.arange(half, width - half, stride)
    R, C = np.meshgrid(rr, cc, indexing="ij")
    R, C = R.ravel(), C.ravel()
    keep = np.isfinite(dem[R, C]) & (dem[R, C] > 0)   # inside the divide
    R, C = R[keep], C[keep]

    MAP_DIR.mkdir(parents=True, exist_ok=True)
    out = MAP_DIR / f"{basin}_grid_s{stride}.npz"
    np.savez(out, rows=R, cols=C, stride=np.array(stride),
             shape=np.array([height, width]),
             grid_rows=rr, grid_cols=cc)
    print(f"[grid] {out.name}: {len(R):,} cells of {R.size / max(keep.mean(),1e-9):,.0f} "
          f"candidates ({stride * 30} m spacing, {keep.mean() * 100:.0f}% in basin)")


# ---------------------------------------------------------------- stage 2b
def stage_fm(basin: str, stride: int, emb_path: Path) -> None:
    from terramind_linprobe import (
        PRITHVI_NAME, embedding_cache_path, embedding_fingerprint,
        load_cached_embeddings,
    )
    rows, cols, y, _feats, _blk = build_dataset(basin, negatives="uniform")
    fp = embedding_fingerprint(PRITHVI_NAME, "pretrained", ["S2L2A"], rows, cols, y)
    train_emb = load_cached_embeddings(
        embedding_cache_path(basin, "prithvi-300m", "pretrained", ["S2L2A"]), fp)
    if train_emb is None:
        raise SystemExit("benchmark embeddings absent or fingerprint mismatch; "
                         "the map must be built from the same encoding the paper used")
    print(f"[cache] training embeddings {train_emb.shape}")

    clf = rf()
    t0 = perf_counter()
    clf.fit(train_emb, y)
    print(f"[fit] PRITHVI on {len(y)} points in {perf_counter()-t0:.0f}s")

    grid = np.load(MAP_DIR / f"{basin}_grid_s{stride}.npz")
    G = np.load(emb_path)
    gemb, grows, gcols = G["embeddings"], G["rows"], G["cols"]
    if not (np.array_equal(grows, grid["rows"]) and np.array_equal(gcols, grid["cols"])):
        raise SystemExit("returned embeddings do not match the grid they claim to cover")
    print(f"[grid] {gemb.shape[0]:,} cells x {gemb.shape[1]} dims")

    scores = np.empty(len(gemb), dtype=np.float32)
    for i in range(0, len(gemb), 200_000):
        sl = slice(i, min(i + 200_000, len(gemb)))
        scores[sl] = clf.predict_proba(gemb[sl])[:, 1]

    gr, gc = grid["grid_rows"], grid["grid_cols"]
    surface = np.full((len(gr), len(gc)), -1.0, dtype=np.float32)
    ri = np.searchsorted(gr, grows)
    ci = np.searchsorted(gc, gcols)
    surface[ri, ci] = scores

    with rasterio.open(basin_dir(basin) / "dem_30m.tif") as src:
        base = src.transform
    tf = rasterio.Affine(base.a * stride, base.b, base.c + base.a * gc[0],
                         base.d, base.e * stride, base.f + base.e * gr[0])
    profile = {"crs": src.crs, "transform": tf,
               "height": surface.shape[0], "width": surface.shape[1]}
    write_raster(MAP_DIR / f"{basin}_susceptibility_PRITHVI_s{stride}.tif",
                 surface, profile)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--basin", default="11_rio_maule", choices=BASINS)
    ap.add_argument("--stage", required=True,
                    choices=("baseline", "grid", "fm"))
    ap.add_argument("--stride", type=int, default=20, help="grid spacing in pixels")
    ap.add_argument("--embeddings", type=Path)
    a = ap.parse_args()
    if a.stage == "baseline":
        stage_baseline(a.basin)
    elif a.stage == "grid":
        stage_grid(a.basin, a.stride)
    else:
        stage_fm(a.basin, a.stride, a.embeddings)


if __name__ == "__main__":
    main()
