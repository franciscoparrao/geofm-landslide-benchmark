"""Fase 2 — Geospatial-foundation-model linear probe on susceptibility.

For one basin: build dataset (positives from SERNAGEOMIN inventory, negatives
sampled with buffer), extract 224x224 patches centered on each point, encode
with a chosen backbone (TerraMind v1-tiny by default, or Prithvi-EO-2.0-300M),
mean-pool last-layer tokens to obtain an embedding per point, train RF, report
AUC ROC / PR with bootstrap CI.

Same protocol as src/susceptibility_h33.py to make Δ-AUC comparable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import warnings
from pathlib import Path
from time import perf_counter

warnings.filterwarnings("ignore")

import numpy as np
import pyproj
import rasterio
import torch
from rasterio.windows import Window
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import (
    GroupKFold, StratifiedGroupKFold, StratifiedKFold,
)

from config import (
    BASINS, DEFAULT_BASIN, INVENTORY_BASE, ML_DATASET_BASE, PRITHVI_PATH,
    RESULTS, S2_COMPOSITE_BASE, basin_dir,
)

PATCH_SIZE = 224
NEG_RATIO = 5
BUFFER_PX = PATCH_SIZE // 2  # 112 px = 3360 m, prevents patch-content leak
# Sensitivity to the negative-exclusion window (review R2, 2026-09-29): the
# 3.36 km window keeps every negative out of the neighbourhood of every
# positive, which can make the classes separable by position alone. Setting
# GEOFM_BUFFER_PX re-runs the design with a smaller window; those runs write
# caches and results under a _buf<px> suffix so the primary ones are untouched.
BUFFER_PX = int(os.environ.get("GEOFM_BUFFER_PX") or BUFFER_PX)
RUN_SUFFIX = "" if BUFFER_PX == PATCH_SIZE // 2 else f"_buf{BUFFER_PX}"
N_FOLDS = 5
N_BOOTSTRAP = 1000
N_TREES = 300
SEED = 42
BATCH_SIZE = 16
SPATIAL_BLOCK_PX = 333  # 333 * 30 m ≈ 10 km blocks for spatial holdout
S2_BAND_COUNT = 12   # composite has 12 S2L2A bands in PRETRAINED_BANDS order

# Encoder choices and their characteristics.
TERRAMIND_NAME = "terramind_v1_tiny"
TERRAMIND_EMBED_DIM = 192

PRITHVI_NAME = "prithvi-300m"
PRITHVI_EMBED_DIM = 1024
DEFAULT_PRITHVI_PATH = PRITHVI_PATH

# HLS band order Prithvi was pretrained on: Blue, Green, Red, NIR-narrow, SWIR1, SWIR2.
# Mapping from our 12-band S2L2A composite (B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B11,B12):
#   HLS B02 (Blue)        <- S2 B02  (composite index 1)
#   HLS B03 (Green)       <- S2 B03  (composite index 2)
#   HLS B04 (Red)         <- S2 B04  (composite index 3)
#   HLS B05 (NIR narrow)  <- S2 B8A  (composite index 8)
#   HLS B06 (SWIR1)       <- S2 B11  (composite index 10)
#   HLS B07 (SWIR2)       <- S2 B12  (composite index 11)
PRITHVI_HLS_INDICES = [1, 2, 3, 8, 10, 11]

# Identity of the source rasters the embeddings are computed from. It enters the
# embedding fingerprint, so bumping it invalidates every cache. Bump it whenever
# an input raster is rebuilt, and say what changed.
#   "globaldraw": composites built by taking the thirty least-cloudy scenes of
#       the whole basin, which left entire Sentinel-2 MGRS tiles with no
#       contributing scene (27% of Maule absent along straight tile edges).
#   "pertile": composites built by drawing the least-cloudy scenes from each
#       MGRS tile separately (2026-09-16). Gaps fall to 0.00%, 0.00% and 3.36%.
INPUT_GENERATION = "pertile"

# How TerraMind's inputs are prepared. It enters the fingerprint of every
# TerraMind embedding, so changing it invalidates those caches.
#   (absent): DEM in metres and Sentinel-2 divided by 10,000, on the belief that
#       the backbone normalises its own inputs. It does not: terratorch's
#       TerraMindViT.forward applies no standardisation, and the pretraining
#       statistics are used only by the generation wrappers, with
#       standardize=False by default. The encoder saw raw metres where it was
#       trained on z-scores.
#   "zscore+offset": the pretraining convention (2026-09-28). Sentinel-2 L2A
#       from processing baseline 04.00 onward carries BOA_ADD_OFFSET = 1000 DN;
#       TerraMesh removed it from post-2022 data before computing its
#       statistics, so it is subtracted here from every valid pixel, and both
#       modalities are then standardised with terratorch's v1 pretraining mean
#       and standard deviation.
TERRAMIND_INPUT = "zscore+offset"
S2_BOA_OFFSET = 1000.0

# Same correction for Prithvi, which is pretrained on HLS: harmonised surface
# reflectance with no BOA_ADD_OFFSET. Absent = raw L2A DN with the 1000 DN
# offset of baseline 04.00 left in (0.44-0.95 sd per band against Prithvi's
# own statistics). "offset" (2026-10-04): subtracted from valid pixels before
# Prithvi's mean/std normalisation. Enters the Prithvi fingerprint.
PRITHVI_INPUT = "offset"


def remove_boa_offset(patches):
    """Subtract BOA_ADD_OFFSET from valid pixels of (n, bands, h, w) raw DN.

    A pixel whose bands are all zero is composite nodata and keeps 0.
    """
    nodata = (patches == 0).all(axis=1, keepdims=True)
    return np.where(nodata, 0.0, patches - S2_BOA_OFFSET).astype(np.float32)


def load_events_xy(basin, raster_crs, raster_transform):
    """Return (rows, cols) of positive events in raster pixel coordinates.

    Prefers ml_dataset/{basin}.csv (positives + negatives pre-processed in UTM
    by the project's pipeline; we use only positives with label==1). Falls back
    to basin_inventory/{basin}.csv (raw SERNAGEOMIN catastro in lat/lon) if the
    ml_dataset file is missing. Uses csv.reader to handle quoted strings.
    """
    import csv
    ml_path = ML_DATASET_BASE / f"{basin}.csv"
    if ml_path.exists():
        xs, ys = [], []
        with open(ml_path) as f:
            reader = csv.DictReader(f)
            for r in reader:
                if int(r.get("label", "0")) != 1:
                    continue
                xs.append(float(r["x"])); ys.append(float(r["y"]))
        xs_arr = np.array(xs); ys_arr = np.array(ys)
        print(f"[load_events] using ml_dataset/{basin}.csv n_positives={len(xs)}")
    else:
        csv_path = INVENTORY_BASE / f"{basin}.csv"
        lats, lons = [], []
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for r in reader:
                lats.append(float(r["lat"])); lons.append(float(r["lon"]))
        transformer = pyproj.Transformer.from_crs(
            "EPSG:4326", raster_crs, always_xy=True
        )
        xs_arr, ys_arr = transformer.transform(lons, lats)
        xs_arr = np.array(xs_arr); ys_arr = np.array(ys_arr)
        print(f"[load_events] using basin_inventory/{basin}.csv n_positives={len(lats)}")
    inv = ~raster_transform
    cols, rrows = inv * (xs_arr, ys_arr)
    return np.array(rrows, dtype=np.int64), np.array(cols, dtype=np.int64)


def extract_patches(dem_src, rows, cols, size=PATCH_SIZE):
    """Extract size×size DEM patches centered at each (row, col)."""
    half = size // 2
    out = np.zeros((len(rows), size, size), dtype=np.float32)
    for i, (r, c) in enumerate(zip(rows, cols)):
        win = Window(c - half, r - half, size, size)
        arr = dem_src.read(1, window=win, boundless=True, fill_value=0.0)
        if arr.shape != (size, size):
            pad_r = size - arr.shape[0]
            pad_c = size - arr.shape[1]
            arr = np.pad(arr, ((0, max(0, pad_r)), (0, max(0, pad_c))),
                         constant_values=0.0)[:size, :size]
        out[i] = arr
    out = np.where(np.isfinite(out), out, 0.0)
    return out


def extract_patches_s2(s2_src, rows, cols, size=PATCH_SIZE,
                      band_indices=None, scale=10000.0):
    """Extract size×size Sentinel-2 multi-band patches centered at each (row, col).

    Args:
        band_indices: list of 0-based band indices to keep. None = all bands.
        scale: divisor applied to raw values. 10000.0 yields reflectance in
            [0, 1]; 1.0 keeps the raw uint16 scale that Prithvi expects (with
            its own mean/std normalization downstream).
    Returns float32 array of shape (n, n_bands_selected, size, size).
    """
    half = size // 2
    n_bands_src = s2_src.count
    sel = list(range(n_bands_src)) if band_indices is None else list(band_indices)
    out = np.zeros((len(rows), len(sel), size, size), dtype=np.float32)
    for i, (r, c) in enumerate(zip(rows, cols)):
        win = Window(c - half, r - half, size, size)
        arr = s2_src.read(window=win, boundless=True, fill_value=0)
        if arr.shape[1:] != (size, size):
            pad_r = size - arr.shape[1]
            pad_c = size - arr.shape[2]
            arr = np.pad(
                arr,
                ((0, 0), (0, max(0, pad_r)), (0, max(0, pad_c))),
                constant_values=0,
            )[:, :size, :size]
        out[i] = arr[sel].astype(np.float32) / scale
    out = np.where(np.isfinite(out), out, 0.0)
    return out


def sample_negatives(dem_src, pos_rows, pos_cols, excluded, n_target, rng,
                     size=PATCH_SIZE):
    """Sample negatives inside the basin with patch completeness matched to positives.

    The DEM codes out-of-basin as literal 0 (it declares no nodata value) and
    patches are zero-padded past the raster edge, so negatives drawn from the
    rectangular bounding box carry systematically more zeros in their patch than
    the in-basin positives. A patch encoder reads that difference directly, while
    the per-pixel baseline cannot -- an artefact favouring the FM pipelines.

    Two constraints remove it: the candidate pixel must be inside the basin
    (DEM != 0), and its patch zero-fraction must not exceed the 95th percentile
    of the positives' patch zero-fraction. Returns (rows, cols, threshold).
    """
    dem = dem_src.read(1).astype(np.float32)
    dem = np.where(np.isfinite(dem), dem, 0.0)
    in_basin = dem != 0.0
    height, width = dem.shape
    half = size // 2

    def patch_zero_frac(r, c):
        r0, r1, c0, c1 = r - half, r + half, c - half, c + half
        inside = dem[max(0, r0):min(height, r1), max(0, c0):min(width, c1)]
        outside = size * size - inside.size
        return (float((inside == 0.0).sum()) + outside) / (size * size)

    pos_frac = np.array([patch_zero_frac(r, c) for r, c in zip(pos_rows, pos_cols)])
    thr = float(np.percentile(pos_frac, 95))
    print(f"[sampling] in-basin negatives, patch zero-fraction <= {thr:.4f} "
          f"(p95 of positives; positives mean {pos_frac.mean():.4f})")

    rows, cols = [], []
    tries = 0
    max_tries = 400 * n_target
    while len(rows) < n_target and tries < max_tries:
        tries += 1
        r = int(rng.integers(half, height - half))
        c = int(rng.integers(half, width - half))
        if excluded[r * width + c] or not in_basin[r, c]:
            continue
        if patch_zero_frac(r, c) > thr:
            continue
        rows.append(r); cols.append(c)
    if len(rows) < n_target:
        print(f"[sampling] WARNING: only {len(rows)}/{n_target} negatives met the "
              f"constraints after {tries} draws")
    return np.array(rows), np.array(cols), thr


_TM_STATS = None


def terramind_norm_stats():
    """TerraMind v1 pretraining mean/std, read from terratorch itself.

    Read from the library rather than copied into this file, so the values are
    the ones the installed model version was trained with.
    """
    global _TM_STATS
    if _TM_STATS is None:
        from terratorch.models.backbones.terramind.model.terramind_register import (
            v1_pretraining_mean as mean, v1_pretraining_std as std,
        )
        _TM_STATS = {
            "DEM": (np.float32(mean["untok_dem@224"][0]),
                    np.float32(std["untok_dem@224"][0])),
            "S2L2A": (np.array(mean["untok_sen2l2a@224"], np.float32)[None, :, None, None],
                      np.array(std["untok_sen2l2a@224"], np.float32)[None, :, None, None]),
        }
    return _TM_STATS


def standardize_terramind(dem_patches, s2_patches):
    """Put raw patches on TerraMind's pretraining scale.

    dem_patches in metres; s2_patches in raw L2A DN (12 bands, B10 excluded).
    A pixel whose twelve bands are all zero is composite nodata: it keeps the
    value 0 DN, as the Prithvi path does, rather than being offset to -1000.
    """
    st = terramind_norm_stats()
    dem = None
    if dem_patches is not None:
        m, sd = st["DEM"]
        dem = ((dem_patches - m) / sd).astype(np.float32)
    s2 = None
    if s2_patches is not None:
        s2 = remove_boa_offset(s2_patches)
        m, sd = st["S2L2A"]
        s2 = ((s2 - m) / sd).astype(np.float32)
    return dem, s2


def build_encoder(name, pretrained, modalities, prithvi_path=None):
    """Build a backbone and return (model, kind, embed_dim, norm_stats).

    kind is "terramind" or "prithvi". For Prithvi, norm_stats is a dict with
    'mean' and 'std' (per-band, raw-reflectance scale) used at encode time.
    For TerraMind it holds the v1 pretraining statistics per modality; the
    backbone does not standardise its inputs (see TERRAMIND_INPUT).
    """
    if name == TERRAMIND_NAME:
        from terratorch import BACKBONE_REGISTRY
        model = BACKBONE_REGISTRY.build(
            name, pretrained=pretrained, modalities=modalities,
        )
        return model, "terramind", TERRAMIND_EMBED_DIM, terramind_norm_stats()

    if name == PRITHVI_NAME:
        if prithvi_path is None:
            prithvi_path = DEFAULT_PRITHVI_PATH
        import sys
        sys.path.insert(0, prithvi_path)
        from prithvi_mae import PrithviMAE  # type: ignore
        cfg = json.loads((Path(prithvi_path) / "config.json").read_text())["pretrained_cfg"]
        cfg["num_frames"] = 1  # single-timestep inference
        norm_stats = {
            "mean": np.array(cfg["mean"], dtype=np.float32)[None, :, None, None],
            "std":  np.array(cfg["std"],  dtype=np.float32)[None, :, None, None],
        }
        model = PrithviMAE(**cfg)
        if pretrained:
            ckpt = torch.load(
                Path(prithvi_path) / "Prithvi_EO_V2_300M.pt",
                map_location="cpu", weights_only=True,
            )
            for k in list(ckpt.keys()):
                if "pos_embed" in k:
                    del ckpt[k]
            missing, unexpected = model.load_state_dict(ckpt, strict=False)
            print(f"[encoder] prithvi pretrained loaded; missing={len(missing)} unexpected={len(unexpected)}")
        return model, "prithvi", PRITHVI_EMBED_DIM, norm_stats

    raise ValueError(f"Unknown encoder: {name}")


def _encode_terramind(model, dem_patches, s2_patches, embed_dim, batch_size):
    n = dem_patches.shape[0]
    embeddings = np.zeros((n, embed_dim), dtype=np.float32)
    use_s2 = s2_patches is not None
    t0 = perf_counter()
    for i in range(0, n, batch_size):
        batch_dem = dem_patches[i:i + batch_size]
        inputs = {"DEM": torch.from_numpy(batch_dem[:, None, :, :])}
        if use_s2:
            inputs["S2L2A"] = torch.from_numpy(s2_patches[i:i + batch_size])
        with torch.no_grad():
            out = model(inputs)
        last = out if isinstance(out, torch.Tensor) else out[-1]
        embeddings[i:i + batch_size] = last.mean(dim=1).cpu().numpy().astype(np.float32)
        if (i // batch_size) % 5 == 0:
            print(f"  [encode] {i + batch_dem.shape[0]}/{n}  elapsed={perf_counter() - t0:.1f}s")
    return embeddings


def _encode_prithvi(model, s2_hls_patches, embed_dim, batch_size,
                    norm_stats, use_bf16=True):
    """Encode raw-reflectance S2 HLS-equivalent patches with Prithvi.

    Mean-pools the spatial tokens of the final encoder layer (drops CLS).
    Uses bf16 by default for ~2.5x speedup on CPU.
    """
    n = s2_hls_patches.shape[0]
    embeddings = np.zeros((n, embed_dim), dtype=np.float32)
    mean = norm_stats["mean"]; std = norm_stats["std"]
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    model = model.to(dtype)
    t0 = perf_counter()
    for i in range(0, n, batch_size):
        batch = (remove_boa_offset(s2_hls_patches[i:i + batch_size]) - mean) / std
        x = torch.from_numpy(batch).to(dtype)
        with torch.no_grad():
            feats = model.forward_features(x)
        last = feats[-1] if isinstance(feats, (list, tuple)) else feats
        # drop CLS token (index 0), mean-pool spatial tokens
        emb = last[:, 1:, :].mean(dim=1).to(torch.float32).cpu().numpy()
        embeddings[i:i + batch_size] = emb.astype(np.float32)
        if (i // batch_size) % 5 == 0:
            print(f"  [encode] {i + batch.shape[0]}/{n}  elapsed={perf_counter() - t0:.1f}s")
    return embeddings


def encode_patches(kind, model, embed_dim, dem_patches=None,
                   s2_patches=None, s2_hls_patches=None,
                   norm_stats=None, batch_size=BATCH_SIZE):
    """Dispatch to the encoder-specific implementation."""
    if kind == "terramind":
        # Expects raw inputs (metres, L2A DN), exactly as encode_streaming does.
        dem_patches, s2_patches = standardize_terramind(dem_patches, s2_patches)
        return _encode_terramind(model, dem_patches, s2_patches, embed_dim, batch_size)
    if kind == "prithvi":
        assert s2_hls_patches is not None and norm_stats is not None
        return _encode_prithvi(model, s2_hls_patches, embed_dim, batch_size, norm_stats)
    raise ValueError(f"Unknown encoder kind: {kind}")


def encode_streaming(kind, model, embed_dim, rows, cols, dem_src=None,
                     s2_src=None, s2_band_indices=None, s2_scale=10000.0,
                     norm_stats=None, chunk=256):
    """Extract and encode patches in chunks, keeping only the embeddings.

    Materialising every patch first costs bands x 224^2 x 4 bytes per point --
    5 GB of Sentinel-2 patches for a 2000-point basin, which is what made the
    multimodal and Prithvi runs die on the larger basins. Streaming caps the
    peak at `chunk` points regardless of dataset size.
    """
    n = len(rows)
    out = np.zeros((n, embed_dim), dtype=np.float32)
    t0 = perf_counter()
    for i in range(0, n, chunk):
        sl = slice(i, min(i + chunk, n))
        r, c = rows[sl], cols[sl]
        dem_p = extract_patches(dem_src, r, c) if dem_src is not None else None
        s2_p = None
        if s2_src is not None:
            # TerraMind is always read in raw DN whatever the caller passes:
            # standardisation needs the unscaled values, and a caller that
            # forgets to change its scale must not be able to undo it.
            s2_p = extract_patches_s2(
                s2_src, r, c, band_indices=s2_band_indices,
                scale=1.0 if kind == "terramind" else s2_scale)
        if kind == "terramind":
            dem_p, s2_p = standardize_terramind(dem_p, s2_p)
        if kind == "prithvi":
            out[sl] = _encode_prithvi(model, s2_p, embed_dim, BATCH_SIZE,
                                      norm_stats, use_bf16=False)
        else:
            out[sl] = _encode_terramind(model, dem_p, s2_p, embed_dim, BATCH_SIZE)
        del dem_p, s2_p
        print(f"  [stream] {min(i + chunk, n)}/{n}  elapsed={perf_counter() - t0:.0f}s",
              flush=True)
    return out


def lookup_pixel_features(X_all, valid_idx, flat_positions):
    """Gather the per-pixel feature rows for the sampled points.

    valid_idx holds one entry per valid pixel of the basin -- 23 million for
    Maule -- so the previous {int(idx): row} dict cost roughly 2.5 GB of Python
    objects and stayed alive through the encoding. That, on top of the 1.6 GB
    feature array, is what made the larger basins die under the OOM killer with
    no traceback. searchsorted performs the identical lookup on the sorted index
    at no memory cost.

    Returns (features, valid_mask); rows without a valid pixel stay NaN.
    """
    order = None
    if valid_idx.size > 1 and not np.all(np.diff(valid_idx) > 0):
        order = np.argsort(valid_idx)
    keys = valid_idx if order is None else valid_idx[order]

    pos = np.clip(np.searchsorted(keys, flat_positions), 0, len(keys) - 1)
    valid_mask = keys[pos] == flat_positions
    rows_in_X = pos if order is None else order[pos]

    feats = np.full((len(flat_positions), X_all.shape[1]), np.nan, np.float32)
    feats[valid_mask] = X_all[rows_in_X[valid_mask]]
    return feats, valid_mask


def embedding_cache_path(basin, encoder_slug, init_mode, modalities):
    return (RESULTS / "_embcache" /
            f"{basin}_{encoder_slug}_{init_mode}_"
            f"{'+'.join(sorted(modalities)).lower()}{RUN_SUFFIX}.npz")


def embedding_fingerprint(encoder_name, init_mode, modalities, rows, cols, y):
    """Identify the exact point set AND inputs an embedding matrix came from.

    Embeddings do not depend on the number of CV folds, so a run that only
    changes N_FOLDS can reuse them -- but only if the sampled points are
    identical. Anything that moves the points (a different seed, ratio, patch
    size, or a change to sample_negatives) must invalidate the cache, so the
    fingerprint covers both the sampling parameters and the resulting
    coordinates themselves.

    For modalities that read the Sentinel-2 composite it also covers
    INPUT_GENERATION, which identifies which build of that composite was used. Without
    it, replacing a source raster leaves every cached embedding silently
    reusable: the points have not moved, so the fingerprint matches, and the
    cache answers for imagery it was never computed from. The composite rebuild
    of September 2026 would have poisoned an entire re-run that way. It is a
    module constant rather than a per-call argument on purpose: the fingerprint
    is computed at thirteen sites across eleven scripts, and a parameter that
    eleven callers must remember is a parameter that some caller will forget,
    which reintroduces exactly the silent-reuse failure it was added to prevent.
    """
    payload = json.dumps({
        "encoder": encoder_name, "init": init_mode,
        "modalities": sorted(modalities),
        "patch_size": PATCH_SIZE, "neg_ratio": NEG_RATIO,
        "buffer_px": BUFFER_PX, "seed": SEED,
        "n_pos": int(y.sum()), "n_neg": int((y == 0).sum()),
        # Only the Sentinel-2 composite was rebuilt, so only embeddings that
        # read it are stale. Keying every modality to it would invalidate the
        # DEM-only caches too, which are still valid and whose encoder is not
        # currently installed anywhere -- an avoidable dead end.
        **({"input_generation": INPUT_GENERATION}
           if "S2L2A" in modalities else {}),
        **({"terramind_input": TERRAMIND_INPUT}
           if encoder_name == TERRAMIND_NAME else {}),
        **({"prithvi_input": PRITHVI_INPUT}
           if encoder_name == PRITHVI_NAME else {}),
    }, sort_keys=True).encode()
    h = hashlib.md5(payload)
    h.update(np.ascontiguousarray(rows, dtype=np.int64).tobytes())
    h.update(np.ascontiguousarray(cols, dtype=np.int64).tobytes())
    return h.hexdigest()


def load_cached_embeddings(path, fingerprint):
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as npz:
        if str(npz["fingerprint"]) != fingerprint:
            print(f"[cache] {path.name} exists but fingerprint differs -- re-encoding")
            return None
        return npz["embeddings"].astype(np.float32)


def save_cached_embeddings(path, fingerprint, embeddings):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, embeddings=embeddings,
                        fingerprint=np.array(fingerprint))
    print(f"[cache] wrote {path.name}")


def auc_with_paired_bootstrap(y, sA, sB, rng, n=N_BOOTSTRAP):
    n_obs = y.size
    boot_deltas_roc = np.empty(n)
    boot_deltas_pr = np.empty(n)
    for b in range(n):
        ix = rng.integers(0, n_obs, size=n_obs)
        if y[ix].sum() == 0 or y[ix].sum() == n_obs:
            boot_deltas_roc[b] = np.nan; boot_deltas_pr[b] = np.nan; continue
        boot_deltas_roc[b] = (
            roc_auc_score(y[ix], sB[ix]) - roc_auc_score(y[ix], sA[ix])
        )
        boot_deltas_pr[b] = (
            average_precision_score(y[ix], sB[ix])
            - average_precision_score(y[ix], sA[ix])
        )
    boot_deltas_roc = boot_deltas_roc[~np.isnan(boot_deltas_roc)]
    boot_deltas_pr = boot_deltas_pr[~np.isnan(boot_deltas_pr)]
    return {
        "delta_roc_mean": float(boot_deltas_roc.mean()),
        "delta_roc_ci_95": [float(np.percentile(boot_deltas_roc, 2.5)),
                            float(np.percentile(boot_deltas_roc, 97.5))],
        "delta_pr_mean": float(boot_deltas_pr.mean()),
        "delta_pr_ci_95": [float(np.percentile(boot_deltas_pr, 2.5)),
                           float(np.percentile(boot_deltas_pr, 97.5))],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--basin", default=DEFAULT_BASIN, choices=BASINS)
    parser.add_argument("--cv", choices=("stratified", "spatial"),
                        default="stratified")
    parser.add_argument("--init", choices=("pretrained", "random"),
                        default="pretrained")
    parser.add_argument(
        "--encoder", choices=(TERRAMIND_NAME, PRITHVI_NAME),
        default=TERRAMIND_NAME,
        help="Backbone to evaluate. Default: TerraMind v1-tiny.",
    )
    parser.add_argument(
        "--prithvi-path", default=DEFAULT_PRITHVI_PATH,
        help="Path to the local Prithvi-EO-2.0-300M model directory.",
    )
    parser.add_argument(
        "--folds", type=int, default=N_FOLDS,
        help=f"Number of CV folds. Default {N_FOLDS} (the primary analysis). "
             "Any other value writes to a _k<folds> file so the primary "
             "results are never overwritten.",
    )
    parser.add_argument(
        "--splitter", choices=("group", "stratified-group"), default="group",
        help="Spatial-CV splitter. 'group' (GroupKFold) is the primary "
             "analysis. 'stratified-group' keeps blocks intact but balances "
             "the classes across folds, which is required at high fold counts: "
             "Huasco's events concentrate in few blocks, so plain GroupKFold "
             "yields a positive-free test fold and an undefined AUC at k=10. "
             "Writes to a _sgkf file so the primary results are preserved.",
    )
    parser.add_argument(
        "--encode-only", action="store_true",
        help="Populate the embedding cache and stop before cross-validation. "
             "Embeddings do not depend on the fold count or the splitter, so "
             "this lets the expensive work start before those are settled.",
    )
    parser.add_argument(
        "--modalities", nargs="+", default=None,
        choices=["DEM", "S2L2A"],
        help="Input modalities. Default: DEM for TerraMind, S2L2A for Prithvi. "
             "Prithvi only accepts S2L2A.",
    )
    args = parser.parse_args()
    basin = args.basin
    cv_mode = args.cv
    init_mode = args.init
    encoder_name = args.encoder
    n_folds = args.folds
    if n_folds < 2:
        raise SystemExit("--folds must be at least 2")

    if args.modalities is None:
        modalities = ["S2L2A"] if encoder_name == PRITHVI_NAME else ["DEM"]
    else:
        modalities = list(args.modalities)
    if encoder_name == PRITHVI_NAME and modalities != ["S2L2A"]:
        raise SystemExit(
            "Prithvi-EO-2.0 was pretrained on HLS only; "
            "use --modalities S2L2A (DEM is not natively supported)."
        )

    encoder_slug = "terramind" if encoder_name == TERRAMIND_NAME else encoder_name
    use_s2 = "S2L2A" in modalities
    if use_s2:
        s2_path = S2_COMPOSITE_BASE / f"{basin}_s2l2a_2023.tif"
        if not s2_path.exists():
            raise SystemExit(f"S2 composite not found: {s2_path}. "
                             "Run paper/scripts/download_s2_composite.py first.")

    rng = np.random.default_rng(SEED)

    dem_path = basin_dir(basin) / "dem_30m.tif"
    dem_src = rasterio.open(dem_path)
    height, width = dem_src.shape
    transform = dem_src.transform
    crs = dem_src.crs
    print(f"[fase2.0] basin={basin} DEM shape={dem_src.shape}")

    pos_rows, pos_cols = load_events_xy(basin, crs, transform)
    in_bounds = (
        (pos_rows >= PATCH_SIZE // 2) & (pos_rows < height - PATCH_SIZE // 2)
        & (pos_cols >= PATCH_SIZE // 2) & (pos_cols < width - PATCH_SIZE // 2)
    )
    pos_rows = pos_rows[in_bounds]; pos_cols = pos_cols[in_bounds]
    n_pos = pos_rows.size
    print(f"[fase2.0] positives_valid_in_bounds={n_pos}")
    if n_pos < 30:
        raise SystemExit("too few positive events for modeling")

    excluded = np.zeros(height * width, dtype=bool)
    for r, c in zip(pos_rows, pos_cols):
        for dr in range(-BUFFER_PX, BUFFER_PX + 1):
            for dc in range(-BUFFER_PX, BUFFER_PX + 1):
                r2 = r + dr; c2 = c + dc
                if 0 <= r2 < height and 0 <= c2 < width:
                    excluded[r2 * width + c2] = True

    n_neg_target = NEG_RATIO * n_pos
    neg_rows, neg_cols, _zero_thr = sample_negatives(
        dem_src, pos_rows, pos_cols, excluded, n_neg_target, rng
    )
    n_neg = neg_rows.size
    print(f"[fase2.0] negatives_sampled={n_neg}")

    all_rows = np.concatenate([pos_rows, neg_rows])
    all_cols = np.concatenate([pos_cols, neg_cols])
    y = np.concatenate([np.ones(n_pos), np.zeros(n_neg)]).astype(np.int8)

    stack_path = RESULTS / f"{basin}_stack.npz"
    print(f"[fase2.0] loading pixel features from {stack_path.name}")
    with np.load(stack_path, allow_pickle=False) as npz:
        # Already float32 on disk, so copy=False avoids duplicating 1.6 GB on
        # the larger basins for no gain.
        X_all = npz["X"].astype(np.float32, copy=False)
        valid_idx = npz["valid_idx"]
        feature_names_pixel = list(npz["feature_names"])
    flat_positions = all_rows.astype(np.int64) * width + all_cols
    pixel_features, valid_pixel_mask = lookup_pixel_features(
        X_all, valid_idx, flat_positions
    )
    # X_all is 1.6 GB for Maule and is not needed past this point; holding it
    # through the encoding is what pushed the larger basins into the OOM killer.
    del X_all, valid_idx
    n_valid_pixel = int(valid_pixel_mask.sum())
    print(f"[fase2.0] pixel_features valid for {n_valid_pixel}/{len(all_rows)} points")
    if n_valid_pixel < len(all_rows):
        keep = valid_pixel_mask
        all_rows = all_rows[keep]; all_cols = all_cols[keep]
        pixel_features = pixel_features[keep]
        y = y[keep]
        n_pos = int(y.sum()); n_neg = int((y == 0).sum())
        print(f"[fase2.0] after pixel-validity filter: pos={n_pos} neg={n_neg}")

    needs_dem = encoder_name == TERRAMIND_NAME
    s2_src = rasterio.open(s2_path) if use_s2 else None
    if needs_dem:
        print(f"[fase2] will stream {len(all_rows)} DEM patches (size={PATCH_SIZE})")
    if use_s2:
        which = "12-band S2L2A" if encoder_name == TERRAMIND_NAME else "6-band S2 HLS-equivalent"
        print(f"[fase2] will stream {len(all_rows)} {which} patches from {s2_path.name}")

    cache_path = embedding_cache_path(basin, encoder_slug, init_mode, modalities)
    fingerprint = embedding_fingerprint(
        encoder_name, init_mode, modalities, all_rows, all_cols, y
    )
    embeddings = load_cached_embeddings(cache_path, fingerprint)
    if embeddings is not None:
        print(f"[cache] reusing {cache_path.name} "
              f"shape={embeddings.shape} (encoding skipped)")
    else:
        print(f"[fase2] loading {encoder_name}  init={init_mode}  modalities={modalities}")
        pretrained = init_mode == "pretrained"
        model, encoder_kind, embed_dim, norm_stats = build_encoder(
            encoder_name, pretrained=pretrained, modalities=modalities,
            prithvi_path=args.prithvi_path,
        )
        if not pretrained:
            torch.manual_seed(SEED)
            for p in model.parameters():
                if p.requires_grad and p.ndim >= 2:
                    torch.nn.init.xavier_uniform_(p)
        model.eval()
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[fase2] params={n_params:,}  embed_dim={embed_dim}  kind={encoder_kind}")

        print(f"[fase2] encoding patches with batch_size={BATCH_SIZE}")
        t0 = perf_counter()
        embeddings = encode_streaming(
            encoder_kind, model, embed_dim, all_rows, all_cols,
            dem_src=dem_src if needs_dem else None,
            s2_src=s2_src,
            s2_band_indices=None if encoder_kind == "terramind" else PRITHVI_HLS_INDICES,
            s2_scale=1.0,
            norm_stats=norm_stats,
        )
        print(f"[fase2] embeddings.shape={embeddings.shape} elapsed={perf_counter() - t0:.1f}s")
        save_cached_embeddings(cache_path, fingerprint, embeddings)

    if args.encode_only:
        print("[fase2] --encode-only: embeddings cached, stopping before CV")
        return

    # Defined here and not in the branch above: on a cache hit no encoder is
    # built, so the dimension has to come from the embeddings themselves.
    embed_dim = embeddings.shape[1]

    if cv_mode == "spatial":
        n_cols_blocks = (width + SPATIAL_BLOCK_PX - 1) // SPATIAL_BLOCK_PX
        block_id = (all_rows // SPATIAL_BLOCK_PX) * n_cols_blocks + (all_cols // SPATIAL_BLOCK_PX)
        unique_blocks = np.unique(block_id)
        pos_blocks = np.unique(block_id[y == 1])
        neg_blocks = np.unique(block_id[y == 0])
        print(f"[fase2.0] spatial CV  block_size={SPATIAL_BLOCK_PX}px (~{SPATIAL_BLOCK_PX * 30 / 1000:.1f} km)")
        print(f"[fase2.0]   unique blocks: {len(unique_blocks)}  with pos: {len(pos_blocks)}  with neg: {len(neg_blocks)}")
        if len(unique_blocks) < n_folds:
            raise SystemExit(
                f"spatial CV needs at least {n_folds} blocks, basin has "
                f"{len(unique_blocks)}"
            )
        if args.splitter == "group":
            cv_iter = GroupKFold(n_splits=n_folds).split(embeddings, y, groups=block_id)
        else:
            cv_iter = StratifiedGroupKFold(n_splits=n_folds, shuffle=False).split(
                embeddings, y, groups=block_id)
    else:
        cv_iter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED).split(embeddings, y)

    fold_results = []
    sA_all = np.empty(len(y), dtype=np.float64)
    sB_all = np.empty(len(y), dtype=np.float64)
    for fi, (tr, te) in enumerate(cv_iter):
        t0 = perf_counter()
        if len(np.unique(y[te])) < 2:
            raise SystemExit(
                f"fold {fi} test set is single-class ({int(y[te].sum())} positives "
                f"of {len(te)}); AUC is undefined at {n_folds} folds for this basin"
            )
        rfA = RandomForestClassifier(
            n_estimators=N_TREES, max_depth=None, min_samples_leaf=5,
            n_jobs=-1, random_state=SEED + fi, class_weight="balanced",
        ).fit(pixel_features[tr], y[tr])
        sA = rfA.predict_proba(pixel_features[te])[:, 1]
        sA_all[te] = sA
        rocA = roc_auc_score(y[te], sA); prA = average_precision_score(y[te], sA)

        rfB = RandomForestClassifier(
            n_estimators=N_TREES, max_depth=None, min_samples_leaf=5,
            n_jobs=-1, random_state=SEED + fi, class_weight="balanced",
        ).fit(embeddings[tr], y[tr])
        sB = rfB.predict_proba(embeddings[te])[:, 1]
        sB_all[te] = sB
        rocB = roc_auc_score(y[te], sB); prB = average_precision_score(y[te], sB)

        elapsed = perf_counter() - t0
        print(
            f"[fase2.0] fold={fi}  A(pix17): ROC={rocA:.3f} PR={prA:.3f}  "
            f"B(tm192): ROC={rocB:.3f} PR={prB:.3f}  "
            f"ΔROC={rocB - rocA:+.3f} ΔPR={prB - prA:+.3f}  time={elapsed:.1f}s"
        )
        fold_results.append({
            "fold": fi,
            "roc_A": float(rocA), "pr_A": float(prA),
            "roc_B": float(rocB), "pr_B": float(prB),
            "delta_roc": float(rocB - rocA), "delta_pr": float(prB - prA),
        })

    rocsA = np.array([r["roc_A"] for r in fold_results])
    prsA = np.array([r["pr_A"] for r in fold_results])
    rocsB = np.array([r["roc_B"] for r in fold_results])
    prsB = np.array([r["pr_B"] for r in fold_results])
    delta_rocs = rocsB - rocsA
    delta_prs = prsB - prsA
    print(f"\n[fase2] A (17 pixel feats):  ROC={rocsA.mean():.3f}±{rocsA.std():.3f}  PR={prsA.mean():.3f}±{prsA.std():.3f}")
    print(f"[fase2] B ({encoder_name} {embed_dim}-dim): ROC={rocsB.mean():.3f}±{rocsB.std():.3f}  PR={prsB.mean():.3f}±{prsB.std():.3f}")
    print(f"[fase2.0] Δ-ROC = {delta_rocs.mean():+.3f} ± {delta_rocs.std():.3f}")
    print(f"[fase2.0] Δ-PR  = {delta_prs.mean():+.3f} ± {delta_prs.std():.3f}")

    boot_rng = np.random.default_rng(SEED + 1)
    n = len(y)
    boot_droc = np.empty(N_BOOTSTRAP); boot_dpr = np.empty(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        ix = boot_rng.integers(0, n, size=n)
        if y[ix].sum() == 0 or y[ix].sum() == n:
            boot_droc[b] = np.nan; boot_dpr[b] = np.nan; continue
        boot_droc[b] = roc_auc_score(y[ix], sB_all[ix]) - roc_auc_score(y[ix], sA_all[ix])
        boot_dpr[b] = average_precision_score(y[ix], sB_all[ix]) - average_precision_score(y[ix], sA_all[ix])
    boot_droc = boot_droc[~np.isnan(boot_droc)]
    boot_dpr = boot_dpr[~np.isnan(boot_dpr)]
    ci_roc = (float(np.percentile(boot_droc, 2.5)), float(np.percentile(boot_droc, 97.5)))
    ci_pr = (float(np.percentile(boot_dpr, 2.5)), float(np.percentile(boot_dpr, 97.5)))
    print(f"[fase2.0] paired bootstrap N={N_BOOTSTRAP}")
    print(f"[fase2.0]   Δ-ROC 95% CI = [{ci_roc[0]:+.3f}, {ci_roc[1]:+.3f}]")
    print(f"[fase2.0]   Δ-PR  95% CI = [{ci_pr[0]:+.3f}, {ci_pr[1]:+.3f}]")

    summary = {
        "basin": basin, "model": encoder_name,
        "init_mode": init_mode, "modalities": modalities,
        "patch_size": PATCH_SIZE, "buffer_px": BUFFER_PX,
        "cv_mode": cv_mode, "spatial_block_px": SPATIAL_BLOCK_PX,
        "n_folds": int(n_folds), "splitter": args.splitter,
        "n_pos": int(n_pos), "n_neg": int(n_neg),
        "embedding_dim": int(embeddings.shape[1]),
        "fold_results": fold_results,
        "A_roc_mean": float(rocsA.mean()), "A_roc_std": float(rocsA.std()),
        "A_pr_mean": float(prsA.mean()), "A_pr_std": float(prsA.std()),
        "B_roc_mean": float(rocsB.mean()), "B_roc_std": float(rocsB.std()),
        "B_pr_mean": float(prsB.mean()), "B_pr_std": float(prsB.std()),
        "delta_roc_mean": float(delta_rocs.mean()),
        "delta_roc_ci_95": list(ci_roc),
        "delta_pr_mean": float(delta_prs.mean()),
        "delta_pr_ci_95": list(ci_pr),
    }
    cv_suffix = "spatial" if cv_mode == "spatial" else "fair"
    init_suffix = "" if init_mode == "pretrained" else "_randinit"
    if encoder_name == TERRAMIND_NAME:
        modality_suffix = "" if modalities == ["DEM"] else "_" + "+".join(sorted(modalities)).lower()
    else:
        modality_suffix = ""  # Prithvi is S2-only, no need to disambiguate
    # The 5-fold run is the primary analysis and owns the unsuffixed filename;
    # any other fold count is a secondary variant and gets its own file.
    folds_suffix = "" if n_folds == N_FOLDS else f"_k{n_folds}"
    splitter_suffix = "" if args.splitter == "group" else "_sgkf"
    out_json = RESULTS / f"{basin}_{encoder_slug}_linprobe_{cv_suffix}{init_suffix}{modality_suffix}{splitter_suffix}{folds_suffix}{RUN_SUFFIX}.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"[fase2] wrote {out_json.name}")


if __name__ == "__main__":
    main()
