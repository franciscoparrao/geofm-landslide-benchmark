"""Point-level probes for the two review confounds of the cross-FM benchmark.

Reproduces the exact benchmark dataset of terramind_linprobe.py (same seed,
sampling, validity filter, and spatial folds) and evaluates, with the same
Random Forest and metrics:

  A       17 pixel geomorphometric features (reproduction check vs stored runs)
  SPEC    spectral-point baseline: 6 HLS-equivalent bands + NDVI + NBR at the
          candidate pixel, from the same 2023 S2 composite the FM pipelines use
          (post-event scar-signal probe, review Issue 1)
  A+SPEC  17 features + spectral-point features
  ACTX    context-matched manual baseline: 17 features + nan-aware mean/std of
          each layer over 7x7, 37x37 and 111x111 pixel windows
          (representation-vs-receptive-field control, review Issue 4)

Univariate AUCs of NDVI and NBR at the point are also reported per fold.
Outputs results/{basin}_point_probes.json with per-fold metrics and paired
per-fold comparisons against the stored FM runs (identical folds).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import rasterio
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from config import BASINS, RESULTS, basin_dir, feature_paths, LOG_TRANSFORM
from terramind_linprobe import (
    BUFFER_PX,
    NEG_RATIO,
    PATCH_SIZE,
    SEED,
    SPATIAL_BLOCK_PX,
    S2_COMPOSITE_BASE,
    load_events_xy,
)

N_FOLDS = 5
CONTEXT_WINDOWS = (7, 37, 111)  # px: 210 m, ~1.1 km, ~3.3 km

# 1-based band indexes in the 12-band composite (B01..B09,B8A?,B11,B12 order:
# B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B11,B12)
HLS_BANDS_1BASED = [2, 3, 4, 9, 11, 12]   # B02,B03,B04,B8A,B11,B12
B04_1BASED, B08_1BASED, B8A_1BASED, B12_1BASED = 4, 8, 9, 12


def build_dataset(basin, negatives="uniform"):
    """Reproduce the benchmark dataset: rows/cols, labels, pixel features, blocks.

    negatives="uniform" reproduces the benchmark sampling exactly.
    negatives="constrained" restricts negative candidates to the positives'
    morphometric envelope (slope >= 10th percentile of slope at positives),
    the hard-task sensitivity variant.
    """
    dem_src = rasterio.open(basin_dir(basin) / "dem_30m.tif")
    height, width = dem_src.shape
    transform, crs = dem_src.transform, dem_src.crs

    pos_rows, pos_cols = load_events_xy(basin, crs, transform)
    in_bounds = (
        (pos_rows >= PATCH_SIZE // 2) & (pos_rows < height - PATCH_SIZE // 2)
        & (pos_cols >= PATCH_SIZE // 2) & (pos_cols < width - PATCH_SIZE // 2)
    )
    pos_rows, pos_cols = pos_rows[in_bounds], pos_cols[in_bounds]
    n_pos = pos_rows.size

    excluded = np.zeros((height, width), dtype=bool)
    for r, c in zip(pos_rows, pos_cols):
        r0, r1 = max(0, r - BUFFER_PX), min(height, r + BUFFER_PX + 1)
        c0, c1 = max(0, c - BUFFER_PX), min(width, c + BUFFER_PX + 1)
        excluded[r0:r1, c0:c1] = True
    excluded = excluded.ravel()

    rng = np.random.default_rng(SEED)
    n_neg_target = NEG_RATIO * n_pos
    half = PATCH_SIZE // 2
    if negatives == "uniform":
        neg_rows, neg_cols = [], []
        while len(neg_rows) < n_neg_target:
            r = int(rng.integers(half, height - half))
            c = int(rng.integers(half, width - half))
            if not excluded[r * width + c]:
                neg_rows.append(r); neg_cols.append(c)
        neg_rows, neg_cols = np.array(neg_rows), np.array(neg_cols)
    else:
        with rasterio.open(feature_paths(basin)["slope"]) as ssrc:
            slope = ssrc.read(1).astype(np.float32)
            snod = ssrc.nodata
        if snod is not None:
            slope = np.where(slope == snod, np.nan, slope)
        thr = float(np.nanpercentile(slope[pos_rows, pos_cols], 10))
        eligible = np.zeros((height, width), dtype=bool)
        eligible[half:height - half, half:width - half] = True
        eligible &= np.nan_to_num(slope, nan=-1.0) >= thr
        eligible &= ~excluded.reshape(height, width)
        flat_eligible = np.flatnonzero(eligible.ravel())
        print(f"[dataset] constrained negatives: slope >= {thr:.1f} deg "
              f"({flat_eligible.size} eligible px)")
        pick = rng.choice(flat_eligible, size=n_neg_target, replace=False)
        neg_rows, neg_cols = pick // width, pick % width

    all_rows = np.concatenate([pos_rows, neg_rows])
    all_cols = np.concatenate([pos_cols, neg_cols])
    y = np.concatenate([np.ones(n_pos), np.zeros(len(neg_rows))]).astype(np.int8)

    with np.load(RESULTS / f"{basin}_stack.npz", allow_pickle=False) as npz:
        X_all = npz["X"].astype(np.float32)
        valid_idx = npz["valid_idx"]
    flat_to_row = {int(idx): i for i, idx in enumerate(valid_idx)}
    flat_positions = all_rows * width + all_cols
    pixel_features = np.full((len(all_rows), X_all.shape[1]), np.nan, np.float32)
    valid_mask = np.zeros(len(all_rows), dtype=bool)
    for i, fp in enumerate(flat_positions):
        if int(fp) in flat_to_row:
            pixel_features[i] = X_all[flat_to_row[int(fp)]]
            valid_mask[i] = True
    all_rows, all_cols = all_rows[valid_mask], all_cols[valid_mask]
    pixel_features, y = pixel_features[valid_mask], y[valid_mask]

    n_cols_blocks = (width + SPATIAL_BLOCK_PX - 1) // SPATIAL_BLOCK_PX
    block_id = (all_rows // SPATIAL_BLOCK_PX) * n_cols_blocks + (all_cols // SPATIAL_BLOCK_PX)
    print(f"[dataset] {basin}: pos={int(y.sum())} neg={int((y == 0).sum())}")
    return all_rows, all_cols, y, pixel_features, block_id


def _window_stats(band, rows, cols, windows=CONTEXT_WINDOWS):
    """Point value plus nan-aware mean/std of `band` around each point."""
    n_pts = len(rows)
    point = band[rows, cols].astype(np.float32)
    stats = []
    for win in windows:
        half = win // 2
        means = np.empty(n_pts, np.float32)
        stds = np.empty(n_pts, np.float32)
        for i in range(n_pts):
            r, c = rows[i], cols[i]
            sl = band[max(0, r - half):r + half + 1, max(0, c - half):c + half + 1]
            means[i] = np.nanmean(sl)
            stds[i] = np.nanstd(sl)
        stats.append(means); stats.append(stds)
    return point, stats


def spectral_layers(basin, path=None):
    """Yield (name, full-resolution float32 array) for the 8 spectral layers.

    `path` overrides the default 2023 composite (used by the pre/post-event
    test, which swaps in an earlier composite for the same basin).
    """
    path = path or S2_COMPOSITE_BASE / f"{basin}_s2l2a_2023.tif"
    names = ["blue", "green", "red", "nir_narrow", "swir1", "swir2"]
    with rasterio.open(path) as src:
        for name, idx in zip(names, HLS_BANDS_1BASED):
            yield name, src.read(idx).astype(np.float32) / 10_000.0
        b04 = src.read(B04_1BASED).astype(np.float32) / 10_000.0
        b08 = src.read(B08_1BASED).astype(np.float32) / 10_000.0
        eps = 1e-6
        yield "ndvi", (b08 - b04) / (b08 + b04 + eps)
        del b08
        b8a = src.read(B8A_1BASED).astype(np.float32) / 10_000.0
        b12 = src.read(B12_1BASED).astype(np.float32) / 10_000.0
        yield "nbr", (b8a - b12) / (b8a + b12 + eps)


def spectral_features(basin, rows, cols):
    """Spectral-point (8) and spectral-context (48) features from the composite."""
    point_cols, point_names = [], []
    ctx_cols, ctx_names = [], []
    for name, band in spectral_layers(basin):
        point, stats = _window_stats(band, rows, cols)
        point_cols.append(point); point_names.append(name)
        ctx_cols.extend(stats)
        for win in CONTEXT_WINDOWS:
            ctx_names.append(f"{name}_m{win}"); ctx_names.append(f"{name}_s{win}")
        del band
    Xp = np.nan_to_num(np.stack(point_cols, axis=1), nan=0.0)
    Xc = np.nan_to_num(np.stack(ctx_cols, axis=1), nan=0.0)
    return Xp, point_names, Xc, ctx_names


def context_features(basin, rows, cols):
    """nan-aware mean/std of each of the 17 layers over CONTEXT_WINDOWS."""
    paths = feature_paths(basin)
    n_pts = len(rows)
    out, names = [], []
    for name, path in paths.items():
        with rasterio.open(path) as src:
            band = src.read(1).astype(np.float32)
            nodata = src.nodata
        if nodata is not None:
            band = np.where(band == nodata, np.nan, band)
        if name in LOG_TRANSFORM:
            band = np.log1p(np.clip(band, 0, None))
        h, w = band.shape
        for win in CONTEXT_WINDOWS:
            half = win // 2
            means = np.empty(n_pts, np.float32)
            stds = np.empty(n_pts, np.float32)
            for i in range(n_pts):
                r, c = rows[i], cols[i]
                sl = band[max(0, r - half):r + half + 1, max(0, c - half):c + half + 1]
                means[i] = np.nanmean(sl)
                stds[i] = np.nanstd(sl)
            out.append(means); out.append(stds)
            names.append(f"{name}_m{win}"); names.append(f"{name}_s{win}")
    feats = np.stack(out, axis=1)
    return np.nan_to_num(feats, nan=0.0), names


def fm_embeddings(basin, rows, cols, prithvi_path, which="all"):
    """Encode the FM pipelines in the current environment."""
    from terramind_linprobe import (
        BATCH_SIZE, PRITHVI_HLS_INDICES, PRITHVI_NAME, TERRAMIND_NAME,
        _encode_prithvi, build_encoder, encode_patches, extract_patches,
        extract_patches_s2,
    )
    dem_src = rasterio.open(basin_dir(basin) / "dem_30m.tif")
    s2_src = rasterio.open(S2_COMPOSITE_BASE / f"{basin}_s2l2a_2023.tif")
    out = {}
    print("[fm] extracting patches")
    dem_patches = extract_patches(dem_src, rows, cols)
    model, kind, dim, ns = build_encoder(TERRAMIND_NAME, True, ["DEM"])
    print("[fm] encoding TerraMind DEM")
    out["TM_DEM_ENV"] = encode_patches(kind, model, dim, dem_patches=dem_patches)
    del model
    s2_patches = extract_patches_s2(s2_src, rows, cols)
    model, kind, dim, ns = build_encoder(TERRAMIND_NAME, True, ["DEM", "S2L2A"])
    print("[fm] encoding TerraMind DEM+S2L2A")
    out["TM_MM_ENV"] = encode_patches(kind, model, dim,
                                      dem_patches=dem_patches,
                                      s2_patches=s2_patches)
    del model, s2_patches, dem_patches
    if which == "all":
        s2_hls = extract_patches_s2(s2_src, rows, cols,
                                    band_indices=PRITHVI_HLS_INDICES, scale=1.0)
        model, kind, dim, ns = build_encoder(PRITHVI_NAME, True, None,
                                             prithvi_path=prithvi_path)
        print("[fm] encoding Prithvi-EO-2.0-300M (fp32)")
        out["PRITHVI_ENV"] = _encode_prithvi(model, s2_hls, dim, BATCH_SIZE,
                                             ns, use_bf16=False)
        del model, s2_hls
    return out


def rf_fold_metrics(X, y, folds):
    rocs, prs = [], []
    for fi, (tr, te) in enumerate(folds):
        clf = RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, n_jobs=-1,
            random_state=SEED + fi, class_weight="balanced",
        )
        clf.fit(X[tr], y[tr])
        s = clf.predict_proba(X[te])[:, 1]
        rocs.append(float(roc_auc_score(y[te], s)))
        prs.append(float(average_precision_score(y[te], s)))
    return rocs, prs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--basin", required=True, choices=BASINS)
    ap.add_argument("--negatives", choices=("uniform", "constrained"),
                    default="uniform")
    ap.add_argument("--with-fms", choices=("all", "terramind"), default=None,
                    help="Also encode and evaluate FM pipelines in the current "
                         "environment (requires terratorch + weights). "
                         "'terramind' skips the slow Prithvi CPU encoding.")
    ap.add_argument("--prithvi-path",
                    default="/home/franciscoparrao/models/prithvi-300m")
    args = ap.parse_args()
    basin = args.basin

    t0 = perf_counter()
    rows, cols, y, X17, block_id = build_dataset(basin, negatives=args.negatives)
    if args.negatives == "uniform":
        folds = list(GroupKFold(n_splits=N_FOLDS).split(X17, y, groups=block_id))
    else:
        # constrained negatives concentrate in fewer blocks; stratify by class
        # (blocks stay intact) so every test fold contains both classes.
        folds = list(StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=False)
                     .split(X17, y, groups=block_id))

    Xs, spec_names, Xsc, sctx_names = spectral_features(basin, rows, cols)
    print(f"[spec] {Xs.shape[1]} spectral-point + {Xsc.shape[1]} spectral-context features")
    t1 = perf_counter()
    Xc, ctx_names = context_features(basin, rows, cols)
    print(f"[ctx] {Xc.shape[1]} context features in {perf_counter() - t1:.0f}s")

    sets = {
        "A": X17,
        "SPEC": Xs,
        "A_SPEC": np.hstack([X17, Xs]),
        "ACTX": np.hstack([X17, Xc]),
        "SCTX": np.hstack([Xs, Xsc]),
        "AFULL": np.hstack([X17, Xc, Xs, Xsc]),
    }
    if args.with_fms:
        sets.update(fm_embeddings(basin, rows, cols, args.prithvi_path,
                                  which=args.with_fms))
    results = {"basin": basin, "negatives": args.negatives,
               "n_pos": int(y.sum()), "n_neg": int((y == 0).sum()),
               "context_windows_px": list(CONTEXT_WINDOWS), "fold_metrics": {}}
    for name, X in sets.items():
        rocs, prs = rf_fold_metrics(X, y, folds)
        results["fold_metrics"][name] = {"roc": rocs, "pr": prs}
        print(f"[rf] {name:7s} roc={np.mean(rocs):.3f}±{np.std(rocs, ddof=1):.3f} "
              f"pr={np.mean(prs):.3f}")

    # univariate scar-sensitive indices, evaluated on each fold's test points
    ndvi_col = spec_names.index("ndvi")
    nbr_col = spec_names.index("nbr")
    for label, col in (("NDVI", ndvi_col), ("NBR", nbr_col)):
        aucs = [float(roc_auc_score(y[te], Xs[te, col])) for _, te in folds]
        results["fold_metrics"][f"univar_{label}"] = {"roc": aucs}
        print(f"[univar] {label} roc per fold mean={np.mean(aucs):.3f}")

    # reproduction check + paired comparisons vs stored FM runs (same folds).
    # Only meaningful for the uniform sampling the stored runs used.
    stored = {} if args.negatives == "constrained" else {
        "A_stored": f"{basin}_terramind_linprobe_spatial.json",
        "TM_MM": f"{basin}_terramind_linprobe_spatial_dem+s2l2a.json",
        "PRITHVI": f"{basin}_prithvi-300m_linprobe_spatial.json",
    }
    for key, fname in stored.items():
        p = RESULTS / fname
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        if key == "A_stored":
            ref = [f["roc_A"] for f in d["fold_results"]]
            diff = np.abs(np.array(ref) - np.array(results["fold_metrics"]["A"]["roc"]))
            results["reproduction_max_abs_diff_roc_A"] = float(diff.max())
            print(f"[check] reproduction of stored A: max |Δroc| = {diff.max():.4f}")
        else:
            results["fold_metrics"][key] = {
                "roc": [f["roc_B"] for f in d["fold_results"]],
                "pr": [f["pr_B"] for f in d["fold_results"]],
            }

    suffix = "_constrained" if args.negatives == "constrained" else ""
    out = RESULTS / f"{basin}_point_probes{suffix}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"[done] wrote {out.name} in {perf_counter() - t0:.0f}s total")


if __name__ == "__main__":
    main()
