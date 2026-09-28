"""Pre- versus post-event composite test for the scar-leakage caveat (Huasco).

Huasco's inventory falls in three discrete rainfall years: 2015 (78 events),
2017 (145) and 2020 (73). A Sentinel-2 composite built from calendar 2016 sits
between them, so it is genuinely PRE-EVENT for the 218 events of 2017 and 2020
(no scar can exist in the imagery) while the 2023 composite used in the paper is
post-event for all of them.

Holding the events, the negatives, the folds and the classifier fixed, and
swapping only the composite, isolates the contribution of post-event scar
signal to the spectral pipelines:

    pre-event (2016)  vs  post-event (2023)   on the same 218 events

If the post-event composite discriminates substantially better, residual scar
signal is doing work. If the two are indistinguishable, the spectral skill is
prospective, not retrospective.

Reported per composite: the spectral-point probe (SPEC), the spectral context
probe (SCTX), the manual baseline A (composite-independent, a sanity anchor)
and univariate NDVI/NBR AUCs. Paired fold-level t-intervals throughout.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pyproj
import rasterio
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from config import RESULTS, basin_dir
from point_probes import (
    CONTEXT_WINDOWS, N_FOLDS, _window_stats, rf_fold_metrics, spectral_layers,
)
from terramind_linprobe import (
    BUFFER_PX, INVENTORY_BASE, NEG_RATIO, PATCH_SIZE, SEED, SPATIAL_BLOCK_PX,
    S2_COMPOSITE_BASE, lookup_pixel_features, sample_negatives,
)

BASIN = "06_rio_huasco"
PRE_TAG = "2016_matched"   # composite between the 2015 and 2017 event waves
POST_TAG = "2023_matched"  # same months, same harmonization, post-event
PRE_EVENT_YEARS = (2017, 2020)   # events the PRE_YEAR composite predates


def load_events_with_year(basin, raster_crs, raster_transform):
    """Positive events as (rows, cols, years) from the dated basin inventory."""
    lats, lons, years = [], [], []
    with open(INVENTORY_BASE / f"{basin}.csv") as f:
        for r in csv.DictReader(f):
            lats.append(float(r["lat"]))
            lons.append(float(r["lon"]))
            years.append(int(float(r["year"])))
    tf = pyproj.Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
    xs, ys = tf.transform(lons, lats)
    inv = ~raster_transform
    cols, rows = inv * (np.array(xs), np.array(ys))
    return (np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64),
            np.array(years, dtype=int))


def build_dataset_dated(basin, keep_years):
    """Benchmark-equivalent dataset restricted to events of `keep_years`."""
    dem_src = rasterio.open(basin_dir(basin) / "dem_30m.tif")
    height, width = dem_src.shape
    pr, pc_, yrs = load_events_with_year(basin, dem_src.crs, dem_src.transform)

    in_bounds = (
        (pr >= PATCH_SIZE // 2) & (pr < height - PATCH_SIZE // 2)
        & (pc_ >= PATCH_SIZE // 2) & (pc_ < width - PATCH_SIZE // 2)
    )
    pr, pc_, yrs = pr[in_bounds], pc_[in_bounds], yrs[in_bounds]
    keep = np.isin(yrs, keep_years)
    pos_rows, pos_cols = pr[keep], pc_[keep]
    n_pos = pos_rows.size
    print(f"[dataset] {basin}: {n_pos} positivos de los años {keep_years} "
          f"(de {pr.size} en total)")

    # exclusion buffer around EVERY inventoried event, not only the kept ones,
    # so negatives are never drawn next to an out-of-scope positive
    excluded = np.zeros((height, width), dtype=bool)
    for r, c in zip(pr, pc_):
        r0, r1 = max(0, r - BUFFER_PX), min(height, r + BUFFER_PX + 1)
        c0, c1 = max(0, c - BUFFER_PX), min(width, c + BUFFER_PX + 1)
        excluded[r0:r1, c0:c1] = True
    excluded = excluded.ravel()

    # Same in-basin, completeness-matched sampler as the benchmark. Drawing
    # negatives from the rectangular bounding box, as this test used to,
    # reintroduces the edge artefact the rest of the study removed: the DEM
    # codes out-of-basin as literal 0, so those negatives carry systematically
    # more zeros in their neighbourhood than the in-basin positives, and a
    # context probe reads that difference directly.
    rng = np.random.default_rng(SEED)
    neg_rows, neg_cols, _thr = sample_negatives(
        dem_src, pos_rows, pos_cols, excluded, NEG_RATIO * n_pos, rng
    )

    all_rows = np.concatenate([pos_rows, neg_rows])
    all_cols = np.concatenate([pos_cols, neg_cols])
    y = np.concatenate([np.ones(n_pos), np.zeros(len(neg_rows))]).astype(np.int8)

    with np.load(RESULTS / f"{basin}_stack.npz", allow_pickle=False) as npz:
        X_all = npz["X"].astype(np.float32, copy=False)
        valid_idx = npz["valid_idx"]
    feats, mask = lookup_pixel_features(
        X_all, valid_idx, all_rows.astype(np.int64) * width + all_cols
    )
    del X_all, valid_idx
    all_rows, all_cols, feats, y = all_rows[mask], all_cols[mask], feats[mask], y[mask]

    n_cols_blocks = (width + SPATIAL_BLOCK_PX - 1) // SPATIAL_BLOCK_PX
    block_id = (all_rows // SPATIAL_BLOCK_PX) * n_cols_blocks + (all_cols // SPATIAL_BLOCK_PX)
    print(f"[dataset] pos={int(y.sum())} neg={int((y == 0).sum())}")
    return all_rows, all_cols, y, feats, block_id


def spectral_sets(basin, tag, rows, cols):
    """Spectral-point (8) and spectral-context (56) features for one composite."""
    path = S2_COMPOSITE_BASE / f"{basin}_s2l2a_{tag}.tif"
    if not path.exists():
        raise SystemExit(f"falta el composite {path}")
    point_cols, ctx_cols, names = [], [], []
    for name, band in spectral_layers(basin, path=path):
        pt, stats = _window_stats(band, rows, cols)
        point_cols.append(pt); ctx_cols.extend(stats); names.append(name)
        del band
    Xp = np.nan_to_num(np.stack(point_cols, axis=1), nan=0.0)
    Xc = np.nan_to_num(np.stack(ctx_cols, axis=1), nan=0.0)
    return Xp, Xc, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--basin", default=BASIN)
    args = ap.parse_args()
    t0 = perf_counter()

    rows, cols, y, X17, block_id = build_dataset_dated(args.basin, PRE_EVENT_YEARS)
    folds = list(GroupKFold(n_splits=N_FOLDS).split(X17, y, groups=block_id))

    out = {"basin": args.basin, "kept_event_years": list(PRE_EVENT_YEARS),
           "n_pos": int(y.sum()), "n_neg": int((y == 0).sum()),
           "pre_tag": PRE_TAG, "post_tag": POST_TAG, "fold_metrics": {}}

    roc, pr_ = rf_fold_metrics(X17, y, folds)
    out["fold_metrics"]["A"] = {"roc": roc, "pr": pr_}
    print(f"[rf] A (composite-independiente)  roc={np.mean(roc):.3f} pr={np.mean(pr_):.3f}")

    for tag, comp in (("pre", PRE_TAG), ("post", POST_TAG)):
        Xp, Xc, names = spectral_sets(args.basin, comp, rows, cols)
        for key, X in (("SPEC", Xp), ("SCTX", np.hstack([Xp, Xc]))):
            roc, pr_ = rf_fold_metrics(X, y, folds)
            out["fold_metrics"][f"{key}_{tag}"] = {"roc": roc, "pr": pr_}
            print(f"[rf] {key}_{tag} ({comp})  roc={np.mean(roc):.3f} pr={np.mean(pr_):.3f}")
        for idx, lab in ((names.index("ndvi"), "NDVI"), (names.index("nbr"), "NBR")):
            aucs = [float(roc_auc_score(y[te], Xp[te, idx])) for _, te in folds]
            out["fold_metrics"][f"univar_{lab}_{tag}"] = {"roc": aucs}
            print(f"[univar] {lab}_{tag} roc={np.mean(aucs):.3f}")

    path = RESULTS / f"{args.basin}_pre_post_event.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"[done] {path.name} en {perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
