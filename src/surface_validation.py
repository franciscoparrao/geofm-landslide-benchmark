r"""Validate the susceptibility surfaces as maps, not as point rankings.

The benchmark scores models on labelled points. A susceptibility map is judged
differently: by how much of the inventory falls in how little of the terrain
ranked most susceptible \citep{ChungFabbri2003}. Two curves carry that test:

  success rate     the surface fitted on every point, scored against the same
                   inventory it was fitted on -- goodness of fit;
  prediction rate  an out-of-fold surface, scored against positives the model
                   producing each pixel never saw -- predictive skill.

The published surfaces (susceptibility_map.py) are fitted on the full point set,
so they can only give the first. The out-of-fold surface is built here from the
benchmark's own five GroupKFold splits over 10 km blocks: every pixel is
predicted by the fold model that held its block out. Blocks that contain no
benchmark point belong to no split; they are assigned to fold block_id mod 5,
which is out-of-sample for them under any assignment. The stitched surface has
seams at block boundaries, which is why it is not the map shown in the paper,
and is irrelevant to curves that only rank pixels.

Both models are compared twice. At native support the baseline is scored at
30 m and Prithvi on its 600 m grid, each against its own area. At common support
the baseline is averaged to the Prithvi grid and both are scored over the same
cells, excluding cells whose Sentinel-2 patch is more than 80% empty -- the
comparison that isolates the model from the support.

For each surface the output gives the area under the curve, the share of the
inventory captured by the top 10% and 20% of the area, and the density of events
per equal-area quintile class with its density ratio (share of events over
share of area), which should not decrease from the lowest class to the
highest.
"""
from __future__ import annotations

import argparse
import json
from time import perf_counter

import numpy as np
import rasterio
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

from config import BASINS, RESULTS, S2_COMPOSITE_BASE, basin_dir
from point_probes import build_dataset
from terramind_linprobe import (
    N_FOLDS, N_TREES, PRITHVI_NAME, SEED, SPATIAL_BLOCK_PX,
    embedding_cache_path, embedding_fingerprint, load_cached_embeddings,
)

MAP_DIR = RESULTS / "susceptibility_maps"
STRIDE = 20
GAP_THRESHOLD = 0.8
HLS_BAND = 2
CHUNK = 2_000_000
N_BINS = 20_000               # score histogram resolution for the curves
CURVE_POINTS = 101
QUINTILES = ("very low", "low", "moderate", "high", "very high")


def rf(n_jobs: int) -> RandomForestClassifier:
    """The benchmark's classifier; n_jobs does not change the fitted forest."""
    return RandomForestClassifier(
        n_estimators=N_TREES, min_samples_leaf=5, n_jobs=n_jobs,
        random_state=SEED, class_weight="balanced",
    )


def block_of(rows, cols, width):
    n_cols_blocks = (width + SPATIAL_BLOCK_PX - 1) // SPATIAL_BLOCK_PX
    return (rows // SPATIAL_BLOCK_PX) * n_cols_blocks + (cols // SPATIAL_BLOCK_PX)


def fold_splits(y, block_id):
    """The benchmark's primary splits, and a block -> held-out-fold lookup."""
    splits = list(GroupKFold(n_splits=N_FOLDS).split(np.zeros(len(y)), y,
                                                     groups=block_id))
    block_fold = {}
    for k, (_tr, te) in enumerate(splits):
        for b in np.unique(block_id[te]):
            block_fold[int(b)] = k
    return splits, block_fold


def pixel_folds(blocks, block_fold):
    lut_keys = np.fromiter(block_fold.keys(), dtype=np.int64)
    lut_vals = np.fromiter(block_fold.values(), dtype=np.int64)
    order = np.argsort(lut_keys)
    lut_keys, lut_vals = lut_keys[order], lut_vals[order]
    pos = np.clip(np.searchsorted(lut_keys, blocks), 0, len(lut_keys) - 1)
    hit = lut_keys[pos] == blocks
    return np.where(hit, lut_vals[pos], blocks % N_FOLDS), hit


# ------------------------------------------------------------- the surfaces
def oof_baseline(basin, rows, y, feats, block_id, splits, block_fold, n_jobs):
    with rasterio.open(basin_dir(basin) / "dem_30m.tif") as src:
        profile, height, width = src.profile, src.height, src.width
    with np.load(RESULTS / f"{basin}_stack.npz", allow_pickle=False) as npz:
        X_all = npz["X"].astype(np.float32, copy=False)
        valid_idx = npz["valid_idx"]
    pix_fold, _ = pixel_folds(block_of(valid_idx // width, valid_idx % width, width),
                              block_fold)
    surface = np.full(height * width, -1.0, dtype=np.float32)
    for k, (tr, _te) in enumerate(splits):
        t0 = perf_counter()
        clf = rf(n_jobs).fit(feats[tr], y[tr])
        sel = np.flatnonzero(pix_fold == k)
        for i in range(0, len(sel), CHUNK):
            s = sel[i:i + CHUNK]
            surface[valid_idx[s]] = clf.predict_proba(X_all[s])[:, 1]
        print(f"  [A oof] fold {k}: {len(sel):,} px in {perf_counter() - t0:.0f}s",
              flush=True)
    del X_all, valid_idx
    surface = surface.reshape(height, width)
    prof = dict(profile)
    prof.update(driver="GTiff", dtype="float32", count=1, nodata=np.float32(-1),
                compress="deflate", predictor=2, tiled=True,
                blockxsize=256, blockysize=256)
    with rasterio.open(MAP_DIR / f"{basin}_susceptibility_A_oof.tif", "w", **prof) as dst:
        dst.write(surface, 1)
    return surface


def oof_prithvi(basin, rows, cols, y, splits, block_fold, width, n_jobs):
    fp = embedding_fingerprint(PRITHVI_NAME, "pretrained", ["S2L2A"], rows, cols, y)
    emb = load_cached_embeddings(
        embedding_cache_path(basin, "prithvi-300m", "pretrained", ["S2L2A"]), fp)
    if emb is None:
        raise SystemExit("benchmark embeddings absent or fingerprint mismatch")
    grid = np.load(MAP_DIR / f"{basin}_grid_s{STRIDE}.npz")
    G = np.load(MAP_DIR / f"{basin}_grid_emb.npz")
    if not (np.array_equal(G["rows"], grid["rows"])
            and np.array_equal(G["cols"], grid["cols"])):
        raise SystemExit("grid embeddings do not match the grid")
    gemb = G["embeddings"]
    cell_fold, _ = pixel_folds(block_of(grid["rows"], grid["cols"], width), block_fold)
    scores = np.empty(len(gemb), dtype=np.float32)
    for k, (tr, _te) in enumerate(splits):
        clf = rf(n_jobs).fit(emb[tr], y[tr])
        sel = cell_fold == k
        scores[sel] = clf.predict_proba(gemb[sel])[:, 1]
    gr, gc = grid["grid_rows"], grid["grid_cols"]
    surface = np.full((len(gr), len(gc)), -1.0, dtype=np.float32)
    surface[np.searchsorted(gr, grid["rows"]), np.searchsorted(gc, grid["cols"])] = scores
    np.save(MAP_DIR / f"{basin}_susceptibility_PRITHVI_s{STRIDE}_oof.npy", surface)
    return surface


# ------------------------------------------------------------- support
def to_grid(A, gr, gc):
    """Average a 30 m surface over the STRIDE x STRIDE block of each grid cell."""
    half = STRIDE // 2
    out = np.full((len(gr), len(gc)), -1.0, dtype=np.float32)
    for i, rr in enumerate(gr):
        band = A[max(0, rr - half):rr + half]
        for j, cc in enumerate(gc):
            w = band[:, max(0, cc - half):cc + half]
            ok = w[w >= 0]
            if ok.size:
                out[i, j] = ok.mean()
    return out


def gap_mask(basin, grid, shape):
    half = 112
    with rasterio.open(S2_COMPOSITE_BASE / f"{basin}_s2l2a_2023.tif") as src:
        z = (src.read(HLS_BAND) == 0).astype(np.float32)
    ii = np.pad(z, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    rows, cols = grid["rows"], grid["cols"]
    r0 = np.clip(rows - half, 0, z.shape[0]); r1 = np.clip(rows + half, 0, z.shape[0])
    c0 = np.clip(cols - half, 0, z.shape[1]); c1 = np.clip(cols + half, 0, z.shape[1])
    area = np.maximum((r1 - r0) * (c1 - c0), 1)
    frac = (ii[r1, c1] - ii[r0, c1] - ii[r1, c0] + ii[r0, c0]) / area
    m = np.zeros(shape, dtype=bool)
    m[np.searchsorted(grid["grid_rows"], rows),
      np.searchsorted(grid["grid_cols"], cols)] = frac > GAP_THRESHOLD
    return m


def positives_on_grid(pr, pc, gr, gc):
    """Grid cell (i, j) of each positive; -1 where it falls outside the grid."""
    half = STRIDE // 2
    i = (pr - (gr[0] - half)) // STRIDE
    j = (pc - (gc[0] - half)) // STRIDE
    ok = (i >= 0) & (i < len(gr)) & (j >= 0) & (j < len(gc))
    return np.where(ok, i, -1), np.where(ok, j, -1)


# ------------------------------------------------------------- the metrics
def rate_curve(area_scores, event_scores, cell_km2):
    """Success- or prediction-rate curve and the class densities behind it.

    Ranking ties are split evenly, so a surface that is constant over a region
    neither gains nor loses from the order in which its pixels are listed.
    """
    area_scores = np.asarray(area_scores, dtype=np.float64)
    event_scores = np.asarray(event_scores, dtype=np.float64)
    n_area, n_ev = area_scores.size, event_scores.size

    # AUC = mean over events of the share of area ranked below it (ties halved).
    srt = np.sort(area_scores)
    below = np.searchsorted(srt, event_scores, side="left")
    tied = np.searchsorted(srt, event_scores, side="right") - below
    auc = float(np.mean((below + 0.5 * tied) / n_area))

    # Curve: cumulative share of events against cumulative share of area, from
    # the most susceptible end, on a fixed grid of area shares. Quantiles are
    # read off the array already sorted above instead of re-sorting per call.
    def q(p):
        return srt[min(n_area - 1, max(0, int(np.floor(p * (n_area - 1)))))]

    area_share = np.linspace(0, 1, CURVE_POINTS)
    captured = ([0.0] + [float(np.mean(event_scores >= q(1 - a))) for a in area_share[1:-1]]
                + [1.0])

    def top_share(share):
        return float(np.mean(event_scores >= q(1 - share)))

    edges = np.array([q(p) for p in (0.2, 0.4, 0.6, 0.8)])
    a_cls = np.digitize(area_scores, edges, right=True)
    e_cls = np.digitize(event_scores, edges, right=True)
    classes = []
    for k, name in enumerate(QUINTILES):
        a_k, e_k = int((a_cls == k).sum()), int((e_cls == k).sum())
        classes.append({
            "class": name, "area_share": a_k / n_area, "event_share": e_k / n_ev,
            "events": e_k, "area_km2": a_k * cell_km2,
            "density_per_100km2": 100 * e_k / (a_k * cell_km2) if a_k else None,
            "density_ratio": (e_k / n_ev) / (a_k / n_area) if a_k else None,
        })
    # An empty class (possible when ties fill a quantile) has no ratio.
    ratios = [c["density_ratio"] for c in classes if c["density_ratio"] is not None]
    return {
        "auc": auc, "n_area_cells": n_area, "n_events": n_ev,
        "top10_capture": top_share(0.10), "top20_capture": top_share(0.20),
        "monotone": bool(all(b >= a for a, b in zip(ratios, ratios[1:]))),
        "classes": classes,
        "curve": {"area_share": area_share.tolist(), "event_share": captured},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--basin", required=True, choices=BASINS)
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    basin = a.basin

    rows, cols, y, feats, block_id = build_dataset(basin, negatives="uniform")
    splits, block_fold = fold_splits(y, block_id)
    with rasterio.open(basin_dir(basin) / "dem_30m.tif") as src:
        width = src.width
        px_km2 = abs(src.transform.a * src.transform.e) / 1e6
    pr, pc = rows[y == 1], cols[y == 1]

    oof_path = MAP_DIR / f"{basin}_susceptibility_A_oof.tif"
    if oof_path.exists():
        with rasterio.open(oof_path) as src:
            A_oof = src.read(1)
        print(f"[A oof] reusing {oof_path.name}")
    else:
        A_oof = oof_baseline(basin, rows, y, feats, block_id, splits, block_fold, a.jobs)
    with rasterio.open(MAP_DIR / f"{basin}_susceptibility_A.tif") as src:
        A_fit = src.read(1)
    with rasterio.open(MAP_DIR / f"{basin}_susceptibility_PRITHVI_s{STRIDE}.tif") as src:
        P_fit = src.read(1)
    P_oof = oof_prithvi(basin, rows, cols, y, splits, block_fold, width, a.jobs)

    out = {"basin": basin, "n_positives": int(y.sum()), "folds": N_FOLDS,
           "block_km": SPATIAL_BLOCK_PX * 30 / 1000,
           "blocks_without_points": "assigned to fold block_id mod 5",
           "native": {}, "common": {}}

    # Native support: every model against its own area.
    for name, S in (("A_fit", A_fit), ("A_oof", A_oof)):
        valid = S >= 0
        ok = valid[pr, pc]
        out["native"][name] = rate_curve(S[valid], S[pr[ok], pc[ok]], px_km2)
        out["native"][name]["events_outside_surface"] = int((~ok).sum())

    grid = np.load(MAP_DIR / f"{basin}_grid_s{STRIDE}.npz")
    gr, gc = grid["grid_rows"], grid["grid_cols"]
    gi, gj = positives_on_grid(pr, pc, gr, gc)
    on_grid = gi >= 0
    cell_km2 = px_km2 * STRIDE ** 2
    for name, S in (("PRITHVI_fit", P_fit), ("PRITHVI_oof", P_oof)):
        valid = S >= 0
        ok = on_grid.copy()
        ok[on_grid] = valid[gi[on_grid], gj[on_grid]]
        out["native"][name] = rate_curve(S[valid], S[gi[ok], gj[ok]], cell_km2)
        out["native"][name]["events_outside_surface"] = int((~ok).sum())

    # Common support: both models over the same imagery-present grid cells.
    gaps = gap_mask(basin, grid, P_fit.shape)
    Ac = {"A_fit": to_grid(A_fit, gr, gc), "A_oof": to_grid(A_oof, gr, gc)}
    common = (P_fit >= 0) & (P_oof >= 0) & (Ac["A_fit"] >= 0) & (Ac["A_oof"] >= 0) & ~gaps
    ok = on_grid.copy()
    ok[on_grid] = common[gi[on_grid], gj[on_grid]]
    out["common"]["n_cells"] = int(common.sum())
    out["common"]["gap_cells_excluded"] = int((gaps & (P_fit >= 0)).sum())
    out["common"]["events_excluded"] = int((~ok).sum())
    for name, S in (("A_fit", Ac["A_fit"]), ("A_oof", Ac["A_oof"]),
                    ("PRITHVI_fit", P_fit), ("PRITHVI_oof", P_oof)):
        out["common"][name] = rate_curve(S[common], S[gi[ok], gj[ok]], cell_km2)

    path = RESULTS / f"{basin}_surface_validation.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"[write] {path.name}")
    for sup in ("native", "common"):
        for name, r in out[sup].items():
            if isinstance(r, dict):
                dr = " ".join(f"{c['density_ratio']:.2f}" if c["density_ratio"] is not None
                              else "-" for c in r["classes"])
                print(f"  {sup:6s} {name:12s} AUC={r['auc']:.3f}  top10={r['top10_capture']:.2f}"
                      f"  top20={r['top20_capture']:.2f}  DR=[{dr}]  mono={r['monotone']}")


if __name__ == "__main__":
    main()
