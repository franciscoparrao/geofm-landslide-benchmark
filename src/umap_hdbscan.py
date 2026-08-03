"""UMAP manifold + HDBSCAN density clustering — alternativa sin K fijo.

Fits UMAP (17 raw features → 5 manifold dims) and HDBSCAN on a 100k-pixel
subsample, then assigns labels to a 1M-pixel evaluation subsample via
approximate_predict for spatial visualization. Reports n_clusters, noise
share, cluster signatures, silhouette, and NMI vs the K-means baseline.

PoC scale: full-raster prediction is left out (UMAP.transform on 10M+ pixels
is the bottleneck) — the 1M subsample suffices for comparison with K-means.
"""
from __future__ import annotations

import argparse
import json
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import umap
from rasterio.transform import Affine
from sklearn.cluster import HDBSCAN
from sklearn.metrics import (
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from config import BASINS, DEFAULT_BASIN, RESULTS

FIT_SUBSAMPLE = 100_000
EVAL_SUBSAMPLE = 1_000_000
UMAP_COMPONENTS = 5
UMAP_NEIGHBORS = 30
UMAP_MIN_DIST = 0.0
HDBSCAN_MIN_CLUSTER_SIZE = 1000
HDBSCAN_MIN_SAMPLES = 50
SILHOUETTE_SUBSAMPLE = 30_000
SEED = 42


def write_cluster_raster(out_path, labels_full, height, width, transform, crs_wkt):
    raster = labels_full.reshape(height, width).astype(np.int16)
    profile = {
        "driver": "GTiff", "height": height, "width": width,
        "count": 1, "dtype": "int16", "nodata": -1,
        "transform": Affine(*transform), "crs": crs_wkt or None,
        "compress": "deflate", "tiled": True,
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(raster, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--basin", default=DEFAULT_BASIN, choices=BASINS)
    args = parser.parse_args()
    basin = args.basin

    rng = np.random.default_rng(SEED)
    stack_path = RESULTS / f"{basin}_stack.npz"
    print(f"[umap_hdb] basin={basin} loading {stack_path.name}")
    with np.load(stack_path, allow_pickle=False) as npz:
        X = npz["X"].astype(np.float32)
        valid_idx = npz["valid_idx"]
        feature_names = list(npz["feature_names"])
        height = int(npz["height"])
        width = int(npz["width"])
        transform = npz["transform"]
        crs_wkt = str(npz["crs_wkt"])

    n_valid = X.shape[0]
    n_total = height * width
    print(f"[umap_hdb] X.shape={X.shape}")

    t0 = perf_counter()
    scaler = StandardScaler(copy=False)
    X_std = scaler.fit_transform(X)
    print(f"[umap_hdb] standardized in {perf_counter() - t0:.1f}s")

    fit_idx = rng.choice(n_valid, size=min(FIT_SUBSAMPLE, n_valid), replace=False)
    X_fit = X_std[fit_idx]
    eval_idx = rng.choice(n_valid, size=min(EVAL_SUBSAMPLE, n_valid), replace=False)
    X_eval = X_std[eval_idx]

    t0 = perf_counter()
    reducer = umap.UMAP(
        n_components=UMAP_COMPONENTS, n_neighbors=UMAP_NEIGHBORS,
        min_dist=UMAP_MIN_DIST, random_state=SEED, verbose=False,
        low_memory=True,
    )
    Z_fit = reducer.fit_transform(X_fit)
    print(
        f"[umap_hdb] UMAP fit in {perf_counter() - t0:.1f}s; "
        f"Z_fit.shape={Z_fit.shape}"
    )

    t0 = perf_counter()
    clusterer = HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        cluster_selection_method="eom",
        n_jobs=-1,
    )
    labels_fit = clusterer.fit_predict(Z_fit)
    print(f"[umap_hdb] HDBSCAN fit in {perf_counter() - t0:.1f}s")

    n_clusters = int(labels_fit.max() + 1) if labels_fit.max() >= 0 else 0
    n_noise = int((labels_fit == -1).sum())
    counts = np.bincount(labels_fit[labels_fit >= 0], minlength=n_clusters).tolist()
    print(
        f"[umap_hdb] subsample fit: n_clusters={n_clusters} "
        f"noise={n_noise} ({100 * n_noise / labels_fit.size:.1f}%)"
    )
    print(f"[umap_hdb]   cluster sizes (fit): {counts}")

    if n_clusters >= 2:
        sil_idx = rng.choice(
            labels_fit.size, size=min(SILHOUETTE_SUBSAMPLE, labels_fit.size),
            replace=False,
        )
        sil_mask = labels_fit[sil_idx] >= 0
        if sil_mask.sum() >= 2 and len(set(labels_fit[sil_idx][sil_mask])) >= 2:
            sil = float(silhouette_score(
                Z_fit[sil_idx][sil_mask],
                labels_fit[sil_idx][sil_mask],
                metric="euclidean",
            ))
        else:
            sil = float("nan")
    else:
        sil = float("nan")
    print(f"[umap_hdb] silhouette (non-noise, fit subsample) = {sil:.3f}")

    t0 = perf_counter()
    Z_eval = reducer.transform(X_eval)
    print(f"[umap_hdb] UMAP transform eval in {perf_counter() - t0:.1f}s")

    t0 = perf_counter()
    nonnoise_mask = labels_fit >= 0
    if nonnoise_mask.sum() < 10:
        raise SystemExit("too few non-noise points to train KNN; loosen HDBSCAN params")
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    knn.fit(Z_fit[nonnoise_mask], labels_fit[nonnoise_mask])
    labels_eval = knn.predict(Z_eval).astype(np.int16)
    n_eval_noise = 0
    print(
        f"[umap_hdb] KNN(5) label propagation in {perf_counter() - t0:.1f}s; "
        f"all {labels_eval.size} eval points labeled"
    )

    print(f"\n[umap_hdb] cluster signatures (z-score by cluster, eval subsample)")
    feat_mean = X.mean(axis=0); feat_std = X.std(axis=0) + 1e-9
    signatures = []
    for c in range(n_clusters):
        sel = labels_eval == c
        n = int(sel.sum())
        if n == 0:
            signatures.append({"id": c, "size": 0})
            continue
        means = X[eval_idx][sel].mean(axis=0)
        z = (means - feat_mean) / feat_std
        signatures.append({
            "id": c, "size": n, "share_of_eval": n / labels_eval.size,
            "mean": [float(v) for v in means],
            "zscore": [float(v) for v in z],
        })
    header = "cluster   size   share%  " + "  ".join(f"{n[:6]:>6}" for n in feature_names)
    print(header)
    for s in signatures:
        if s["size"] == 0:
            continue
        row = (
            f"  c{s['id']:02d}  {s['size']:>6}  {100 * s['share_of_eval']:5.1f}  "
            + "  ".join(f"{v:+6.2f}" for v in s["zscore"])
        )
        print(row)

    km_path = RESULTS / f"{basin}_kmeans_K05.tif"
    with rasterio.open(km_path) as src:
        km_full = src.read(1).reshape(-1)
    km_eval = km_full[valid_idx[eval_idx]]
    common = (labels_eval >= 0) & (km_eval >= 0)
    if common.sum() >= 100:
        nmi_vs_kmeans = float(normalized_mutual_info_score(
            km_eval[common], labels_eval[common]
        ))
    else:
        nmi_vs_kmeans = float("nan")
    print(f"\n[umap_hdb] NMI vs K-means K=5 (non-noise, eval) = {nmi_vs_kmeans:.3f}")

    labels_full = np.full(n_total, -1, dtype=np.int16)
    labels_full[valid_idx[eval_idx]] = labels_eval.astype(np.int16)
    out_tif = RESULTS / f"{basin}_umap_hdbscan_eval.tif"
    write_cluster_raster(out_tif, labels_full, height, width, transform, crs_wkt)
    print(f"[umap_hdb] wrote {out_tif.name}")

    summary = {
        "basin": basin, "fit_subsample": FIT_SUBSAMPLE, "eval_subsample": EVAL_SUBSAMPLE,
        "umap": {
            "n_components": UMAP_COMPONENTS, "n_neighbors": UMAP_NEIGHBORS,
            "min_dist": UMAP_MIN_DIST,
        },
        "hdbscan": {
            "min_cluster_size": HDBSCAN_MIN_CLUSTER_SIZE,
            "min_samples": HDBSCAN_MIN_SAMPLES,
            "cluster_selection_method": "eom",
        },
        "n_clusters": n_clusters, "fit_noise_share": n_noise / labels_fit.size,
        "eval_noise_share": n_eval_noise / labels_eval.size,
        "silhouette_fit": sil,
        "nmi_vs_kmeans_K5_eval": nmi_vs_kmeans,
        "feature_names": feature_names,
        "cluster_signatures": signatures,
    }
    out_json = RESULTS / f"{basin}_umap_hdbscan_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"[umap_hdb] wrote {out_json.name}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    sample_plot = rng.choice(Z_fit.shape[0], size=min(20_000, Z_fit.shape[0]),
                             replace=False)
    ax = axes[0]
    pal = plt.colormaps["tab20"](
        np.linspace(0, 1, max(n_clusters, 1))
    )
    for c in range(n_clusters):
        m = labels_fit[sample_plot] == c
        if m.any():
            ax.scatter(Z_fit[sample_plot][m, 0], Z_fit[sample_plot][m, 1],
                       s=2, alpha=0.5, color=pal[c], label=f"c{c}")
    m = labels_fit[sample_plot] == -1
    if m.any():
        ax.scatter(Z_fit[sample_plot][m, 0], Z_fit[sample_plot][m, 1],
                   s=1, alpha=0.3, color="gray", label="noise")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.set_title(f"{basin} — UMAP({UMAP_COMPONENTS}D) + HDBSCAN labels")
    ax.legend(fontsize=7, loc="best", markerscale=3)

    ax = axes[1]
    ax.bar(range(n_clusters), counts, color=pal[:n_clusters])
    ax.set_xlabel("cluster id"); ax.set_ylabel("size (fit subsample)")
    ax.set_title(
        f"n_clusters={n_clusters}  noise={100 * n_noise / labels_fit.size:.1f}%  "
        f"NMI(K-means K=5)={nmi_vs_kmeans:.3f}"
    )
    fig.suptitle(f"UMAP+HDBSCAN — {basin}", fontsize=13)
    out_png = RESULTS / f"{basin}_umap_hdbscan.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"[umap_hdb] wrote {out_png.name}")


if __name__ == "__main__":
    main()
