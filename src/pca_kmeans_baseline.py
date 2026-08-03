"""PCA + MiniBatchKMeans baseline on the basin feature stack.

Estandariza features, ajusta PCA con un subsample, entrena MiniBatchKMeans para
varios K, rasteriza el mapa de clusters y reporta silhouette sobre subsample.
"""
from __future__ import annotations

import argparse
import json
from time import perf_counter

import numpy as np
import rasterio
from rasterio.transform import Affine
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from config import BASINS, DEFAULT_BASIN, RESULTS

K_VALUES = (5, 10, 15)
PCA_COMPONENTS = 10
PCA_SUBSAMPLE = 500_000
SILHOUETTE_SUBSAMPLE = 50_000
KMEANS_BATCH = 8192
KMEANS_MAX_ITER = 200
SEED = 42


def write_cluster_raster(out_path, labels_full, height, width, transform, crs_wkt):
    raster = labels_full.reshape(height, width).astype(np.int16)
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "int16",
        "nodata": -1,
        "transform": Affine(*transform),
        "crs": crs_wkt or None,
        "compress": "deflate",
        "tiled": True,
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
    print(f"[baseline] loading {stack_path}")
    with np.load(stack_path, allow_pickle=False) as npz:
        X = npz["X"].astype(np.float32)
        valid_idx = npz["valid_idx"]
        feature_names = list(npz["feature_names"])
        height = int(npz["height"])
        width = int(npz["width"])
        transform = npz["transform"]
        crs_wkt = str(npz["crs_wkt"])

    n_valid, n_features = X.shape
    n_total = height * width
    print(
        f"[baseline] X.shape={X.shape} features={feature_names} "
        f"valid={n_valid:,}/{n_total:,}"
    )

    t0 = perf_counter()
    scaler = StandardScaler(copy=False)
    X_std = scaler.fit_transform(X)
    print(f"[baseline] standardized in {perf_counter() - t0:.1f}s")

    pca_idx = rng.choice(n_valid, size=min(PCA_SUBSAMPLE, n_valid), replace=False)
    t0 = perf_counter()
    pca = PCA(n_components=PCA_COMPONENTS, random_state=SEED)
    pca.fit(X_std[pca_idx])
    explained = pca.explained_variance_ratio_
    cum = np.cumsum(explained)
    print(
        f"[baseline] PCA fit on {pca_idx.size:,} samples in "
        f"{perf_counter() - t0:.1f}s; cum_var@{PCA_COMPONENTS}={cum[-1]:.3f}"
    )

    t0 = perf_counter()
    Z = pca.transform(X_std).astype(np.float32)
    print(f"[baseline] PCA transform full in {perf_counter() - t0:.1f}s; Z.shape={Z.shape}")

    silhouette_idx = rng.choice(
        n_valid, size=min(SILHOUETTE_SUBSAMPLE, n_valid), replace=False
    )

    summary = {
        "basin": basin,
        "n_valid_pixels": int(n_valid),
        "n_features": int(n_features),
        "feature_names": feature_names,
        "pca_components": PCA_COMPONENTS,
        "pca_explained_variance_ratio": [float(v) for v in explained],
        "pca_cumulative": [float(v) for v in cum],
        "runs": [],
    }

    for k in K_VALUES:
        t0 = perf_counter()
        km = MiniBatchKMeans(
            n_clusters=k,
            random_state=SEED,
            batch_size=KMEANS_BATCH,
            max_iter=KMEANS_MAX_ITER,
            n_init=5,
            reassignment_ratio=0.005,
        )
        labels = km.fit_predict(Z)
        fit_t = perf_counter() - t0

        t0 = perf_counter()
        sil = float(
            silhouette_score(
                Z[silhouette_idx], labels[silhouette_idx], metric="euclidean"
            )
        )
        sil_t = perf_counter() - t0

        counts = np.bincount(labels, minlength=k).tolist()
        inertia = float(km.inertia_)
        print(
            f"[baseline] K={k} fit={fit_t:.1f}s sil={sil:.3f} "
            f"inertia={inertia:.0f} sil_t={sil_t:.1f}s"
        )

        labels_full = np.full(n_total, -1, dtype=np.int16)
        labels_full[valid_idx] = labels.astype(np.int16)
        out_tif = RESULTS / f"{basin}_kmeans_K{k:02d}.tif"
        write_cluster_raster(
            out_tif, labels_full, height, width, transform, crs_wkt
        )
        print(f"[baseline]   wrote {out_tif.name}")

        summary["runs"].append(
            {
                "k": k,
                "fit_seconds": fit_t,
                "silhouette_subsample": sil,
                "silhouette_subsample_size": int(silhouette_idx.size),
                "inertia": inertia,
                "cluster_counts": counts,
                "raster": out_tif.name,
            }
        )

    summary_path = RESULTS / f"{basin}_pca_kmeans_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[baseline] wrote {summary_path}")


if __name__ == "__main__":
    main()
