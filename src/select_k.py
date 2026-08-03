"""Grid search over K for MiniBatchKMeans + GMM-BIC.

Reports silhouette, Davies-Bouldin, Calinski-Harabasz, inertia (elbow) for
MiniBatchKMeans, and BIC for GMM, evaluated on a fixed subsample of the
basin's PCA-reduced features. Output: JSON + diagnostic plot.
"""
from __future__ import annotations

import argparse
import json
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from config import BASINS, DEFAULT_BASIN, RESULTS

K_VALUES = list(range(2, 21))
PCA_COMPONENTS = 10
PCA_SUBSAMPLE = 500_000
EVAL_SUBSAMPLE = 50_000
GMM_SUBSAMPLE = 50_000
KMEANS_BATCH = 8192
KMEANS_MAX_ITER = 100
KMEANS_N_INIT = 1
SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--basin", default=DEFAULT_BASIN, choices=BASINS)
    args = parser.parse_args()
    basin = args.basin

    rng = np.random.default_rng(SEED)
    stack_path = RESULTS / f"{basin}_stack.npz"
    print(f"[select_k] basin={basin} loading {stack_path.name}")
    with np.load(stack_path, allow_pickle=False) as npz:
        X = npz["X"].astype(np.float32)

    n_valid = X.shape[0]
    print(f"[select_k] X.shape={X.shape}")

    t0 = perf_counter()
    scaler = StandardScaler(copy=False)
    X_std = scaler.fit_transform(X)
    print(f"[select_k] standardized in {perf_counter() - t0:.1f}s")

    pca_idx = rng.choice(n_valid, size=min(PCA_SUBSAMPLE, n_valid), replace=False)
    pca = PCA(n_components=PCA_COMPONENTS, random_state=SEED)
    pca.fit(X_std[pca_idx])
    Z = pca.transform(X_std).astype(np.float32)
    print(
        f"[select_k] PCA done; cum_var@{PCA_COMPONENTS}="
        f"{np.cumsum(pca.explained_variance_ratio_)[-1]:.3f}"
    )

    eval_idx = rng.choice(n_valid, size=min(EVAL_SUBSAMPLE, n_valid), replace=False)
    Z_eval = Z[eval_idx]
    gmm_idx = rng.choice(n_valid, size=min(GMM_SUBSAMPLE, n_valid), replace=False)
    Z_gmm = Z[gmm_idx]

    rows = []
    for k in K_VALUES:
        t0 = perf_counter()
        km = MiniBatchKMeans(
            n_clusters=k,
            random_state=SEED,
            batch_size=KMEANS_BATCH,
            max_iter=KMEANS_MAX_ITER,
            n_init=KMEANS_N_INIT,
            reassignment_ratio=0.005,
        )
        km.fit(Z)
        fit_t = perf_counter() - t0
        labels_eval = km.predict(Z_eval)
        n_unique = len(set(labels_eval.tolist()))
        if n_unique < 2:
            sil = db = ch = float("nan")
        else:
            sil = float(silhouette_score(Z_eval, labels_eval, metric="euclidean"))
            db = float(davies_bouldin_score(Z_eval, labels_eval))
            ch = float(calinski_harabasz_score(Z_eval, labels_eval))
        inertia = float(km.inertia_)

        t1 = perf_counter()
        gmm = GaussianMixture(
            n_components=k, covariance_type="diag",
            random_state=SEED, max_iter=200, n_init=1, reg_covar=1e-4,
        )
        gmm.fit(Z_gmm)
        bic = float(gmm.bic(Z_gmm))
        gmm_t = perf_counter() - t1

        print(
            f"[select_k] K={k:2d} fit={fit_t:5.1f}s sil={sil:+.3f} "
            f"db={db:.3f} ch={ch:8.0f} inertia={inertia:11.0f} "
            f"bic={bic:11.0f} gmm_t={gmm_t:5.1f}s"
        )
        rows.append({
            "k": k, "kmeans_fit_seconds": fit_t,
            "silhouette": sil, "davies_bouldin": db,
            "calinski_harabasz": ch, "inertia": inertia,
            "gmm_bic": bic, "gmm_fit_seconds": gmm_t,
        })

    out_json = RESULTS / f"{basin}_select_k.json"
    out_json.write_text(json.dumps({"basin": basin, "rows": rows}, indent=2))
    print(f"[select_k] wrote {out_json.name}")

    ks = [r["k"] for r in rows]
    sils = [r["silhouette"] for r in rows]
    dbs = [r["davies_bouldin"] for r in rows]
    chs = [r["calinski_harabasz"] for r in rows]
    inertias = [r["inertia"] for r in rows]
    bics = [r["gmm_bic"] for r in rows]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    panels = [
        ("Silhouette (↑)", sils),
        ("Davies-Bouldin (↓)", dbs),
        ("Calinski-Harabasz (↑)", chs),
        ("Inertia (elbow)", inertias),
        ("GMM BIC (↓)", bics),
    ]
    for ax, (title, ys) in zip(axes.flat, panels):
        ax.plot(ks, ys, marker="o")
        ax.set_xlabel("K"); ax.set_title(title); ax.grid(alpha=0.3)
        ax.set_xticks(ks[::2])
    axes[1, 2].axis("off")
    fig.suptitle(f"Selección de K — {basin}", fontsize=13)
    out_png = RESULTS / f"{basin}_select_k.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"[select_k] wrote {out_png.name}")


if __name__ == "__main__":
    main()
