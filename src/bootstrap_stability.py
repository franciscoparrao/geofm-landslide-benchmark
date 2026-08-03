"""Intra-basin clustering stability via bootstrap.

For a given K, fits MiniBatchKMeans N times on 80% subsamples, predicts on a
fixed evaluation subsample, and reports pairwise Adjusted Rand Index between
bootstraps. Mean ARI ≈ partition stability.
"""
from __future__ import annotations

import argparse
import json
from time import perf_counter

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

from config import BASINS, DEFAULT_BASIN, RESULTS

DEFAULT_KS = (3, 5, 7, 10, 12)
N_BOOTSTRAPS = 20
SUBSAMPLE_RATIO = 0.8
PCA_COMPONENTS = 10
PCA_SUBSAMPLE = 500_000
EVAL_SUBSAMPLE = 50_000
KMEANS_BATCH = 8192
KMEANS_MAX_ITER = 200
SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--basin", default=DEFAULT_BASIN, choices=BASINS)
    parser.add_argument("--ks", type=int, nargs="+", default=list(DEFAULT_KS))
    parser.add_argument("--n-boot", type=int, default=N_BOOTSTRAPS)
    args = parser.parse_args()
    basin = args.basin
    ks = args.ks
    n_boot = args.n_boot

    rng = np.random.default_rng(SEED)
    stack_path = RESULTS / f"{basin}_stack.npz"
    print(f"[bootstrap] basin={basin} loading {stack_path.name}")
    with np.load(stack_path, allow_pickle=False) as npz:
        X = npz["X"].astype(np.float32)
    n_valid = X.shape[0]
    print(f"[bootstrap] X.shape={X.shape}")

    t0 = perf_counter()
    scaler = StandardScaler(copy=False)
    X_std = scaler.fit_transform(X)
    pca_idx = rng.choice(n_valid, size=min(PCA_SUBSAMPLE, n_valid), replace=False)
    pca = PCA(n_components=PCA_COMPONENTS, random_state=SEED)
    pca.fit(X_std[pca_idx])
    Z = pca.transform(X_std).astype(np.float32)
    print(f"[bootstrap] standardize+PCA in {perf_counter() - t0:.1f}s; Z.shape={Z.shape}")

    eval_idx = rng.choice(n_valid, size=min(EVAL_SUBSAMPLE, n_valid), replace=False)
    Z_eval = Z[eval_idx]
    sub_size = int(SUBSAMPLE_RATIO * n_valid)

    runs = []
    for k in ks:
        boot_labels = np.empty((n_boot, eval_idx.size), dtype=np.int16)
        t_start = perf_counter()
        for b in range(n_boot):
            sub_idx = rng.choice(n_valid, size=sub_size, replace=False)
            km = MiniBatchKMeans(
                n_clusters=k, random_state=SEED + b,
                batch_size=KMEANS_BATCH, max_iter=KMEANS_MAX_ITER,
                n_init=3, reassignment_ratio=0.005,
            )
            km.fit(Z[sub_idx])
            boot_labels[b] = km.predict(Z_eval).astype(np.int16)

        aris = []
        nmis = []
        for i in range(n_boot):
            for j in range(i + 1, n_boot):
                aris.append(adjusted_rand_score(boot_labels[i], boot_labels[j]))
                nmis.append(normalized_mutual_info_score(boot_labels[i], boot_labels[j]))
        aris = np.array(aris)
        nmis = np.array(nmis)
        elapsed = perf_counter() - t_start
        print(
            f"[bootstrap] K={k:2d} n_boot={n_boot} elapsed={elapsed:.1f}s "
            f"ARI mean={aris.mean():.3f} std={aris.std():.3f} "
            f"min={aris.min():.3f} max={aris.max():.3f} "
            f"NMI mean={nmis.mean():.3f}"
        )
        runs.append({
            "k": k, "n_bootstraps": n_boot,
            "ari_mean": float(aris.mean()), "ari_std": float(aris.std()),
            "ari_min": float(aris.min()), "ari_max": float(aris.max()),
            "ari_median": float(np.median(aris)),
            "nmi_mean": float(nmis.mean()), "nmi_std": float(nmis.std()),
            "elapsed_seconds": elapsed,
        })

    out = RESULTS / f"{basin}_bootstrap_stability.json"
    out.write_text(json.dumps({"basin": basin, "runs": runs}, indent=2))
    print(f"[bootstrap] wrote {out.name}")


if __name__ == "__main__":
    main()
