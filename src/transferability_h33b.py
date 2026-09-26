"""H3.3b — transferibilidad inter-cuenca.

Train a susceptibility classifier on Huasco, predict on Bueno (zero-shot).
The cluster feature is computed by fitting StandardScaler+PCA+KMeans on Huasco
ONLY and applying that same pipeline to Bueno's pixels — this gives a single
typology shared across basins so the cluster id is comparable.

Compares:
  A: 17 raw features (subject to distribution shift Huasco→Bueno)
  B: 17 raw features + cluster_K05 one-hot (cluster fit only on Huasco)
  C: cluster_K05 one-hot only (does the typology alone transfer?)
"""
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pyproj
import rasterio
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from config import INVENTORY_BASE, RESULTS, basin_dir

SOURCE_BASIN = "06_rio_huasco"
TARGET_BASIN = "13_rio_bueno"
NEG_RATIO = 5
BUFFER_PX = 5
N_TREES = 300
SEED = 42
K = 5
PCA_COMPONENTS = 10
PCA_SUBSAMPLE = 500_000
KMEANS_BATCH = 8192
KMEANS_MAX_ITER = 200
N_BOOTSTRAP = 1000


def load_basin_stack(basin):
    path = RESULTS / f"{basin}_stack.npz"
    with np.load(path, allow_pickle=False) as npz:
        X = npz["X"].astype(np.float32)
        valid_idx = npz["valid_idx"]
        feat_names = list(npz["feature_names"])
        height = int(npz["height"])
        width = int(npz["width"])
    cluster_path = RESULTS / f"{basin}_kmeans_K{K:02d}.tif"
    with rasterio.open(cluster_path) as src:
        crs = src.crs; transform = src.transform
    return X, valid_idx, feat_names, height, width, crs, transform


def load_events_xy(basin, raster_crs, raster_transform):
    csv_path = INVENTORY_BASE / f"{basin}.csv"
    rows = []
    with open(csv_path) as f:
        header = f.readline().strip().split(",")
        lat_idx = header.index("lat"); lon_idx = header.index("lon")
        for line in f:
            parts = line.strip().split(",")
            if len(parts) <= max(lat_idx, lon_idx):
                continue
            rows.append((float(parts[lat_idx]), float(parts[lon_idx])))
    transformer = pyproj.Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
    xs, ys = transformer.transform([r[1] for r in rows], [r[0] for r in rows])
    inv = ~raster_transform
    cols, rrows = inv * (np.array(xs), np.array(ys))
    return np.array(rrows, dtype=np.int64), np.array(cols, dtype=np.int64)


def build_dataset(basin, X_full, valid_idx, height, width, crs, transform,
                  cluster_full, rng):
    pos_rows, pos_cols = load_events_xy(basin, crs, transform)
    in_bounds = (
        (pos_rows >= 0) & (pos_rows < height)
        & (pos_cols >= 0) & (pos_cols < width)
    )
    pos_rows = pos_rows[in_bounds]; pos_cols = pos_cols[in_bounds]
    pos_flat = pos_rows * width + pos_cols
    valid_mask = np.zeros(height * width, dtype=bool)
    valid_mask[valid_idx] = True
    pos_clusters = cluster_full[pos_flat]
    pos_valid = valid_mask[pos_flat] & (pos_clusters >= 0)
    pos_flat = pos_flat[pos_valid]
    n_pos = pos_flat.size

    excluded = np.zeros(height * width, dtype=bool)
    rows_grid = pos_flat // width; cols_grid = pos_flat % width
    for dr in range(-BUFFER_PX, BUFFER_PX + 1):
        for dc in range(-BUFFER_PX, BUFFER_PX + 1):
            r2 = rows_grid + dr; c2 = cols_grid + dc
            ok = (r2 >= 0) & (r2 < height) & (c2 >= 0) & (c2 < width)
            excluded[(r2[ok] * width + c2[ok])] = True
    candidate = valid_mask & (cluster_full >= 0) & (~excluded)
    candidate_idx = np.flatnonzero(candidate)
    n_neg = NEG_RATIO * n_pos
    n_neg = min(n_neg, candidate_idx.size)
    neg_flat = rng.choice(candidate_idx, size=n_neg, replace=False)

    flat_to_row = {int(idx): i for i, idx in enumerate(valid_idx)}
    pos_X_idx = np.array([flat_to_row[int(p)] for p in pos_flat])
    neg_X_idx = np.array([flat_to_row[int(p)] for p in neg_flat])
    X_pos = X_full[pos_X_idx]; X_neg = X_full[neg_X_idx]
    cl_pos = cluster_full[pos_flat]; cl_neg = cluster_full[neg_flat]
    X = np.vstack([X_pos, X_neg])
    cl = np.concatenate([cl_pos, cl_neg]).astype(int)
    y = np.concatenate([np.ones(n_pos), np.zeros(n_neg)]).astype(np.int8)
    return X, cl, y, n_pos, n_neg


def one_hot(cl, k=K):
    out = np.zeros((cl.size, k), dtype=np.float32)
    out[np.arange(cl.size), cl] = 1.0
    return out


def auc_with_ci(y, s, rng, n=N_BOOTSTRAP):
    roc = roc_auc_score(y, s); pr = average_precision_score(y, s)
    n_obs = y.size
    boot_roc = np.empty(n); boot_pr = np.empty(n)
    for b in range(n):
        ix = rng.integers(0, n_obs, size=n_obs)
        if y[ix].sum() == 0 or y[ix].sum() == n_obs:
            boot_roc[b] = np.nan; boot_pr[b] = np.nan; continue
        boot_roc[b] = roc_auc_score(y[ix], s[ix])
        boot_pr[b] = average_precision_score(y[ix], s[ix])
    boot_roc = boot_roc[~np.isnan(boot_roc)]
    boot_pr = boot_pr[~np.isnan(boot_pr)]
    return {
        "roc": float(roc), "pr": float(pr),
        "roc_ci": [float(np.percentile(boot_roc, 2.5)), float(np.percentile(boot_roc, 97.5))],
        "pr_ci": [float(np.percentile(boot_pr, 2.5)), float(np.percentile(boot_pr, 97.5))],
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    print(f"[h33b] source={SOURCE_BASIN}  target={TARGET_BASIN}")

    Xs, sidx, fnames, sh, sw, scrs, str_ = load_basin_stack(SOURCE_BASIN)
    Xt, tidx, fnamet, th, tw, tcrs, ttr = load_basin_stack(TARGET_BASIN)
    assert fnames == fnamet
    print(f"[h33b] Huasco stack {Xs.shape}  Bueno stack {Xt.shape}")

    t0 = perf_counter()
    scaler = StandardScaler(copy=False).fit(Xs)
    Zs_std = scaler.transform(Xs)
    Zt_std = scaler.transform(Xt)
    pca_idx = rng.choice(Xs.shape[0], size=min(PCA_SUBSAMPLE, Xs.shape[0]),
                         replace=False)
    pca = PCA(n_components=PCA_COMPONENTS, random_state=SEED).fit(Zs_std[pca_idx])
    Zs = pca.transform(Zs_std).astype(np.float32)
    Zt = pca.transform(Zt_std).astype(np.float32)
    km = MiniBatchKMeans(
        n_clusters=K, random_state=SEED, batch_size=KMEANS_BATCH,
        max_iter=KMEANS_MAX_ITER, n_init=5, reassignment_ratio=0.005,
    ).fit(Zs)
    cl_s = km.predict(Zs).astype(np.int16)
    cl_t = km.predict(Zt).astype(np.int16)
    print(f"[h33b] shared pipeline fit+predict in {perf_counter() - t0:.1f}s")

    cluster_full_s = np.full(sh * sw, -1, dtype=np.int16)
    cluster_full_s[sidx] = cl_s
    cluster_full_t = np.full(th * tw, -1, dtype=np.int16)
    cluster_full_t[tidx] = cl_t

    Xs_data, cls_data, ys, n_pos_s, n_neg_s = build_dataset(
        SOURCE_BASIN, Xs, sidx, sh, sw, scrs, str_, cluster_full_s, rng
    )
    Xt_data, clt_data, yt, n_pos_t, n_neg_t = build_dataset(
        TARGET_BASIN, Xt, tidx, th, tw, tcrs, ttr, cluster_full_t, rng
    )
    print(f"[h33b] Huasco dataset pos={n_pos_s} neg={n_neg_s}")
    print(f"[h33b] Bueno  dataset pos={n_pos_t} neg={n_neg_t}")

    cls_oh = one_hot(cls_data); clt_oh = one_hot(clt_data)
    XB_train = np.hstack([Xs_data, cls_oh])
    XB_test = np.hstack([Xt_data, clt_oh])
    XC_train = cls_oh; XC_test = clt_oh

    results = {}
    for label, Xtr, Xte in [
        ("A_features_only", Xs_data, Xt_data),
        ("B_features_plus_cluster", XB_train, XB_test),
        ("C_cluster_only", XC_train, XC_test),
    ]:
        rf = RandomForestClassifier(
            n_estimators=N_TREES, max_depth=None, min_samples_leaf=5,
            n_jobs=-1, random_state=SEED, class_weight="balanced",
        ).fit(Xtr, ys)
        s_self = rf.predict_proba(Xtr)[:, 1]
        s_transfer = rf.predict_proba(Xte)[:, 1]
        m_self = auc_with_ci(ys, s_self, rng)
        m_trans = auc_with_ci(yt, s_transfer, rng)
        print(
            f"[h33b] {label:<26} TRAIN huasco "
            f"ROC={m_self['roc']:.3f}  TEST bueno ROC={m_trans['roc']:.3f} "
            f"CI=[{m_trans['roc_ci'][0]:.3f},{m_trans['roc_ci'][1]:.3f}] "
            f"PR={m_trans['pr']:.3f}"
        )
        results[label] = {
            "train_self": m_self,
            "test_transfer": m_trans,
            "n_features": int(Xtr.shape[1]),
        }

    summary = {
        "source": SOURCE_BASIN, "target": TARGET_BASIN,
        "k": K, "n_pos_huasco": int(n_pos_s), "n_neg_huasco": int(n_neg_s),
        "n_pos_bueno": int(n_pos_t), "n_neg_bueno": int(n_neg_t),
        "results": results,
    }
    out_json = RESULTS / "h33b_transferability.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"[h33b] wrote {out_json.name}")

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    labels = ["A: features only", "B: features + cluster", "C: cluster only"]
    keys = ["A_features_only", "B_features_plus_cluster", "C_cluster_only"]
    self_roc = [results[k]["train_self"]["roc"] for k in keys]
    trans_roc = [results[k]["test_transfer"]["roc"] for k in keys]
    trans_lo = [results[k]["test_transfer"]["roc_ci"][0] for k in keys]
    trans_hi = [results[k]["test_transfer"]["roc_ci"][1] for k in keys]
    x = np.arange(len(labels))
    ax.bar(x - 0.2, self_roc, width=0.4, color="#7570b3",
           label=f"Train self (Huasco, n={n_pos_s + n_neg_s})")
    ax.bar(x + 0.2, trans_roc, width=0.4, color="#1b9e77",
           label=f"Transfer (Bueno, n={n_pos_t + n_neg_t})")
    err_lo = np.array(trans_roc) - np.array(trans_lo)
    err_hi = np.array(trans_hi) - np.array(trans_roc)
    ax.errorbar(x + 0.2, trans_roc, yerr=[err_lo, err_hi], fmt="none",
                ecolor="black", capsize=4)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("AUC ROC"); ax.set_ylim(0.4, 1.0)
    ax.axhline(0.5, color="gray", lw=1, ls=":")
    ax.legend(); ax.set_title("H3.3b — Train Huasco → Test Bueno (zero-shot)")
    out_png = RESULTS / "h33b_transferability.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"[h33b] wrote {out_png.name}")


if __name__ == "__main__":
    main()
