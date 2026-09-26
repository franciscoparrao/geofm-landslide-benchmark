"""H3.3 — utilidad downstream: ¿agrega información el cluster K-means a un
modelo de susceptibilidad?

Compara dos modelos RandomForest sobre un dataset de eventos del catastro
SERNAGEOMIN para una cuenca:
  A: 17 features raster originales
  B: 17 features + cluster_K05 one-hot (5 dummies)

Mismos splits (StratifiedKFold), AUC ROC y AUC PR por fold + paired bootstrap
de Δ-AUC.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pyproj
import rasterio
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from config import (
    BASINS, DEFAULT_BASIN, INVENTORY_BASE, RESULTS, basin_dir, feature_paths,
    feature_names,
)

NEG_RATIO = 5
BUFFER_PX = 5
N_FOLDS = 5
N_BOOTSTRAP = 1000
N_TREES = 300
SEED = 42
K_FOR_FEATURE = 5

MINIMAL_FEATURES = ("slope", "curvature", "log1p_flow_accumulation_mfd")


def load_events_xy(basin, raster_crs, raster_transform):
    csv_path = INVENTORY_BASE / f"{basin}.csv"
    if not csv_path.exists():
        raise SystemExit(f"inventory not found: {csv_path}")
    rows = []
    with open(csv_path) as f:
        header = f.readline().strip().split(",")
        lat_idx = header.index("lat")
        lon_idx = header.index("lon")
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--basin", default=DEFAULT_BASIN, choices=BASINS)
    parser.add_argument("--features", choices=("full", "minimal"), default="full")
    args = parser.parse_args()
    basin = args.basin
    features_mode = args.features

    rng = np.random.default_rng(SEED)
    stack_path = RESULTS / f"{basin}_stack.npz"
    print(f"[h33] basin={basin} loading stack")
    with np.load(stack_path, allow_pickle=False) as npz:
        X_all = npz["X"].astype(np.float32)
        valid_idx = npz["valid_idx"]
        feat_names = list(npz["feature_names"])
        height = int(npz["height"])
        width = int(npz["width"])

    cluster_path = RESULTS / f"{basin}_kmeans_K{K_FOR_FEATURE:02d}.tif"
    with rasterio.open(cluster_path) as src:
        clusters_full = src.read(1).reshape(-1)
        raster_crs = src.crs
        raster_transform = src.transform

    pos_rows, pos_cols = load_events_xy(basin, raster_crs, raster_transform)
    in_bounds = (
        (pos_rows >= 0) & (pos_rows < height)
        & (pos_cols >= 0) & (pos_cols < width)
    )
    pos_rows = pos_rows[in_bounds]
    pos_cols = pos_cols[in_bounds]
    pos_flat = pos_rows * width + pos_cols
    pos_clusters = clusters_full[pos_flat]
    valid_mask = np.zeros(height * width, dtype=bool)
    valid_mask[valid_idx] = True
    pos_valid = valid_mask[pos_flat] & (pos_clusters >= 0)
    pos_flat = pos_flat[pos_valid]
    n_pos = pos_flat.size
    print(f"[h33] events_in_bounds={int(in_bounds.sum())} valid_for_model={n_pos}")
    if n_pos < 30:
        raise SystemExit("too few positive events for modeling")

    excluded = np.zeros(height * width, dtype=bool)
    rows_grid, cols_grid = pos_flat // width, pos_flat % width
    for dr in range(-BUFFER_PX, BUFFER_PX + 1):
        for dc in range(-BUFFER_PX, BUFFER_PX + 1):
            r2 = rows_grid + dr; c2 = cols_grid + dc
            ok = (r2 >= 0) & (r2 < height) & (c2 >= 0) & (c2 < width)
            excluded[(r2[ok] * width + c2[ok])] = True
    candidate = valid_mask & (clusters_full >= 0) & (~excluded)
    candidate_idx = np.flatnonzero(candidate)
    n_neg = NEG_RATIO * n_pos
    neg_flat = rng.choice(candidate_idx, size=n_neg, replace=False)
    print(f"[h33] negatives sampled={n_neg} from {candidate_idx.size:,} candidates")

    flat_to_row = {int(idx): i for i, idx in enumerate(valid_idx)}
    pos_X_idx = np.array([flat_to_row[int(p)] for p in pos_flat])
    neg_X_idx = np.array([flat_to_row[int(p)] for p in neg_flat])
    X_full_pos = X_all[pos_X_idx]
    X_full_neg = X_all[neg_X_idx]
    cl_pos = clusters_full[pos_flat]
    cl_neg = clusters_full[neg_flat]
    X_full = np.vstack([X_full_pos, X_full_neg])
    cl = np.concatenate([cl_pos, cl_neg])
    y = np.concatenate([np.ones(n_pos), np.zeros(n_neg)]).astype(np.int8)

    if features_mode == "minimal":
        keep = [feat_names.index(n) for n in MINIMAL_FEATURES]
        X = X_full[:, keep]
        feat_names_used = list(MINIMAL_FEATURES)
    else:
        X = X_full
        feat_names_used = feat_names
    print(
        f"[h33] features_mode={features_mode} X.shape={X.shape} "
        f"positives={n_pos} negatives={n_neg}"
    )

    cl_oh = np.zeros((cl.size, K_FOR_FEATURE), dtype=np.float32)
    cl_oh[np.arange(cl.size), cl.astype(int)] = 1.0
    XB = np.hstack([X, cl_oh])
    feat_A = feat_names_used
    feat_B = feat_names_used + [f"cluster_c{i}" for i in range(K_FOR_FEATURE)]

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_results = []
    test_scores_A_all, test_scores_B_all, test_y_all = [], [], []
    importances_A = np.zeros(len(feat_A))
    importances_B = np.zeros(len(feat_B))

    for fi, (tr, te) in enumerate(skf.split(X, y)):
        t0 = perf_counter()
        rfA = RandomForestClassifier(
            n_estimators=N_TREES, max_depth=None, min_samples_leaf=5,
            n_jobs=-1, random_state=SEED + fi, class_weight="balanced",
        ).fit(X[tr], y[tr])
        sA = rfA.predict_proba(X[te])[:, 1]
        rfB = RandomForestClassifier(
            n_estimators=N_TREES, max_depth=None, min_samples_leaf=5,
            n_jobs=-1, random_state=SEED + fi, class_weight="balanced",
        ).fit(XB[tr], y[tr])
        sB = rfB.predict_proba(XB[te])[:, 1]

        roc_A = roc_auc_score(y[te], sA); roc_B = roc_auc_score(y[te], sB)
        pr_A = average_precision_score(y[te], sA); pr_B = average_precision_score(y[te], sB)
        importances_A += rfA.feature_importances_
        importances_B += rfB.feature_importances_
        elapsed = perf_counter() - t0
        print(
            f"[h33] fold={fi} ROC A={roc_A:.3f} B={roc_B:.3f} ΔROC={roc_B - roc_A:+.3f} "
            f"PR A={pr_A:.3f} B={pr_B:.3f} ΔPR={pr_B - pr_A:+.3f} time={elapsed:.1f}s"
        )
        fold_results.append({
            "fold": fi, "roc_A": float(roc_A), "roc_B": float(roc_B),
            "pr_A": float(pr_A), "pr_B": float(pr_B),
            "delta_roc": float(roc_B - roc_A), "delta_pr": float(pr_B - pr_A),
        })
        test_scores_A_all.append((y[te], sA))
        test_scores_B_all.append((y[te], sB))
        test_y_all.append(y[te])

    importances_A /= N_FOLDS
    importances_B /= N_FOLDS

    delta_rocs = np.array([r["delta_roc"] for r in fold_results])
    delta_prs = np.array([r["delta_pr"] for r in fold_results])
    print(
        f"\n[h33] mean ΔROC={delta_rocs.mean():+.3f} std={delta_rocs.std():.3f}"
        f"  mean ΔPR={delta_prs.mean():+.3f} std={delta_prs.std():.3f}"
    )

    y_concat = np.concatenate([yt for yt, _ in test_scores_A_all])
    sA_concat = np.concatenate([s for _, s in test_scores_A_all])
    sB_concat = np.concatenate([s for _, s in test_scores_B_all])
    n = y_concat.size
    boot_deltas_roc = np.empty(N_BOOTSTRAP)
    boot_deltas_pr = np.empty(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        ix = rng.integers(0, n, size=n)
        if y_concat[ix].sum() == 0 or y_concat[ix].sum() == n:
            boot_deltas_roc[b] = np.nan; boot_deltas_pr[b] = np.nan; continue
        boot_deltas_roc[b] = (
            roc_auc_score(y_concat[ix], sB_concat[ix])
            - roc_auc_score(y_concat[ix], sA_concat[ix])
        )
        boot_deltas_pr[b] = (
            average_precision_score(y_concat[ix], sB_concat[ix])
            - average_precision_score(y_concat[ix], sA_concat[ix])
        )
    boot_deltas_roc = boot_deltas_roc[~np.isnan(boot_deltas_roc)]
    boot_deltas_pr = boot_deltas_pr[~np.isnan(boot_deltas_pr)]
    ci_roc = (np.percentile(boot_deltas_roc, 2.5), np.percentile(boot_deltas_roc, 97.5))
    ci_pr = (np.percentile(boot_deltas_pr, 2.5), np.percentile(boot_deltas_pr, 97.5))
    print(
        f"[h33] bootstrap N={N_BOOTSTRAP} ΔROC mean={boot_deltas_roc.mean():+.3f} "
        f"95% CI=({ci_roc[0]:+.3f}, {ci_roc[1]:+.3f})"
    )
    print(
        f"[h33] bootstrap         ΔPR  mean={boot_deltas_pr.mean():+.3f} "
        f"95% CI=({ci_pr[0]:+.3f}, {ci_pr[1]:+.3f})"
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    order = np.argsort(importances_B)[::-1]
    ax = axes[0]
    ax.barh(np.arange(len(feat_B)), importances_B[order],
            color=["#d95f02" if "cluster" in feat_B[order[i]] else "#7570b3"
                   for i in range(len(feat_B))])
    ax.set_yticks(np.arange(len(feat_B)))
    ax.set_yticklabels([feat_B[i] for i in order], fontsize=8)
    ax.invert_yaxis(); ax.set_xlabel("RF feature importance (model B mean)")
    ax.set_title("Model B importances (cluster dummies highlighted)")

    ax = axes[1]
    ax.hist(boot_deltas_roc, bins=40, color="#1b9e77", alpha=0.7,
            label=f"ΔROC mean={boot_deltas_roc.mean():+.3f}")
    ax.axvline(0, color="black", lw=1, ls="--")
    ax.axvline(ci_roc[0], color="black", lw=1, ls=":")
    ax.axvline(ci_roc[1], color="black", lw=1, ls=":")
    ax.set_xlabel("Δ AUC ROC (B − A)"); ax.set_ylabel("bootstrap count")
    ax.set_title(f"95% CI [{ci_roc[0]:+.3f}, {ci_roc[1]:+.3f}]")
    ax.legend()
    out_png = RESULTS / f"{basin}_h33_susceptibility_{features_mode}.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"[h33] wrote {out_png.name}")

    summary = {
        "basin": basin, "features_mode": features_mode,
        "features_used": feat_names_used, "k_for_feature": K_FOR_FEATURE,
        "n_positives": int(n_pos), "n_negatives": int(n_neg),
        "neg_ratio": NEG_RATIO, "buffer_px": BUFFER_PX, "n_folds": N_FOLDS,
        "fold_results": fold_results,
        "delta_roc_mean": float(delta_rocs.mean()), "delta_roc_std": float(delta_rocs.std()),
        "delta_pr_mean": float(delta_prs.mean()), "delta_pr_std": float(delta_prs.std()),
        "bootstrap_n": N_BOOTSTRAP,
        "delta_roc_ci_95": [float(ci_roc[0]), float(ci_roc[1])],
        "delta_pr_ci_95": [float(ci_pr[0]), float(ci_pr[1])],
        "feature_importances_A": dict(zip(feat_A, [float(v) for v in importances_A])),
        "feature_importances_B": dict(zip(feat_B, [float(v) for v in importances_B])),
    }
    out_json = RESULTS / f"{basin}_h33_susceptibility_{features_mode}.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"[h33] wrote {out_json.name}")


if __name__ == "__main__":
    main()
