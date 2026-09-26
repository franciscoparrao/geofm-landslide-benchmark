"""Per-cluster feature signatures for K-means baseline runs.

Computes mean and std for each feature per cluster (in original units) and
writes a JSON summary so clusters can be interpreted without rerunning the
clustering.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import rasterio

from config import BASINS, DEFAULT_BASIN, RESULTS

K_VALUES = (5, 10, 15)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--basin", default=DEFAULT_BASIN, choices=BASINS)
    args = parser.parse_args()
    basin = args.basin

    stack_path = RESULTS / f"{basin}_stack.npz"
    with np.load(stack_path, allow_pickle=False) as npz:
        X = npz["X"].astype(np.float32)
        valid_idx = npz["valid_idx"]
        feature_names = list(npz["feature_names"])
        height = int(npz["height"])
        width = int(npz["width"])

    n_total = height * width
    signatures = {"basin": basin, "feature_names": feature_names, "runs": []}

    for k in K_VALUES:
        tif = RESULTS / f"{basin}_kmeans_K{k:02d}.tif"
        with rasterio.open(tif) as src:
            labels_full = src.read(1).reshape(-1)
        labels = labels_full[valid_idx]
        run = {"k": k, "clusters": []}
        for c in range(k):
            sel = labels == c
            n = int(sel.sum())
            if n == 0:
                run["clusters"].append({"id": c, "size": 0})
                continue
            xc = X[sel]
            run["clusters"].append(
                {
                    "id": c,
                    "size": n,
                    "share": n / X.shape[0],
                    "mean": [float(v) for v in xc.mean(axis=0)],
                    "std": [float(v) for v in xc.std(axis=0)],
                }
            )
        signatures["runs"].append(run)

    out = RESULTS / f"{basin}_cluster_signatures.json"
    out.write_text(json.dumps(signatures, indent=2))
    print(f"[analyze] wrote {out}")

    print("\n[analyze] standardized signatures (feature z-score by cluster)\n")
    feat_mean = X.mean(axis=0)
    feat_std = X.std(axis=0) + 1e-9
    for run in signatures["runs"]:
        k = run["k"]
        print(f"=== K={k} ===")
        header = "cluster  size%  " + "  ".join(f"{n[:6]:>6}" for n in feature_names)
        print(header)
        for c in run["clusters"]:
            if c["size"] == 0:
                continue
            z = (np.array(c["mean"]) - feat_mean) / feat_std
            row = (
                f"  c{c['id']:02d}   {100 * c['share']:5.1f}  "
                + "  ".join(f"{v:+6.2f}" for v in z)
            )
            print(row)
        print()


if __name__ == "__main__":
    main()
