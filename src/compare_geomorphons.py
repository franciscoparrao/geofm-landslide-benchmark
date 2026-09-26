"""Compare K-means cluster maps to SAGA geomorphons (rule-based expert baseline).

For each K in cluster maps, compute NMI / AMI / ARI on common-valid pixels and
write a confusion matrix (rows=geomorphons class, cols=k-means cluster). Saves
JSON + PNG (heatmap of normalized confusion).

H3.2 expectation: NMI moderately high (≥0.3) but not perfect — clustering
preserves expert knowledge but adds refinement.
"""
from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)

from config import BASINS, DEFAULT_BASIN, RESULTS, basin_dir

K_VALUES = (5, 10, 15)
GEOMORPHON_NAMES = {
    1: "flat", 2: "peak", 3: "ridge", 4: "shoulder", 5: "spur",
    6: "slope", 7: "hollow", 8: "footslope", 9: "valley", 10: "pit",
}


def load_geomorphons(basin):
    path = basin_dir(basin) / "terrain" / "geomorphons.tif"
    with rasterio.open(path) as src:
        a = src.read(1, masked=True)
    arr = np.ma.getdata(a).astype(np.float32)
    arr[np.ma.getmaskarray(a)] = np.nan
    return arr


def load_cluster_map(basin, k):
    path = RESULTS / f"{basin}_kmeans_K{k:02d}.tif"
    with rasterio.open(path) as src:
        return src.read(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--basin", default=DEFAULT_BASIN, choices=BASINS)
    args = parser.parse_args()
    basin = args.basin

    geo = load_geomorphons(basin).reshape(-1)
    geo_valid = np.isfinite(geo) & (geo > 0)
    print(f"[geomorphons] basin={basin} valid_pixels={int(geo_valid.sum()):,}")
    geo_int = np.zeros_like(geo, dtype=np.int16)
    geo_int[geo_valid] = geo[geo_valid].astype(np.int16)

    runs = []
    for k in K_VALUES:
        clusters = load_cluster_map(basin, k).reshape(-1)
        cl_valid = clusters >= 0
        joint = geo_valid & cl_valid
        n_joint = int(joint.sum())
        g = geo_int[joint]
        c = clusters[joint].astype(np.int16)
        nmi = float(normalized_mutual_info_score(g, c))
        ami = float(adjusted_mutual_info_score(g, c))
        ari = float(adjusted_rand_score(g, c))
        print(
            f"[geomorphons] K={k:2d} joint_pixels={n_joint:,} "
            f"NMI={nmi:.3f} AMI={ami:.3f} ARI={ari:.3f}"
        )

        geo_classes = np.arange(1, 11)
        cm = np.zeros((10, k), dtype=np.int64)
        for gi, gc in enumerate(geo_classes):
            mask = g == gc
            if mask.any():
                bc = np.bincount(c[mask], minlength=k)
                cm[gi] = bc

        cm_norm = cm / (cm.sum(axis=1, keepdims=True) + 1e-9)

        fig, ax = plt.subplots(figsize=(0.6 * k + 4, 6), constrained_layout=True)
        im = ax.imshow(cm_norm, cmap="viridis", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(k))
        ax.set_yticks(range(10))
        ax.set_xticklabels([f"c{i}" for i in range(k)])
        ax.set_yticklabels([f"{i}-{GEOMORPHON_NAMES[i]}" for i in geo_classes])
        ax.set_xlabel(f"K-means cluster (K={k})")
        ax.set_ylabel("Geomorphon class (SAGA)")
        ax.set_title(
            f"{basin} — geomorphons → cluster (row-normalized)\n"
            f"NMI={nmi:.3f} AMI={ami:.3f} ARI={ari:.3f}"
        )
        for gi in range(10):
            for ci in range(k):
                v = cm_norm[gi, ci]
                if v > 0.05:
                    ax.text(ci, gi, f"{v:.2f}", ha="center", va="center",
                            color="white" if v < 0.5 else "black", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.04, label="row share")
        out_png = RESULTS / f"{basin}_geomorphons_vs_K{k:02d}.png"
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[geomorphons]   wrote {out_png.name}")

        runs.append({
            "k": k, "joint_pixels": n_joint,
            "nmi": nmi, "ami": ami, "ari": ari,
            "confusion_matrix_counts": cm.tolist(),
            "geomorphon_classes": geo_classes.tolist(),
            "geomorphon_names": [GEOMORPHON_NAMES[g] for g in geo_classes],
        })

    out = RESULTS / f"{basin}_geomorphons_vs_kmeans.json"
    out.write_text(json.dumps({"basin": basin, "runs": runs}, indent=2))
    print(f"[geomorphons] wrote {out.name}")


if __name__ == "__main__":
    main()
