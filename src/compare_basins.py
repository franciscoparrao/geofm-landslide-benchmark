"""All-pairs cross-basin cluster correspondence using centroid matching.

For a given K, computes per-cluster z-score signatures within each basin and
matches clusters between every pair of basins via Hungarian assignment on
cosine similarity. Outputs:
  - all-pairs JSON with similarity matrices and Hungarian matches
  - per-pair heatmap PNGs
  - aggregate summary (mean cosine, fraction of pairs with cos>=0.8 by morphotype)
"""
from __future__ import annotations

import argparse
import itertools
import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

from config import BASINS, RESULTS

DEFAULT_BASINS = list(BASINS)
SHORT_NAMES = {
    "01_rio_lluta": "Llut",
    "06_rio_huasco": "Huas",
    "09_rio_maipo": "Maip",
    "11_rio_maule": "Maul",
    "13_rio_bueno": "Buen",
}


def per_basin_zscore_signatures(signatures_path):
    data = json.loads(signatures_path.read_text())
    feature_names = data["feature_names"]
    runs = {run["k"]: run for run in data["runs"]}
    return feature_names, runs


def to_z_matrix(run, feature_names):
    means = np.array(
        [c["mean"] if c["size"] > 0 else [np.nan] * len(feature_names)
         for c in run["clusters"]],
        dtype=np.float64,
    )
    shares = np.array(
        [c.get("share", 0.0) for c in run["clusters"]], dtype=np.float64
    )
    valid = shares > 0
    feat_mu = np.nanmean(means[valid], axis=0)
    feat_sd = np.nanstd(means[valid], axis=0) + 1e-9
    z = (means - feat_mu) / feat_sd
    return z, shares


def cosine_sim(A, B):
    a = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    b = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    return a @ b.T


def heatmap(sim, k, ba, bb, out_png, row_ind, col_ind):
    fig, ax = plt.subplots(figsize=(5.5, 5), constrained_layout=True)
    im = ax.imshow(sim, cmap="RdBu_r", vmin=-1, vmax=1)
    sa, sb = SHORT_NAMES.get(ba, ba[:4]), SHORT_NAMES.get(bb, bb[:4])
    ax.set_xticks(range(k)); ax.set_yticks(range(k))
    ax.set_xticklabels([f"{sb}-c{j}" for j in range(k)])
    ax.set_yticklabels([f"{sa}-c{i}" for i in range(k)])
    ax.set_xlabel(bb); ax.set_ylabel(ba)
    ax.set_title(f"Cosine sim K={k}  {sa}↔{sb}")
    for i, j in zip(row_ind, col_ind):
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor="black", lw=2))
    for i in range(k):
        for j in range(k):
            ax.text(j, i, f"{sim[i, j]:+.2f}", ha="center", va="center",
                    color="white" if abs(sim[i, j]) > 0.5 else "black",
                    fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.04, label="cos")
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--basins", nargs="+", default=DEFAULT_BASINS)
    args = parser.parse_args()
    k = args.k
    basins = args.basins

    sigs = {}
    feat_ref = None
    for b in basins:
        path = RESULTS / f"{b}_cluster_signatures.json"
        if not path.exists():
            print(f"[compare] WARN signatures missing for {b} — skip")
            continue
        feat, runs = per_basin_zscore_signatures(path)
        if feat_ref is None:
            feat_ref = feat
        elif feat != feat_ref:
            raise SystemExit(f"feature lists differ between basins ({b})")
        if k not in runs:
            print(f"[compare] WARN k={k} missing for {b} — skip")
            continue
        sigs[b] = to_z_matrix(runs[k], feat_ref)

    available = list(sigs.keys())
    print(f"[compare] K={k} basins available: {available}")
    pairs = list(itertools.combinations(available, 2))
    print(f"[compare] all-pairs: {len(pairs)}")

    pair_records = []
    cosines_matched = []
    diag_strengths = {b: [] for b in available}

    for ba, bb in pairs:
        Za, sha = sigs[ba]
        Zb, shb = sigs[bb]
        sim = cosine_sim(Za, Zb)
        row_ind, col_ind = linear_sum_assignment(-sim)
        matches = []
        cos_per = []
        for i, j in zip(row_ind, col_ind):
            c = float(sim[i, j])
            cos_per.append(c)
            matches.append({
                "a_cluster": int(i), "b_cluster": int(j), "cosine": c,
                "share_a_pct": float(sha[i] * 100),
                "share_b_pct": float(shb[j] * 100),
            })
        mean_cos = float(np.mean(cos_per))
        median_cos = float(np.median(cos_per))
        n_high = int(sum(1 for c in cos_per if c >= 0.8))
        cosines_matched.extend(cos_per)
        diag_strengths[ba].extend(cos_per)
        diag_strengths[bb].extend(cos_per)

        out_png = RESULTS / f"compare_{SHORT_NAMES.get(ba, ba)}_{SHORT_NAMES.get(bb, bb)}_K{k:02d}.png"
        heatmap(sim, k, ba, bb, out_png, row_ind, col_ind)

        rec = {
            "basin_a": ba, "basin_b": bb, "k": k,
            "mean_cosine": mean_cos, "median_cosine": median_cos,
            "n_pairs_cos_ge_0.8": n_high,
            "matches": matches,
            "similarity_matrix": sim.tolist(),
        }
        pair_records.append(rec)
        print(
            f"  {SHORT_NAMES.get(ba, ba)}↔{SHORT_NAMES.get(bb, bb):>4}  "
            f"mean_cos={mean_cos:+.3f} median={median_cos:+.3f}  "
            f"high_cos(≥0.8)={n_high}/{k}"
        )

    cosines_matched = np.array(cosines_matched)
    summary = {
        "k": k, "basins": available, "n_pairs": len(pairs),
        "global_mean_matched_cosine": float(cosines_matched.mean()),
        "global_median_matched_cosine": float(np.median(cosines_matched)),
        "fraction_matches_cos_ge_0.8": float((cosines_matched >= 0.8).mean()),
        "fraction_matches_cos_ge_0.5": float((cosines_matched >= 0.5).mean()),
        "pairs": pair_records,
    }

    print(
        f"\n[compare] K={k} GLOBAL  mean={summary['global_mean_matched_cosine']:+.3f}  "
        f"median={summary['global_median_matched_cosine']:+.3f}  "
        f"frac(cos≥0.8)={summary['fraction_matches_cos_ge_0.8']:.2f}"
    )

    fig, ax = plt.subplots(figsize=(7.5, 5), constrained_layout=True)
    matrix = np.full((len(available), len(available)), np.nan)
    for r in pair_records:
        i = available.index(r["basin_a"])
        j = available.index(r["basin_b"])
        matrix[i, j] = r["mean_cosine"]
        matrix[j, i] = r["mean_cosine"]
    np.fill_diagonal(matrix, 1.0)
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    ticks = [SHORT_NAMES.get(b, b[:4]) for b in available]
    ax.set_xticks(range(len(available))); ax.set_yticks(range(len(available)))
    ax.set_xticklabels(ticks); ax.set_yticklabels(ticks)
    ax.set_title(f"All-pairs mean Hungarian-matched cosine (K={k})")
    for i in range(len(available)):
        for j in range(len(available)):
            v = matrix[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                        color="white" if abs(v) > 0.5 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.04, label="mean cos")
    out_png = RESULTS / f"compare_all_pairs_K{k:02d}.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"[compare] wrote {out_png.name}")

    out_json = RESULTS / f"compare_all_pairs_K{k:02d}.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"[compare] wrote {out_json.name}")


if __name__ == "__main__":
    main()
