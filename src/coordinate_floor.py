"""How much of the reported discrimination is position rather than terrain?

A blind review of this study asked whether the 10 km spatial blocks actually
break the spatial structure of the inventory. They do not: a Random Forest given
nothing but the row and column of each point -- no terrain, no spectra, no
embedding -- discriminates at AUC 0.83-0.93 under the published folds, and in
Maipo it outscores the entire seventeen-layer stack. That is a floor every other
number in the paper has to be read against, because a model scoring 0.94 in a
basin where coordinates alone score 0.93 has demonstrated very little about
terrain.

It also biases the A-versus-B comparison. A 6.72 km patch is close to a location
fingerprint, while pipeline A sees one pixel, so positional information is
available asymmetrically to the side of the foundation models.

This script quantifies both: the coordinate-only floor and every pipeline's
score as a function of block size, from the published 10 km up to 40 km. Blocks
are assigned with StratifiedGroupKFold throughout, because at sizes above 10 km
plain GroupKFold leaves a positive-free test fold in Huasco, whose events
concentrate in few blocks; holding the splitter fixed across sizes is what makes
the sizes comparable to each other.
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np
import rasterio
from scipy.stats import t as student_t
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from config import RESULTS, basin_dir
from point_probes import build_dataset
from terramind_linprobe import (
    N_FOLDS, N_TREES, PRITHVI_NAME, SEED, TERRAMIND_NAME, embedding_cache_path,
    embedding_fingerprint,
)

BASINS = [("06_rio_huasco", "Huasco"), ("09_rio_maipo", "Maipo"),
          ("11_rio_maule", "Maule")]
SIZES_PX = [333, 667, 1000, 1333]          # 30 m pixels -> 10, 20, 30, 40 km


def tci(v):
    n = len(v)
    m = sum(v) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1))
    h = float(student_t.ppf(0.975, n - 1)) * sd / math.sqrt(n)
    return m, m - h, m + h, not (m - h <= 0 <= m + h)


def fit_auc(X, y, folds):
    out = []
    for k, (tr, te) in enumerate(folds):
        clf = RandomForestClassifier(
            n_estimators=N_TREES, min_samples_leaf=5, n_jobs=-1,
            random_state=SEED + k, class_weight="balanced",
        ).fit(X[tr], y[tr])
        out.append(float(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1])))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=N_FOLDS)
    ap.add_argument("--negatives", choices=("uniform", "constrained"),
                    default="uniform",
                    help="Negative pool. The floor has to be computed under the\n                          same sampling as the results it governs; review of this\n                          work pointed out that reporting it only for the uniform\n                          design leaves the constrained arm, where the headline\n                          significance lives, without one.")
    a = ap.parse_args()
    report = {}

    for slug, name in BASINS:
        rows, cols, y, X17, _ = build_dataset(slug, negatives=a.negatives)
        width = rasterio.open(basin_dir(slug) / "dem_30m.tif").width
        XY = np.c_[rows, cols].astype(np.float32)

        emb = {}
        for lbl, enc, mods in [("Prithvi", PRITHVI_NAME, ["S2L2A"]),
                               ("TM+DEM", TERRAMIND_NAME, ["DEM"])]:
            s = "terramind" if enc == TERRAMIND_NAME else enc
            # The embeddings are a convenience here, not the point: the floor is
            # a statement about the manual baseline. Under a negative pool with
            # no cached encoding we report A against coordinates and skip the
            # foundation-model columns rather than refusing to run.
            cache = embedding_cache_path(slug, s, "pretrained", mods)
            if a.negatives != "uniform":
                cache = cache.with_name(
                    cache.name.replace(".npz", f"_{a.negatives}.npz"))
            if not cache.exists():
                print(f"  [skip] {lbl}: no cached embedding for the "
                      f"{a.negatives} point set")
                continue
            d = np.load(cache, allow_pickle=False)
            want = embedding_fingerprint(enc, "pretrained", mods, rows, cols, y)
            if str(d["fingerprint"]) != want:
                print(f"  [skip] {lbl}: cached embedding describes another "
                      f"point set")
                continue
            emb[lbl] = d["embeddings"].astype(np.float32)

        report[slug] = {"basin": name, "sizes": {}}
        print(f"\n=== {name} ===")
        print(f"{'block':>7s} {'nblk':>5s} {'coords':>8s} {'A':>8s} "
              f"{'Prithvi':>9s} {'TM+DEM':>8s} {'Pr-A':>17s} {'TM-A':>17s}")
        for S in SIZES_PX:
            ncb = (width + S - 1) // S
            blk = (rows // S) * ncb + (cols // S)
            nb = len(np.unique(blk))
            if nb < a.folds:
                print(f"{S*30/1000:5.0f}km  too few blocks ({nb})")
                continue
            folds = list(StratifiedGroupKFold(n_splits=a.folds, shuffle=False)
                         .split(X17, y, groups=blk))
            if min(int(y[te].sum()) for _, te in folds) == 0:
                print(f"{S*30/1000:5.0f}km  a test fold has no positives")
                continue

            r_xy = fit_auc(XY, y, folds)
            r_a = fit_auc(X17, y, folds)
            nan5 = [float("nan")] * len(folds)
            r_p = fit_auc(emb["Prithvi"], y, folds) if "Prithvi" in emb else nan5
            r_t = fit_auc(emb["TM+DEM"], y, folds) if "TM+DEM" in emb else nan5
            d_p = tci([x - z for x, z in zip(r_p, r_a)]) if "Prithvi" in emb \
                else (float("nan"),) * 3 + (False,)
            d_t = tci([x - z for x, z in zip(r_t, r_a)]) if "TM+DEM" in emb \
                else (float("nan"),) * 3 + (False,)
            print(f"{S*30/1000:5.0f}km {nb:5d} {np.mean(r_xy):8.3f} "
                  f"{np.mean(r_a):8.3f} {np.mean(r_p):9.3f} {np.mean(r_t):8.3f} "
                  f"{d_p[0]:+7.3f}[{d_p[1]:+.3f},{d_p[2]:+.3f}]"
                  f"{'*' if d_p[3] else ' '}"
                  f"{d_t[0]:+7.3f}[{d_t[1]:+.3f},{d_t[2]:+.3f}]"
                  f"{'*' if d_t[3] else ' '}", flush=True)
            report[slug]["sizes"][f"{S*30/1000:.0f}km"] = {
                "block_px": S, "n_blocks": nb,
                "coords_roc": r_xy, "A_roc": r_a,
                "Prithvi_roc": r_p, "TM+DEM_roc": r_t,
                "delta_Prithvi_vs_A": {"mean": d_p[0], "ci": [d_p[1], d_p[2]],
                                       "significant": d_p[3]},
                "delta_TM+DEM_vs_A": {"mean": d_t[0], "ci": [d_t[1], d_t[2]],
                                      "significant": d_t[3]},
            }

    suffix = "" if a.negatives == "uniform" else f"_{a.negatives}"
    dst = RESULTS / f"coordinate_floor{suffix}.json"
    dst.write_text(json.dumps(report, indent=2))
    print(f"\n[done] wrote {dst.name}")


if __name__ == "__main__":
    main()
