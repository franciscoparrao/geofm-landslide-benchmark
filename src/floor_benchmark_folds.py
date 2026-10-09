"""Positional floor on the benchmark's own folds, for every pipeline, plus an
elevation arm for the manual baseline.

Two objections of blind review b1 (2026-10-07) motivate this script.

1. The floor of coordinate_floor.py uses StratifiedGroupKFold, so its margins
   paired a differently-folded baseline with the coordinates. Here the floor is
   computed on exactly the benchmark folds (GroupKFold, 10 km blocks), and the
   rule of Algorithm 1 is applied to every pipeline, not only to the baseline.
2. The baseline excludes absolute elevation as a positional proxy, but
   TerraMind receives the absolute DEM (z-scored), which carries the mean patch
   elevation. "A+elev" adds the elevation at the pixel to the seventeen layers,
   so the asymmetry can be measured rather than argued.

Every interval is reported twice: the fold-level paired t-interval of the main
text, and the Nadeau-Bengio corrected resampled interval, whose variance factor
is 1/k + n_test/n_train (Nadeau and Bengio 2003).

Output: results/floor_benchmark_folds.json
"""
from __future__ import annotations

import json
import math

import numpy as np
import rasterio
from scipy.stats import t as student_t
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from config import RESULTS, basin_dir
from point_probes import build_dataset
from terramind_linprobe import (
    N_FOLDS, N_TREES, PRITHVI_NAME, SEED, TERRAMIND_NAME,
    embedding_cache_path, embedding_fingerprint, load_cached_embeddings,
)

BASINS = [("06_rio_huasco", "Huasco"), ("09_rio_maipo", "Maipo"),
          ("11_rio_maule", "Maule")]
PIPELINES = [("TM+DEM", TERRAMIND_NAME, "terramind", ["DEM"]),
             ("TM+DEM+S2", TERRAMIND_NAME, "terramind", ["DEM", "S2L2A"]),
             ("Prithvi", PRITHVI_NAME, "prithvi-300m", ["S2L2A"])]


def intervals(d, k=N_FOLDS):
    """Fold-level paired t-interval and its Nadeau-Bengio correction."""
    n = len(d)
    m = sum(d) / n
    s2 = sum((x - m) ** 2 for x in d) / (n - 1)
    tq = float(student_t.ppf(0.975, n - 1))
    h = tq * math.sqrt(s2 / n)
    h_nb = tq * math.sqrt((1 / k + 1 / (k - 1)) * s2)   # n_test/n_train = 1/(k-1)
    return {"mean": m, "ci": [m - h, m + h], "ci_nb": [m - h_nb, m + h_nb],
            "sig": not (m - h <= 0 <= m + h), "sig_nb": not (m - h_nb <= 0 <= m + h_nb)}


def fit_auc(X, y, folds):
    out = []
    for k, (tr, te) in enumerate(folds):
        clf = RandomForestClassifier(n_estimators=N_TREES, min_samples_leaf=5, n_jobs=-1,
                                     random_state=SEED + k, class_weight="balanced")
        clf.fit(X[tr], y[tr])
        out.append(float(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1])))
    return out


def main():
    report = {}
    for slug, name in BASINS:
        rows, cols, y, X17, blk = build_dataset(slug, negatives="uniform")
        folds = list(GroupKFold(n_splits=N_FOLDS).split(X17, y, groups=blk))
        with rasterio.open(basin_dir(slug) / "dem_30m.tif") as src:
            elev = src.read(1)[rows, cols].astype(np.float32)
        XY = np.c_[rows, cols].astype(np.float32)
        roc = {"coords": fit_auc(XY, y, folds), "A": fit_auc(X17, y, folds),
               "A+elev": fit_auc(np.c_[X17, elev], y, folds)}
        for lbl, enc, s, mods in PIPELINES:
            emb = load_cached_embeddings(embedding_cache_path(slug, s, "pretrained", mods),
                                         embedding_fingerprint(enc, "pretrained", mods, rows, cols, y))
            if emb is None:
                raise SystemExit(f"{slug} {lbl}: no cached embedding for the benchmark points")
            roc[lbl] = fit_auc(emb, y, folds)
        diff = lambda a, b: [x - z for x, z in zip(roc[a], roc[b])]
        r = {"roc_mean": {k: float(np.mean(v)) for k, v in roc.items()},
             "roc_folds": roc,
             "vs_coords": {k: intervals(diff(k, "coords")) for k in roc if k != "coords"},
             "vs_A": {k: intervals(diff(k, "A")) for k in roc if k not in ("coords", "A")},
             "vs_A+elev": {lbl: intervals(diff(lbl, "A+elev")) for lbl, *_ in PIPELINES}}
        report[slug] = {"basin": name, **r}
        print(f"\n=== {name} ===  " + "  ".join(f"{k} {v:.3f}" for k, v in r["roc_mean"].items()))
        for grp in ("vs_coords", "vs_A", "vs_A+elev"):
            for k, v in r[grp].items():
                print(f"  {grp:9s} {k:10s} {v['mean']:+.3f} t[{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}]"
                      f"{'*' if v['sig'] else ' '} NB[{v['ci_nb'][0]:+.3f},{v['ci_nb'][1]:+.3f}]"
                      f"{'*' if v['sig_nb'] else ''}")
    out = RESULTS / "floor_benchmark_folds.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"\n[write] {out.name}")


if __name__ == "__main__":
    main()
