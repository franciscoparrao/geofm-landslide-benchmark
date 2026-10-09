"""Does the Maule deficit survive a reader tuned for each feature space?

Blind review b1 (2026-10-07): the benchmark reads a 1024-dimensional embedding
and a seventeen-column table with the same untuned forest, and a gradient-
boosting reader halves the deficit, so the gap may belong to the reader rather
than to the embedding. linear_probe_arm.py fixed the boosting settings and tuned
the logistic C on a random inner split, which leaks across blocks.

Here every reader is tuned by nested cross-validation, separately for every
feature space, so each side gets the reader settings that suit it:

  outer  the benchmark folds (GroupKFold, k=5, 10 km blocks), scored once
  inner  GroupKFold (k=4) over the blocks of the outer training set only; the
         grid point with the highest mean inner ROC AUC is refitted on the whole
         outer training set

Readers and grids:
  rf   Random Forest, 300 trees, balanced; max_features {sqrt, 0.1, 0.3} x
       min_samples_leaf {1, 5, 20}
  gb   HistGradientBoosting, 300 iterations; learning_rate {0.03, 0.1} x
       max_leaf_nodes {7, 31} x l2_regularization {0, 1}
  lin  standardised logistic regression, balanced; C {1e-3, 1e-2, 1e-1, 1, 10}

Paired differences against A under the same reader carry the fold-level
t-interval and the Nadeau-Bengio correction. The selected settings per outer
fold are kept.

Output: results/{basin}_tuned_reader.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import statistics as st

import numpy as np
import rasterio
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import RESULTS, basin_dir
from floor_benchmark_folds import intervals
from point_probes import build_dataset
from terramind_linprobe import (
    N_FOLDS, N_TREES, PRITHVI_NAME, SEED, TERRAMIND_NAME,
    embedding_cache_path, embedding_fingerprint, load_cached_embeddings,
)

N_INNER = 4
N_JOBS = int(os.environ.get("GEOFM_N_JOBS", "-1"))
PIPELINES = [("TM+DEM", TERRAMIND_NAME, "terramind", ["DEM"]),
             ("TM+DEM+S2", TERRAMIND_NAME, "terramind", ["DEM", "S2L2A"]),
             ("Prithvi", PRITHVI_NAME, "prithvi-300m", ["S2L2A"])]


def grid(reader):
    if reader == "rf":
        return [dict(max_features=f, min_samples_leaf=l)
                for f, l in itertools.product(("sqrt", 0.1, 0.3), (1, 5, 20))]
    if reader == "gb":
        return [dict(learning_rate=r, max_leaf_nodes=n, l2_regularization=g)
                for r, n, g in itertools.product((0.03, 0.1), (7, 31), (0.0, 1.0))]
    return [dict(C=c) for c in (1e-3, 1e-2, 1e-1, 1.0, 10.0)]


def make(reader, params, seed):
    if reader == "rf":
        return RandomForestClassifier(n_estimators=N_TREES, class_weight="balanced",
                                      n_jobs=N_JOBS, random_state=seed, **params)
    if reader == "gb":
        return HistGradientBoostingClassifier(max_iter=300, random_state=seed, **params)
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=5000, class_weight="balanced", **params))


def auc(m, X, y):
    return float(roc_auc_score(y, m.predict_proba(X)[:, 1]))


def nested(reader, X, y, groups, outer):
    """Outer-fold ROC with settings chosen on spatial inner folds."""
    rocs, chosen = [], []
    for k, (tr, te) in enumerate(outer):
        inner = list(GroupKFold(n_splits=N_INNER).split(X[tr], y[tr], groups=groups[tr]))
        best, best_score = None, -1.0
        for p in grid(reader):
            s = []
            for itr, iva in inner:
                ytr, yva = y[tr][itr], y[tr][iva]
                if len(np.unique(yva)) < 2:
                    continue
                s.append(auc(make(reader, p, SEED + k).fit(X[tr][itr], ytr), X[tr][iva], yva))
            if s and st.mean(s) > best_score:
                best, best_score = p, st.mean(s)
        m = make(reader, best, SEED + k).fit(X[tr], y[tr])
        rocs.append(auc(m, X[te], y[te]))
        chosen.append({"params": best, "inner_roc": best_score})
    return rocs, chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--basin", default="11_rio_maule")
    ap.add_argument("--readers", default="rf,gb,lin")
    a = ap.parse_args()

    rows, cols, y, X17, blk = build_dataset(a.basin)
    outer = list(GroupKFold(n_splits=N_FOLDS).split(X17, y, groups=blk))
    with rasterio.open(basin_dir(a.basin) / "dem_30m.tif") as src:
        elev = src.read(1)[rows, cols].astype(np.float32)
    spaces = {"coords": np.c_[rows, cols].astype(np.float32), "A": X17,
              "A+elev": np.c_[X17, elev]}
    for lbl, enc, s, mods in PIPELINES:
        emb = load_cached_embeddings(embedding_cache_path(a.basin, s, "pretrained", mods),
                                     embedding_fingerprint(enc, "pretrained", mods, rows, cols, y))
        if emb is None:
            raise SystemExit(f"{a.basin} {lbl}: no cached embedding for the benchmark points")
        spaces[lbl] = emb

    report = {"basin": a.basin, "n_outer": N_FOLDS, "n_inner": N_INNER, "readers": {}}
    for reader in a.readers.split(","):
        roc, chosen = {}, {}
        for name, X in spaces.items():
            roc[name], chosen[name] = nested(reader, X, y, blk, outer)
            print(f"  [{reader}] {name:10s} ROC {st.mean(roc[name]):.3f}  "
                  + " ".join(str(c["params"]) for c in chosen[name]), flush=True)
        diff = lambda p, q: [u - v for u, v in zip(roc[p], roc[q])]
        r = {"roc_folds": roc, "roc_mean": {k: st.mean(v) for k, v in roc.items()},
             "chosen": chosen,
             "vs_A": {k: intervals(diff(k, "A")) for k in roc if k not in ("A", "coords")},
             "vs_coords": {k: intervals(diff(k, "coords")) for k in roc if k != "coords"}}
        report["readers"][reader] = r
        for k, v in r["vs_A"].items():
            print(f"  [{reader}] {k:10s} - A  {v['mean']:+.3f} t[{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}]"
                  f"{'*' if v['sig'] else ' '} NB[{v['ci_nb'][0]:+.3f},{v['ci_nb'][1]:+.3f}]"
                  f"{'*' if v['sig_nb'] else ''}", flush=True)
    out = RESULTS / f"{a.basin}_tuned_reader.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"[write] {out.name}")


if __name__ == "__main__":
    main()
