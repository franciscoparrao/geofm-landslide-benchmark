"""Is the Random Forest the reason the foundation models lose?

Review of this work raised the objection squarely: the benchmark scores a
1024-dimensional dense embedding and a 17-column geomorphometric table with the
same untuned forest, and `max_features='sqrt'` means each split inspects four of
seventeen manual features but thirty-two of a thousand embedding dimensions.
Axis-aligned trees on rotation-entangled transformer features are a weak reader,
which is exactly why the frozen-encoder literature settled on a regularised
linear probe. If the deficit is an artefact of the classifier, a linear probe on
the same cached embeddings should close it.

So this runs three readers on identical points, folds and embeddings:

  * the benchmark's Random Forest, re-fitted here as an internal control (it must
    reproduce Table 3 or something is wrong with this script, not with the paper)
  * a regularised logistic probe on standardised features, with C chosen by an
    inner split carved from the training folds only
  * a gradient-boosted tree ensemble, which is what this journal's own corpus
    most often pits against a forest

Everything else is held fixed: the same spatial blocks, the same 5:1 negatives,
the same fold assignment. The manual baseline A is re-fitted with each reader in
turn, so every comparison is reader-matched -- comparing an FM under a linear
probe against a baseline under a forest would answer a different question.

Output: {basin}_linear_probe_arm.json with per-fold ROC for every (reader,
pipeline) cell and fold-level paired intervals against that reader's own A.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from config import RESULTS
from point_probes import build_dataset
from terramind_linprobe import (
    N_FOLDS, N_TREES, PRITHVI_NAME, SEED, TERRAMIND_NAME,
    embedding_cache_path, embedding_fingerprint,
)

BASINS = ["06_rio_huasco", "09_rio_maipo", "11_rio_maule"]
T_CRIT = {5: 2.776, 10: 2.262}
C_GRID = (0.01, 0.1, 1.0, 10.0)


def _forest(seed):
    return RandomForestClassifier(
        n_estimators=N_TREES, min_samples_leaf=5, n_jobs=-1,
        random_state=seed, class_weight="balanced",
    )


def _boost(seed):
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.1, max_leaf_nodes=31,
        l2_regularization=1.0, random_state=seed,
    )


def _linear(seed, X, y):
    """Logistic probe; C picked on an inner stratified split of the training data."""
    best, best_auc = C_GRID[0], -1.0
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    for C in C_GRID:
        aucs = []
        for tr, va in inner.split(X, y):
            m = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=C, max_iter=2000, class_weight="balanced"),
            ).fit(X[tr], y[tr])
            aucs.append(roc_auc_score(y[va], m.predict_proba(X[va])[:, 1]))
        if st.mean(aucs) > best_auc:
            best, best_auc = C, st.mean(aucs)
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=best, max_iter=2000, class_weight="balanced"),
    ), best


def fold_scores(reader, X, y, folds):
    out = []
    for k, (tr, te) in enumerate(folds):
        if reader == "rf":
            m = _forest(SEED + k).fit(X[tr], y[tr])
        elif reader == "gb":
            m = _boost(SEED + k).fit(X[tr], y[tr])
        else:
            m, _ = _linear(SEED + k, X[tr], y[tr])
            m.fit(X[tr], y[tr])
        out.append(float(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1])))
    return out


def paired(a, b):
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    m = st.mean(d)
    h = T_CRIT[n] * st.stdev(d) / math.sqrt(n)
    return {"mean": m, "ci": [m - h, m + h], "significant": bool(m - h > 0 or m + h < 0)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--basin", action="append", choices=BASINS)
    ap.add_argument("--folds", type=int, default=N_FOLDS)
    a = ap.parse_args()

    for basin in (a.basin or BASINS):
        rows, cols, y, X17, blk = build_dataset(basin, negatives="uniform")
        folds = list(GroupKFold(n_splits=a.folds).split(X17, y, groups=blk))

        spaces = {"A": X17.astype(np.float32)}
        for lbl, enc, mods in (("Prithvi", PRITHVI_NAME, ["S2L2A"]),
                               ("TM+DEM", TERRAMIND_NAME, ["DEM"])):
            slug = "terramind" if enc == TERRAMIND_NAME else enc
            d = np.load(embedding_cache_path(basin, slug, "pretrained", mods),
                        allow_pickle=False)
            if str(d["fingerprint"]) != embedding_fingerprint(
                    enc, "pretrained", mods, rows, cols, y):
                raise SystemExit(f"{basin} {lbl}: cache describes another point set")
            spaces[lbl] = d["embeddings"].astype(np.float32)

        print(f"\n=== {basin} ===")
        print(f"{'reader':<10}{'A':>8}{'Prithvi':>10}{'TM+DEM':>9}"
              f"{'Pr - A':>22}{'TM - A':>22}")
        report = {"basin": basin, "n_folds": a.folds, "readers": {}}
        for reader, name in (("rf", "RandomForest"), ("lin", "Logistic"),
                             ("gb", "HistGradBoost")):
            sc = {k: fold_scores(reader, X, y, folds) for k, X in spaces.items()}
            cells = {k: {"roc": v, "mean": st.mean(v)} for k, v in sc.items()}
            dp = paired(sc["Prithvi"], sc["A"])
            dt = paired(sc["TM+DEM"], sc["A"])
            report["readers"][name] = {"cells": cells,
                                       "Prithvi_vs_A": dp, "TM+DEM_vs_A": dt}
            s = lambda d_: "*" if d_["significant"] else " "
            print(f"{name:<10}{st.mean(sc['A']):>8.3f}{st.mean(sc['Prithvi']):>10.3f}"
                  f"{st.mean(sc['TM+DEM']):>9.3f}"
                  f"  {dp['mean']:+.3f}{s(dp)}[{dp['ci'][0]:+.3f},{dp['ci'][1]:+.3f}]"
                  f"  {dt['mean']:+.3f}{s(dt)}[{dt['ci'][0]:+.3f},{dt['ci'][1]:+.3f}]")

        dst = RESULTS / f"{basin}_linear_probe_arm.json"
        dst.write_text(json.dumps(report, indent=2))
        print(f"[done] wrote {dst.name}")


if __name__ == "__main__":
    main()
