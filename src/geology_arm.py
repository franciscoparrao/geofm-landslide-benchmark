"""Does the manual baseline change once geology is in it?

Pipeline A is purely geomorphometric. The scoping reason was symmetry: TerraMind
receives a DEM patch, so the manual comparator was built from DEM derivatives and
nothing else. That rationale did not survive the ground-motion control, which
supplied a variable neither competitor could derive from the DEM and gave it to
both sides; the same treatment is owed to geology, which review identified as the
conditioning factor most conspicuously missing from a susceptibility baseline.

The rasters exist in the processed substrate: lithology class, rock type and
geological age, rasterised from the SERNAGEOMIN 1:1,000,000 national map. That
scale is coarse -- four lithology classes and nine to thirteen rock types per
basin -- and coarseness is itself part of what this arm measures: if a national
map at 1:1M cannot improve the baseline, that is worth reporting, because it is
the coverage most practitioners in this region actually have.

The three layers are categorical and are one-hot encoded, fitted on the training
folds only so that a class absent from training cannot leak in through the
encoding. They are offered to both competitors -- appended to the manual stack and
concatenated to the frozen embeddings -- on the published points, folds and
classifier.
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np
import rasterio
from scipy.stats import t as student_t
from sklearn.model_selection import GroupKFold

from config import RESULTS, basin_dir
from point_probes import build_dataset, rf_fold_metrics
from terramind_linprobe import (
    N_FOLDS, PRITHVI_NAME, TERRAMIND_NAME, embedding_cache_path,
    embedding_fingerprint,
)

LAYERS = ["lithology_class", "rock_type", "geological_age"]
BASINS = [("06_rio_huasco", "Huasco"), ("09_rio_maipo", "Maipo"),
          ("11_rio_maule", "Maule")]


def tci(v):
    n = len(v)
    m = sum(v) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1))
    h = float(student_t.ppf(0.975, n - 1)) * sd / math.sqrt(n)
    return m, m - h, m + h, not (m - h <= 0 <= m + h)


def fmt(v):
    m, lo, hi, s = tci(v)
    return f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]{'*' if s else ''}"


def sample_geology(slug, rows, cols):
    """Raw categorical codes at each point, one column per layer."""
    out = []
    for name in LAYERS:
        with rasterio.open(basin_dir(slug) / "geology" / f"{name}.tif") as src:
            band = src.read(1)
            out.append(band[rows, cols].astype(np.float64))
    return np.stack(out, axis=1)


def onehot(codes_tr, codes_all):
    """One-hot over classes seen in training; unseen classes map to all-zero."""
    blocks = []
    for j in range(codes_all.shape[1]):
        classes = np.unique(codes_tr[:, j])
        blocks.append((codes_all[:, [j]] == classes[None, :]).astype(np.float32))
    return np.hstack(blocks)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=N_FOLDS)
    a = ap.parse_args()
    report = {}

    for slug, name in BASINS:
        rows, cols, y, X17, blk = build_dataset(slug, negatives="uniform")
        geo = sample_geology(slug, rows, cols)
        nclass = [len(np.unique(geo[:, j])) for j in range(geo.shape[1])]
        print(f"\n=== {name} === classes per layer: "
              + ", ".join(f"{l} {c}" for l, c in zip(LAYERS, nclass)))

        cv = list(GroupKFold(n_splits=a.folds).split(X17, y, groups=blk))
        emb = {}
        for lbl, enc, mods in [("Prithvi", PRITHVI_NAME, ["S2L2A"]),
                               ("TM+DEM", TERRAMIND_NAME, ["DEM"])]:
            s_ = "terramind" if enc == TERRAMIND_NAME else enc
            d = np.load(embedding_cache_path(slug, s_, "pretrained", mods),
                        allow_pickle=False)
            if str(d["fingerprint"]) != embedding_fingerprint(
                    enc, "pretrained", mods, rows, cols, y):
                raise SystemExit(f"{slug} {lbl}: cache is for another point set")
            emb[lbl] = d["embeddings"].astype(np.float32)

        # Encoding is fitted per fold on the training rows only.
        res = {k: [] for k in ["A", "A+GEO", "Prithvi", "Prithvi+GEO",
                               "TM+DEM", "TM+DEM+GEO"]}
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import roc_auc_score
        from terramind_linprobe import N_TREES, SEED
        for k, (tr, te) in enumerate(cv):
            G = onehot(geo[tr], geo)
            feats = {"A": X17, "A+GEO": np.hstack([X17, G]),
                     "Prithvi": emb["Prithvi"],
                     "Prithvi+GEO": np.hstack([emb["Prithvi"], G]),
                     "TM+DEM": emb["TM+DEM"],
                     "TM+DEM+GEO": np.hstack([emb["TM+DEM"], G])}
            for lbl, X in feats.items():
                clf = RandomForestClassifier(
                    n_estimators=N_TREES, min_samples_leaf=5, n_jobs=-1,
                    random_state=SEED + k, class_weight="balanced",
                ).fit(X[tr], y[tr])
                res[lbl].append(float(roc_auc_score(
                    y[te], clf.predict_proba(X[te])[:, 1])))
        for lbl in res:
            print(f"   {lbl:12s} ROC={np.mean(res[lbl]):.3f}")

        def d(x, z):
            return [p - q for p, q in zip(res[x], res[z])]
        print(f"   A+GEO vs A            : {fmt(d('A+GEO','A'))}")
        for m in ["Prithvi", "TM+DEM"]:
            print(f"   {m:8s} vs A+GEO      : {fmt(d(m,'A+GEO'))}")
            print(f"   {m}+GEO vs A+GEO  : {fmt(d(f'{m}+GEO','A+GEO'))}")
        report[slug] = {"basin": name, "classes_per_layer": nclass, "roc": res,
                        "deltas": {f"{x}_vs_{z}": {"mean": tci(d(x, z))[0],
                                                   "ci": list(tci(d(x, z))[1:3]),
                                                   "significant": tci(d(x, z))[3]}
                                   for x, z in [("A+GEO", "A"),
                                                ("Prithvi", "A"), ("TM+DEM", "A"),
                                                ("Prithvi", "A+GEO"), ("TM+DEM", "A+GEO"),
                                                ("Prithvi+GEO", "A+GEO"),
                                                ("TM+DEM+GEO", "A+GEO")]}}

    dst = RESULTS / "geology_arm.json"
    dst.write_text(json.dumps(report, indent=2))
    print(f"\n[done] wrote {dst.name}")


if __name__ == "__main__":
    main()
