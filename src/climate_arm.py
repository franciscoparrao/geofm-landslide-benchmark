"""Does the manual baseline change once climate is in it?

Companion to the geology arm, and the second half of the same criticism: a
susceptibility baseline built only from DEM derivatives omits the conditioning
factors the field regards as first-order, and rainfall is the trigger in two of
these three basins. The WorldClim normals already sit in the processed substrate
at 30 m and had never been used: annual precipitation, precipitation of the
wettest and driest months, and precipitation seasonality.

These are climatological normals, not event rainfall. That is the right variable
type for a static susceptibility model, which estimates where failure is possible
rather than when it occurs; event-specific rainfall would support a different and
more demanding design, discussed in the manuscript.

As with ground motion and geology, the layers go to both competitors, on the
published points, folds and classifier.
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
from sklearn.model_selection import GroupKFold

from config import RESULTS, basin_dir
from point_probes import build_dataset
from terramind_linprobe import (
    N_FOLDS, N_TREES, PRITHVI_NAME, SEED, TERRAMIND_NAME, embedding_cache_path,
    embedding_fingerprint,
)

LAYERS = ["bio_12", "bio_13", "bio_14", "bio_15"]   # annual, wettest, driest, seasonality
BASINS = [("06_rio_huasco", "Huasco"), ("09_rio_maipo", "Maipo"),
          ("11_rio_maule", "Maule")]


def tci(v):
    n = len(v); m = sum(v) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1))
    h = float(student_t.ppf(0.975, n - 1)) * sd / math.sqrt(n)
    return m, m - h, m + h, not (m - h <= 0 <= m + h)


def fmt(v):
    m, lo, hi, s = tci(v)
    return f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]{'*' if s else ''}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=N_FOLDS)
    a = ap.parse_args()
    report = {}
    for slug, name in BASINS:
        rows, cols, y, X17, blk = build_dataset(slug, negatives="uniform")
        cl = []
        for lay in LAYERS:
            with rasterio.open(basin_dir(slug) / "climate" / f"{lay}.tif") as s_:
                b = s_.read(1).astype(np.float32)
                v = b[rows, cols]
            cl.append(np.where(np.isfinite(v) & (v > -1e30), v, np.nan))
        C = np.stack(cl, axis=1)
        bad = int(np.isnan(C).any(axis=1).sum())
        C = np.nan_to_num(C, nan=float(np.nanmedian(C)))
        print(f"\n=== {name} === climate sampled, {bad} of {len(rows)} points "
              f"outside coverage (median-filled); bio_12 range "
              f"{np.nanmin(C[:,0]):.0f}-{np.nanmax(C[:,0]):.0f} mm")

        cv = list(GroupKFold(n_splits=a.folds).split(X17, y, groups=blk))
        emb = {}
        for lbl, enc, mods in [("Prithvi", PRITHVI_NAME, ["S2L2A"]),
                               ("TM+DEM", TERRAMIND_NAME, ["DEM"])]:
            s2 = "terramind" if enc == TERRAMIND_NAME else enc
            d = np.load(embedding_cache_path(slug, s2, "pretrained", mods),
                        allow_pickle=False)
            if str(d["fingerprint"]) != embedding_fingerprint(
                    enc, "pretrained", mods, rows, cols, y):
                raise SystemExit(f"{slug} {lbl}: cache is for another point set")
            emb[lbl] = d["embeddings"].astype(np.float32)

        feats = {"A": X17, "A+CLIM": np.hstack([X17, C]),
                 "Prithvi": emb["Prithvi"],
                 "Prithvi+CLIM": np.hstack([emb["Prithvi"], C]),
                 "TM+DEM": emb["TM+DEM"],
                 "TM+DEM+CLIM": np.hstack([emb["TM+DEM"], C])}
        res = {}
        for lbl, X in feats.items():
            r = []
            for k, (tr, te) in enumerate(cv):
                clf = RandomForestClassifier(
                    n_estimators=N_TREES, min_samples_leaf=5, n_jobs=-1,
                    random_state=SEED + k, class_weight="balanced").fit(X[tr], y[tr])
                r.append(float(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1])))
            res[lbl] = r
            print(f"   {lbl:14s} ROC={np.mean(r):.3f}")
        def d(x, z): return [p - q for p, q in zip(res[x], res[z])]
        print(f"   A+CLIM vs A            : {fmt(d('A+CLIM','A'))}")
        for m in ["Prithvi", "TM+DEM"]:
            print(f"   {m:8s} vs A+CLIM      : {fmt(d(m,'A+CLIM'))}")
        report[slug] = {"basin": name, "n_outside_coverage": bad, "roc": res,
                        "deltas": {f"{x}_vs_{z}": {"mean": tci(d(x,z))[0],
                                                   "ci": list(tci(d(x,z))[1:3]),
                                                   "significant": tci(d(x,z))[3]}
                                   for x,z in [("A+CLIM","A"),("Prithvi","A+CLIM"),
                                               ("TM+DEM","A+CLIM"),
                                               ("Prithvi+CLIM","A+CLIM"),
                                               ("TM+DEM+CLIM","A+CLIM")]}}
    (RESULTS / "climate_arm.json").write_text(json.dumps(report, indent=2))
    print("\n[done] wrote climate_arm.json")


if __name__ == "__main__":
    main()
