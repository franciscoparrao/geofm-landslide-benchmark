"""Range of spatial dependence, to set the block size against data, not geometry.

Blind review b1 (2026-10-07) noted that the 10 km block of the benchmark is set
by the patch geometry (a 6.72 km patch plus margin), not by an estimated range
of spatial dependence. This script estimates that range three ways per basin:

  covariates  empirical variogram of each of the seventeen standardised layers
              over a uniform sample of basin pixels; the median practical range
              across layers is the blockCV summary (Valavi et al. 2019)
  labels      indicator variogram of presence/background on the benchmark points
  residuals   variogram of the out-of-fold residuals y - p of the manual
              baseline (Random Forest on the seventeen layers, benchmark folds);
              this is the dependence the cross-validation actually has to break

Label and residual variograms are fitted with an exponential and a spherical
model with nugget, weighted by the number of pairs per lag; the practical range
is the distance at which the model reaches 95 % of its sill (3a for the
exponential, a for the spherical). A fit whose range hits the maximum lag is
reported as censored: the dependence extends at least that far. Below the
negative-exclusion radius (BUFFER_PX, 3.36 km) the benchmark has no
presence/background pairs by design, so the label variogram is zero there and
the residual variogram is biased low; both are fitted on all lags and read
with that caveat.

The covariate variograms are nested: a local structure that levels off within
a few kilometres sits on a regional gradient (coast to cordillera) that has no
sill inside a basin. A single-structure model then fits the gradient and its
range runs to the maximum lag. They are fitted instead with a nugget, a
spherical local structure and a linear drift, and the local range is the
spherical one. A layer whose local structure carries under 10 % of the
variance at the first lag beyond its range has no resolvable local range.

Output: results/dependence_range.json
"""
from __future__ import annotations

import json

import numpy as np
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

from config import RESULTS
from point_probes import build_dataset
from terramind_linprobe import BUFFER_PX, N_FOLDS, N_TREES, SEED, SPATIAL_BLOCK_PX

BASINS = [("06_rio_huasco", "Huasco"), ("09_rio_maipo", "Maipo"),
          ("11_rio_maule", "Maule")]
PX_KM = 0.030
MAX_LAG_KM = 60.0
EDGES_KM = np.r_[0.0, np.geomspace(0.25, MAX_LAG_KM, 28)]
N_COV_SAMPLE = 10_000


def pairs_within(rows, cols, max_km=MAX_LAG_KM):
    """Index pairs closer than max_km and their distances in km."""
    xy = np.c_[rows, cols].astype(np.float64) * PX_KM
    ij = cKDTree(xy).query_pairs(max_km, output_type="ndarray")
    d = np.hypot(*(xy[ij[:, 0]] - xy[ij[:, 1]]).T)
    return ij, d


def empirical(z, ij, d):
    """Binned semivariance; returns lag centres, gamma, pair counts."""
    b = np.digitize(d, EDGES_KM) - 1
    sq = 0.5 * (z[ij[:, 0]] - z[ij[:, 1]]) ** 2
    n = np.bincount(b, minlength=len(EDGES_KM) - 1)[:len(EDGES_KM) - 1]
    s = np.bincount(b, weights=sq, minlength=len(EDGES_KM) - 1)[:len(EDGES_KM) - 1]
    h = np.bincount(b, weights=d, minlength=len(EDGES_KM) - 1)[:len(EDGES_KM) - 1]
    ok = n >= 30
    return h[ok] / n[ok], s[ok] / n[ok], n[ok]


def exponential(h, c0, c, a):
    return c0 + c * (1 - np.exp(-h / a))


def spherical(h, c0, c, a):
    r = np.minimum(h / a, 1.0)
    return c0 + c * (1.5 * r - 0.5 * r ** 3)


def fit(h, g, n):
    """Practical range (km) of both models; None when the fit fails."""
    out = {}
    sill0 = float(np.median(g[-5:]))
    for name, f, to_range in (("exponential", exponential, 3.0),
                              ("spherical", spherical, 1.0)):
        try:
            p, _ = curve_fit(f, h, g, p0=[0.1 * sill0, 0.9 * sill0, 5.0],
                             sigma=1 / np.sqrt(n), bounds=([0, 0, 0.05], [np.inf, np.inf, 10 * MAX_LAG_KM]),
                             maxfev=20000)
        except RuntimeError:
            out[name] = None
            continue
        rng = float(to_range * p[2])
        out[name] = {"nugget": float(p[0]), "psill": float(p[1]),
                     "range_km": min(rng, MAX_LAG_KM), "censored": rng >= MAX_LAG_KM,
                     "nugget_ratio": float(p[0] / (p[0] + p[1])) if p[0] + p[1] > 0 else None}
    return out


def nested(h, c0, c1, a1, b):
    return spherical(h, c0, c1, a1) + b * h


def fit_nested(h, g, n):
    """Local range (km) of nugget + spherical + linear drift."""
    try:
        p, _ = curve_fit(nested, h, g, p0=[0.1 * g[0], max(g[len(g) // 2] - g[0], 1e-3), 2.0, 1e-3],
                         sigma=1 / np.sqrt(n), bounds=([0, 0, 0.05, 0], [np.inf, np.inf, MAX_LAG_KM, np.inf]),
                         maxfev=20000)
    except RuntimeError:
        return None
    c0, c1, a1, b = map(float, p)
    share = c1 / (c0 + c1 + b * a1) if c0 + c1 + b * a1 > 0 else 0.0
    return {"nugget": c0, "psill_local": c1, "range_local_km": a1, "drift_per_km": b,
            "local_share": share, "resolved": share >= 0.10 and a1 < 0.9 * MAX_LAG_KM}


def summarise_cov(v):
    return {"lag_km": v[0].round(3).tolist(), "gamma": v[1].tolist(), "n_pairs": v[2].tolist(),
            "fit": fit(*v), "nested": fit_nested(*v)}


def summarise(v):
    return {"lag_km": v[0].round(3).tolist(), "gamma": v[1].tolist(), "n_pairs": v[2].tolist(),
            "fit": fit(*v)}


def oof_residuals(X, y, folds):
    p = np.empty(len(y), dtype=np.float64)
    for k, (tr, te) in enumerate(folds):
        clf = RandomForestClassifier(n_estimators=N_TREES, min_samples_leaf=5, n_jobs=-1,
                                     random_state=SEED + k, class_weight="balanced")
        clf.fit(X[tr], y[tr])
        p[te] = clf.predict_proba(X[te])[:, 1]
    return y - p


def covariate_variograms(slug, rng):
    with np.load(RESULTS / f"{slug}_stack.npz", allow_pickle=False) as npz:
        valid_idx, width = npz["valid_idx"], int(npz["width"])
        names = [str(s) for s in npz["feature_names"]]
        pick = np.sort(rng.choice(valid_idx.size, size=N_COV_SAMPLE, replace=False))
        X = npz["X"][pick].astype(np.float64)
        flat = valid_idx[pick]
    X = (X - X.mean(0)) / X.std(0)
    ij, d = pairs_within(flat // width, flat % width)
    return {nm: summarise_cov(empirical(X[:, j], ij, d)) for j, nm in enumerate(names)}


def median_local_range(per_layer):
    r = [v["nested"]["range_local_km"] for v in per_layer.values()
         if v["nested"] and v["nested"]["resolved"]]
    return {"median_km": float(np.median(r)) if r else None,
            "iqr_km": [float(np.percentile(r, 25)), float(np.percentile(r, 75))] if r else None,
            "n_resolved": len(r), "n_layers": len(per_layer)}


def main():
    report = {"block_km": SPATIAL_BLOCK_PX * PX_KM, "max_lag_km": MAX_LAG_KM,
              "exclusion_radius_km": BUFFER_PX * PX_KM}
    for slug, name in BASINS:
        rng = np.random.default_rng(SEED)
        rows, cols, y, X17, blk = build_dataset(slug, negatives="uniform")
        folds = list(GroupKFold(n_splits=N_FOLDS).split(X17, y, groups=blk))
        ij, d = pairs_within(rows, cols)
        cov = covariate_variograms(slug, rng)
        r = {"basin": name, "n_points": int(len(y)), "n_pos": int(y.sum()),
             "labels": summarise(empirical(y.astype(np.float64), ij, d)),
             "residuals_A": summarise(empirical(oof_residuals(X17, y, folds), ij, d)),
             "covariates": cov,
             "covariates_local_range": median_local_range(cov)}
        report[slug] = r
        print(f"\n=== {name} ===  n={r['n_points']} pos={r['n_pos']}")
        for key in ("labels", "residuals_A"):
            for m, v in r[key]["fit"].items():
                if v:
                    print(f"  {key:12s} {m:11s} range {v['range_km']:6.1f} km"
                          f"{' (censored)' if v['censored'] else ''}  nugget/sill {v['nugget_ratio']:.2f}")
        cl = r["covariates_local_range"]
        print(f"  covariates   local range median {cl['median_km']:.1f} km  IQR {cl['iqr_km']}"
              f"  ({cl['n_resolved']}/{cl['n_layers']} layers resolved)")
        for nm, v in cov.items():
            e = v["nested"]
            if e:
                print(f"    {nm:28s} local {e['range_local_km']:5.1f} km  share {e['local_share']:.2f}"
                      f"  drift {e['drift_per_km']:.4f}/km{'' if e['resolved'] else '  (unresolved)'}")
    out = RESULTS / "dependence_range.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"\n[write] {out.name}")


if __name__ == "__main__":
    main()
