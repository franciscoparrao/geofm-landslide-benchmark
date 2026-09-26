"""Movement typology: composition, terrain signature, and a single-type arm.

The benchmark pools every mass movement into one binary positive class. Debris
flows, falls and slides respond to different conditioning factors, so a fair
objection is that the comparison is run on a heterogeneous target and that the
result might be driven by whichever type dominates a basin. A second objection
is that catalogue coordinates may mark deposits or reported damage rather than
source areas, in which case part of what is being modelled is runout.

The `type` field exists in the inventories and was simply never used. This
script reports three things:

  1. Composition per basin. Huasco and Maipo carry SERNAGEOMIN's Spanish
     taxonomy; Maule is dominated by the Serey et al. inventory, which follows
     Keefer's coseismic scheme (disrupted / coherent / lateral spread). The two
     taxonomies are not interchangeable and the manuscript should say so.

  2. Terrain signature per type. If flow-type coordinates sat systematically
     lower and flatter than fall or slide coordinates within the same basin,
     that would be the fingerprint of deposits rather than scarps. This cannot
     settle the question -- only the catalogue's own documentation can -- but it
     bounds it with evidence rather than leaving it open.

  3. A single-type arm. Restricting the positive class to each basin's dominant
     type, holding negatives, folds and classifier fixed, tests whether the
     foundation-model result survives on a homogeneous target. Embeddings come
     from the published cache, subset by index, so the representation is
     untouched.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter

import numpy as np
import rasterio
from scipy.stats import t as student_t
from sklearn.model_selection import GroupKFold

from config import ML_DATASET_BASE, RESULTS, basin_dir, feature_paths
from point_probes import build_dataset, rf_fold_metrics
from terramind_linprobe import (
    N_FOLDS, PATCH_SIZE, PRITHVI_NAME, TERRAMIND_NAME, embedding_cache_path,
    embedding_fingerprint, load_events_xy, lookup_pixel_features,
)

BASINS = ["06_rio_huasco", "09_rio_maipo", "11_rio_maule"]
DOMINANT = {"06_rio_huasco": "Flujo", "09_rio_maipo": "Flujo",
            "11_rio_maule": "disrupted"}


def tci(v):
    n = len(v)
    m = sum(v) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1))
    h = float(student_t.ppf(0.975, n - 1)) * sd / math.sqrt(n)
    return m, m - h, m + h, not (m - h <= 0 <= m + h)


def fmt(v):
    m, lo, hi, s = tci(v)
    return f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]{'*' if s else ''}"


def aligned_types(basin, n_pos_expected):
    """Types of the positives that survive the pipeline's filters, in order.

    load_events_xy reads ml_dataset in file order keeping label==1, so the CSV
    type column aligns 1:1 with the events before filtering. The in-bounds and
    pixel-validity filters are replayed here in the same order build_dataset
    applies them; the count is asserted against the pipeline's own n_pos so a
    silent misalignment cannot pass.
    """
    types = [r["type"] for r in csv.DictReader(open(ML_DATASET_BASE / f"{basin}.csv"))
             if int(r.get("label", "0")) == 1]
    src = rasterio.open(basin_dir(basin) / "dem_30m.tif")
    h, w = src.shape
    pr, pc = load_events_xy(basin, src.crs, src.transform)
    t = np.array(types, dtype=object)
    inb = ((pr >= PATCH_SIZE // 2) & (pr < h - PATCH_SIZE // 2)
           & (pc >= PATCH_SIZE // 2) & (pc < w - PATCH_SIZE // 2))
    pr, pc, t = pr[inb], pc[inb], t[inb]

    with np.load(RESULTS / f"{basin}_stack.npz", allow_pickle=False) as npz:
        X_all = npz["X"].astype(np.float32, copy=False)
        valid_idx = npz["valid_idx"]
    _, ok = lookup_pixel_features(X_all, valid_idx,
                                  pr.astype(np.int64) * w + pc)
    t = t[ok]
    if len(t) != n_pos_expected:
        raise SystemExit(f"{basin}: aligned {len(t)} types vs {n_pos_expected} positives")
    return t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=N_FOLDS)
    a = ap.parse_args()
    report = {}

    for b in BASINS:
        rows, cols, y, X17, blk = build_dataset(b, negatives="uniform")
        npos = int(y.sum())
        t = aligned_types(b, npos)
        comp = Counter(t)
        print(f"\n=== {b}  n_pos={npos} ===")
        for k, v in comp.most_common():
            print(f"   {k:26s} {v:4d}  {v / npos * 100:5.1f}%")

        with rasterio.open(feature_paths(b)["slope"]) as s:
            slope = s.read(1)[rows, cols]
        with rasterio.open(basin_dir(b) / "dem_30m.tif") as s:
            elev = s.read(1)[rows, cols]
        pos = y == 1
        print("   -- terrain signature by type --")
        sig = {}
        for k, v in comp.most_common():
            if v < 5:
                continue
            m = pos.copy()
            m[pos] = (t == k)
            sig[k] = {"n": int(v), "slope": float(slope[m].mean()),
                      "elev": float(elev[m].mean())}
            print(f"   {k:26s} slope {slope[m].mean():5.1f} deg   "
                  f"elev {elev[m].mean():7.0f} m")
        print(f"   {'(negatives)':26s} slope {slope[~pos].mean():5.1f} deg   "
              f"elev {elev[~pos].mean():7.0f} m")

        # single-type arm
        dom = DOMINANT[b]
        keep = ~pos.copy()
        keep[pos] = (t == dom)
        sub = np.flatnonzero(keep | (~pos))
        ysub = y[sub]
        cv = list(GroupKFold(n_splits=a.folds).split(X17[sub], ysub, groups=blk[sub]))
        per_fold_pos = [int(ysub[te].sum()) for _, te in cv]
        print(f"   -- single-type arm ({dom}, {comp[dom]} positives) --")
        print(f"   positives per test fold: {per_fold_pos}")
        if min(per_fold_pos) == 0:
            print("   SKIPPED: a test fold has no positives at this restriction")
            report[b] = {"composition": dict(comp), "signature": sig,
                         "single_type": None}
            continue

        sets = {"A": X17[sub]}
        for label, enc, mods in [("Prithvi", PRITHVI_NAME, ["S2L2A"]),
                                 ("TM+DEM", TERRAMIND_NAME, ["DEM"])]:
            slug = "terramind" if enc == TERRAMIND_NAME else enc
            d = np.load(embedding_cache_path(b, slug, "pretrained", mods),
                        allow_pickle=False)
            want = embedding_fingerprint(enc, "pretrained", mods, rows, cols, y)
            if str(d["fingerprint"]) != want:
                raise SystemExit(f"{b} {label}: cache is for other points")
            sets[label] = d["embeddings"].astype(np.float32)[sub]

        res = {}
        for k, X in sets.items():
            roc, pr_ = rf_fold_metrics(X, ysub, cv)
            res[k] = {"roc": roc, "pr": pr_}
            print(f"   {k:9s} ROC={np.mean(roc):.3f}  PR={np.mean(pr_):.3f}")
        deltas = {}
        for m in ["Prithvi", "TM+DEM"]:
            dv = [x - z for x, z in zip(res[m]["roc"], res["A"]["roc"])]
            deltas[m] = {"mean": tci(dv)[0], "ci": list(tci(dv)[1:3]),
                         "significant": tci(dv)[3]}
            print(f"   {m:9s} vs A: {fmt(dv)}")
        report[b] = {"composition": dict(comp), "signature": sig,
                     "single_type": {"type": dom, "n_positives": int(comp[dom]),
                                     "per_fold_positives": per_fold_pos,
                                     "sets": res, "deltas": deltas}}

    dst = RESULTS / "typology_arm.json"
    dst.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n[done] wrote {dst.name}")


if __name__ == "__main__":
    main()
