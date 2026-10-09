"""Sampling and partition variance that the benchmark intervals leave out.

Blind review b1 (2026-10-07): negatives are drawn once (one seed) on one block
grid, so neither the negative draw nor the placement of the block grid enters
the reported intervals. Two re-analyses put them in.

  offsets  the same points on 3 x 3 = 9 placements of the 10 km grid, shifted
           by thirds of a block along each axis (offset (0, 0) is the
           benchmark). Every pipeline is re-scored, embeddings from the cache.
  draws    fresh negatives under the benchmark rules (in-basin, outside the
           exclusion window, patch zero-fraction matched to the positives),
           one seed per draw, on the benchmark grid. Positives are unchanged,
           so the foundation-model embeddings of the positives are reused and
           only the new negatives are encoded (--encode; CPU, about one point
           per second for Prithvi). Without --encode only the coordinates, A
           and A+elev are re-scored.

For every comparison it reports the fold-mean difference per grid or draw, the
benchmark value, and how many replicates keep its sign and exclude zero at
fold level.

Output: results/resampling_offsets.json, results/resampling_draws[_<basin>].json
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import rasterio
from scipy.stats import t as student_t
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from config import RESULTS, S2_COMPOSITE_BASE, basin_dir
from point_probes import build_dataset
from terramind_linprobe import (
    N_FOLDS, N_TREES, PRITHVI_HLS_INDICES, PRITHVI_NAME, SEED, SPATIAL_BLOCK_PX,
    TERRAMIND_NAME, build_encoder, embedding_cache_path, embedding_fingerprint,
    encode_streaming, load_cached_embeddings, save_cached_embeddings,
)

BASINS = [("06_rio_huasco", "Huasco"), ("09_rio_maipo", "Maipo"),
          ("11_rio_maule", "Maule")]
PIPELINES = [("TM+DEM", TERRAMIND_NAME, "terramind", ["DEM"]),
             ("TM+DEM+S2", TERRAMIND_NAME, "terramind", ["DEM", "S2L2A"]),
             ("Prithvi", PRITHVI_NAME, "prithvi-300m", ["S2L2A"])]
OFFSETS = [(dy, dx) for dy in (0, 111, 222) for dx in (0, 111, 222)]
DRAW_SEED_STEP = 1000
DRAW_CACHE = RESULTS / "_embcache_draws"
N_JOBS = int(os.environ.get("GEOFM_N_JOBS", "-1"))


def blocks(rows, cols, offset, width):
    """Block ids on a grid shifted by offset; numbered as build_dataset numbers
    them, so offset (0, 0) reproduces the benchmark groups and folds exactly."""
    dy, dx = offset
    n_cols = (width + dx + SPATIAL_BLOCK_PX - 1) // SPATIAL_BLOCK_PX
    return ((rows + dy) // SPATIAL_BLOCK_PX) * n_cols + (cols + dx) // SPATIAL_BLOCK_PX


def fold_auc(X, y, folds):
    out = []
    for k, (tr, te) in enumerate(folds):
        if len(np.unique(y[te])) < 2:
            return None
        clf = RandomForestClassifier(n_estimators=N_TREES, min_samples_leaf=5, n_jobs=N_JOBS,
                                     random_state=SEED + k, class_weight="balanced")
        clf.fit(X[tr], y[tr])
        out.append(float(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1])))
    return out


def paired(a, b):
    d = np.asarray(a) - np.asarray(b)
    m, s = float(d.mean()), float(d.std(ddof=1))
    h = float(student_t.ppf(0.975, len(d) - 1)) * s / math.sqrt(len(d))
    return {"mean": m, "ci": [m - h, m + h], "sig": not (m - h <= 0 <= m + h)}


def score(feats, y, groups):
    """Fold ROC of every feature set and the paired comparisons of interest."""
    folds = list(GroupKFold(n_splits=N_FOLDS).split(feats["A"], y, groups=groups))
    roc = {}
    for name, X in feats.items():
        r = fold_auc(X, y, folds)
        if r is None:
            return None
        roc[name] = r
    cmp = {f"A-coords": paired(roc["A"], roc["coords"])}
    for name in roc:
        if name not in ("A", "coords"):
            cmp[f"{name}-A"] = paired(roc[name], roc["A"])
            cmp[f"{name}-coords"] = paired(roc[name], roc["coords"])
    return {"roc_mean": {k: float(np.mean(v)) for k, v in roc.items()}, "cmp": cmp}


def summarise(reps):
    """Across replicates: per comparison, the spread and sign agreement with replicate 0."""
    ok = [r for r in reps if r is not None]
    out = {"n": len(ok), "n_failed": len(reps) - len(ok)}
    for key in ok[0]["cmp"]:
        m = np.array([r["cmp"][key]["mean"] for r in ok])
        ref = ok[0]["cmp"][key]["mean"]
        out[key] = {"benchmark": ref, "mean": float(m.mean()), "sd": float(m.std(ddof=1)),
                    "min": float(m.min()), "max": float(m.max()),
                    "same_sign": int((np.sign(m) == np.sign(ref)).sum()),
                    "sig_same_sign": int(sum(r["cmp"][key]["sig"] and np.sign(r["cmp"][key]["mean"]) == np.sign(ref)
                                             for r in ok))}
    return out


def manual_features(slug, rows, cols, X17):
    with rasterio.open(basin_dir(slug) / "dem_30m.tif") as src:
        elev = src.read(1)[rows, cols].astype(np.float32)
    return {"coords": np.c_[rows, cols].astype(np.float32), "A": X17, "A+elev": np.c_[X17, elev]}


def benchmark_embeddings(slug, rows, cols, y):
    out = {}
    for lbl, enc, s, mods in PIPELINES:
        emb = load_cached_embeddings(embedding_cache_path(slug, s, "pretrained", mods),
                                     embedding_fingerprint(enc, "pretrained", mods, rows, cols, y))
        if emb is None:
            raise SystemExit(f"{slug} {lbl}: no cached embedding for the benchmark points")
        out[lbl] = emb
    return out


def run_offsets():
    report = {"offsets_px": OFFSETS, "block_px": SPATIAL_BLOCK_PX}
    for slug, name in BASINS:
        rows, cols, y, X17, blk = build_dataset(slug)
        with rasterio.open(basin_dir(slug) / "dem_30m.tif") as src:
            width = src.width
        assert np.array_equal(blocks(rows, cols, (0, 0), width), blk)
        feats = {**manual_features(slug, rows, cols, X17), **benchmark_embeddings(slug, rows, cols, y)}
        reps = [score(feats, y, blocks(rows, cols, o, width)) for o in OFFSETS]
        report[slug] = {"basin": name, "replicates": reps, "summary": summarise(reps)}
        print_summary(name, report[slug]["summary"])
    (RESULTS / "resampling_offsets.json").write_text(json.dumps(report, indent=1))
    print("[write] resampling_offsets.json")


def draw_embeddings(slug, rows, cols, y, seed, bench_emb, bench_rows, bench_cols):
    """Embeddings of a draw: benchmark positives reused, new negatives encoded and cached."""
    n_pos = int(y.sum())
    assert np.array_equal(rows[:n_pos], bench_rows[:n_pos]) and np.array_equal(cols[:n_pos], bench_cols[:n_pos])
    nr, nc = rows[n_pos:], cols[n_pos:]
    out = {}
    dem_src = rasterio.open(basin_dir(slug) / "dem_30m.tif")
    s2_src = rasterio.open(S2_COMPOSITE_BASE / f"{slug}_s2l2a_2023.tif")
    for lbl, enc, s, mods in PIPELINES:
        path = DRAW_CACHE / f"{slug}_{s}_{'+'.join(sorted(mods)).lower()}_seed{seed}.npz"
        fp = embedding_fingerprint(enc, "pretrained", mods, nr, nc, np.zeros(len(nr)))
        neg = load_cached_embeddings(path, fp)
        if neg is None:
            model, kind, dim, norm = build_encoder(enc, pretrained=True, modalities=mods)
            model.eval()
            neg = encode_streaming(kind, model, dim, nr, nc,
                                   dem_src=dem_src if enc == TERRAMIND_NAME else None,
                                   s2_src=s2_src if "S2L2A" in mods else None,
                                   s2_band_indices=None if kind == "terramind" else PRITHVI_HLS_INDICES,
                                   s2_scale=1.0, norm_stats=norm)
            save_cached_embeddings(path, fp, neg)
            del model
        out[lbl] = np.r_[bench_emb[lbl][:n_pos], neg]
    return out


def run_draws(n_draws, encode, only):
    basins = [(s, n) for s, n in BASINS if only is None or s == only]
    report = {"n_draws": n_draws, "encode": encode, "seeds": [SEED + DRAW_SEED_STEP * d for d in range(n_draws)]}
    for slug, name in basins:
        b_rows, b_cols, b_y, _, _ = build_dataset(slug)
        bench_emb = benchmark_embeddings(slug, b_rows, b_cols, b_y) if encode else None
        reps = []
        for d in range(n_draws):
            seed = SEED + DRAW_SEED_STEP * d
            rows, cols, y, X17, blk = build_dataset(slug, seed=seed)
            feats = manual_features(slug, rows, cols, X17)
            if encode:
                feats.update(bench_emb if d == 0 else
                             draw_embeddings(slug, rows, cols, y, seed, bench_emb, b_rows, b_cols))
            reps.append(score(feats, y, blk))
            r = reps[-1]
            print(f"  [{name} draw {d} seed {seed}] " + ("failed" if r is None else
                  "  ".join(f"{k} {v['mean']:+.3f}{'*' if v['sig'] else ''}" for k, v in r["cmp"].items())), flush=True)
        report[slug] = {"basin": name, "replicates": reps, "summary": summarise(reps)}
        print_summary(name, report[slug]["summary"])
    suffix = "" if only is None else f"_{only}"
    suffix += "_fm" if encode else ""
    (RESULTS / f"resampling_draws{suffix}.json").write_text(json.dumps(report, indent=1))
    print(f"[write] resampling_draws{suffix}.json")


def print_summary(name, s):
    print(f"\n=== {name} === replicates {s['n']} (failed {s['n_failed']})")
    for k, v in s.items():
        if isinstance(v, dict):
            print(f"  {k:18s} bench {v['benchmark']:+.3f}  mean {v['mean']:+.3f} sd {v['sd']:.3f}"
                  f"  [{v['min']:+.3f},{v['max']:+.3f}]  same sign {v['same_sign']}/{s['n']}"
                  f"  sig {v['sig_same_sign']}/{s['n']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["offsets", "draws"])
    ap.add_argument("--n-draws", type=int, default=20)
    ap.add_argument("--encode", action="store_true", help="encode new negatives with the foundation models")
    ap.add_argument("--basin", default=None, help="restrict draws to one basin slug")
    a = ap.parse_args()
    if a.encode:
        import torch
        torch.set_num_threads(int(os.environ.get("GEOFM_TORCH_THREADS", "8")))
    run_offsets() if a.mode == "offsets" else run_draws(a.n_draws, a.encode, a.basin)


if __name__ == "__main__":
    main()
