"""Does the Maule baseline change once ground motion is in it?

Maule contributes the paper's central claim and 203 of its 211 events are
coseismic landslides from the 2010 Mw 8.8 earthquake, yet the seventeen-layer
stack is purely geomorphometric. For a coseismic inventory the shaking field is
a first-order control, so its absence leaves the manual baseline misspecified in
exactly the basin the claim rests on, and a reviewer is right to ask whether the
comparison there is between two feature spaces that both lack the governing
variable.

Adds PGA, PGV and MMI from the USGS ShakeMap for that event (see
data/ground_motion/PROVENANCE.md) and re-runs the comparison on the published
points, folds and classifier. Ground motion is offered to both sides -- the
manual stack and the foundation-model embeddings -- so the test cannot be
dismissed as handing one competitor a variable the other was denied.

The expected direction is worth stating in advance: supplying a missing control
should make the baseline stronger and therefore the foundation-model deficit
larger, not smaller. The arm is informative either way, but only if the shaking
field actually varies across the basin, which it does (PGA 7.8-51.6 %g).
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np
import rasterio
from scipy.stats import t as student_t
from sklearn.model_selection import GroupKFold

from config import RESULTS, ROOT
from point_probes import build_dataset, rf_fold_metrics
from terramind_linprobe import (
    N_FOLDS, PRITHVI_NAME, TERRAMIND_NAME, embedding_cache_path,
    embedding_fingerprint,
)

GM_DIR = ROOT / "data" / "ground_motion"
GM_LAYERS = ["pga", "pgv", "mmi"]
BASIN = "11_rio_maule"


def tci(v):
    n = len(v)
    m = sum(v) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1))
    h = float(student_t.ppf(0.975, n - 1)) * sd / math.sqrt(n)
    return m, m - h, m + h, not (m - h <= 0 <= m + h)


def fmt(v):
    m, lo, hi, s = tci(v)
    return f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]{'*' if s else ''}"


def sample_gm(rows, cols):
    out = []
    for name in GM_LAYERS:
        with rasterio.open(GM_DIR / f"maule_{name}_30m.tif") as src:
            band = src.read(1)
            out.append(band[rows, cols].astype(np.float32))
    return np.stack(out, axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=N_FOLDS)
    a = ap.parse_args()

    rows, cols, y, X17, blk = build_dataset(BASIN, negatives="uniform")
    gm = sample_gm(rows, cols)
    print(f"[gm] sampled {gm.shape[1]} layers at {len(rows)} points; "
          f"PGA {gm[:, 0].min():.1f}-{gm[:, 0].max():.1f} %g")
    print(f"[gm] PGA at positives {gm[y == 1, 0].mean():.2f} vs "
          f"negatives {gm[y == 0, 0].mean():.2f} %g")

    cv = list(GroupKFold(n_splits=a.folds).split(X17, y, groups=blk))
    fp = embedding_fingerprint(PRITHVI_NAME, "pretrained", ["S2L2A"], rows, cols, y)

    sets = {"A": X17, "A+GM": np.hstack([X17, gm])}
    for label, enc, mods in [("Prithvi", PRITHVI_NAME, ["S2L2A"]),
                             ("TM+DEM", TERRAMIND_NAME, ["DEM"])]:
        slug = "terramind" if enc == TERRAMIND_NAME else enc
        path = embedding_cache_path(BASIN, slug, "pretrained", mods)
        d = np.load(path, allow_pickle=False)
        want = embedding_fingerprint(enc, "pretrained", mods, rows, cols, y)
        if str(d["fingerprint"]) != want:
            raise SystemExit(f"{label}: cached embeddings are for other points")
        e = d["embeddings"].astype(np.float32)
        sets[label] = e
        sets[f"{label}+GM"] = np.hstack([e, gm])

    res = {}
    for k, X in sets.items():
        roc, pr = rf_fold_metrics(X, y, cv)
        res[k] = {"roc": roc, "pr": pr}
        print(f"  {k:12s} ROC={np.mean(roc):.3f}  PR={np.mean(pr):.3f}")

    def delta(a_, b_):
        return [x - z for x, z in zip(res[a_]["roc"], res[b_]["roc"])]

    print("\n--- does ground motion improve the manual baseline? ---")
    print(f"  A+GM vs A             : {fmt(delta('A+GM', 'A'))}")
    print("\n--- foundation models against the published baseline ---")
    for m in ["Prithvi", "TM+DEM"]:
        print(f"  {m:8s} vs A          : {fmt(delta(m, 'A'))}")
    print("\n--- foundation models against the ground-motion-aware baseline ---")
    for m in ["Prithvi", "TM+DEM"]:
        print(f"  {m:8s} vs A+GM       : {fmt(delta(m, 'A+GM'))}")
    print("\n--- ground motion offered to the foundation models too ---")
    for m in ["Prithvi", "TM+DEM"]:
        print(f"  {m}+GM vs A+GM   : {fmt(delta(f'{m}+GM', 'A+GM'))}")
        print(f"  {m}+GM vs {m}  : {fmt(delta(f'{m}+GM', m))}")

    out = {"basin": BASIN, "fingerprint": fp, "n_folds": a.folds,
           "gm_layers": GM_LAYERS,
           "pga_positives_mean": float(gm[y == 1, 0].mean()),
           "pga_negatives_mean": float(gm[y == 0, 0].mean()),
           "sets": {k: v for k, v in res.items()},
           "deltas": {f"{x}_vs_{z}": {"mean": tci(delta(x, z))[0],
                                      "ci": list(tci(delta(x, z))[1:3]),
                                      "significant": tci(delta(x, z))[3]}
                      for x, z in [("A+GM", "A"), ("Prithvi", "A"), ("TM+DEM", "A"),
                                   ("Prithvi", "A+GM"), ("TM+DEM", "A+GM"),
                                   ("Prithvi+GM", "A+GM"), ("TM+DEM+GM", "A+GM"),
                                   ("Prithvi+GM", "Prithvi"), ("TM+DEM+GM", "TM+DEM")]}}
    dst = RESULTS / f"{BASIN}_ground_motion_arm.json"
    dst.write_text(json.dumps(out, indent=2))
    print(f"\n[done] wrote {dst.name}")


if __name__ == "__main__":
    main()
