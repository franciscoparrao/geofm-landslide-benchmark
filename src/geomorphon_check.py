"""Are the flow-type coordinates source areas or deposits?

The movement-type control showed flow entries sitting at lower slopes than
randomly drawn in-basin negatives, which is the signature one would expect of
deposit or channel positions rather than scarps. Mean slope is a blunt
instrument for that question, though: it measures steepness, not position in the
landscape, and Andean debris flows frequently initiate inside the channel by bed
mobilisation rather than by a discrete slope failure, which legitimately places a
source coordinate on gentle ground.

Two sharper diagnostics are used here.

Geomorphons (Jasiewicz and Stepinski, 2013) classify each cell by the ternary
pattern of its line-of-sight neighbourhood, so they encode landform position
directly. Classes 3-7 (ridge, shoulder, spur, slope, hollow) are hillslope
positions where failure initiates; 1, 8, 9 and 10 (flat, footslope, valley, pit)
are transport and depositional positions.

Position in the drainage network separates the two readings that geomorphons
alone cannot: a channel-initiated debris flow starts high in the network, on a
small contributing area, while its deposit lies low in it on a large one. We
therefore also report contributing area and height above nearest drainage at the
positives against the negatives, per movement type.

Neither diagnostic can prove what the catalogue intended to record. Together they
can say whether the coordinates are consistent with initiation, which is what the
manuscript needs to state honestly.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter

import numpy as np
import rasterio

from config import ML_DATASET_BASE, RESULTS, basin_dir, feature_paths
from point_probes import build_dataset
from typology_arm import aligned_types

GEOMORPHON = {1: "flat", 2: "peak", 3: "ridge", 4: "shoulder", 5: "spur",
              6: "slope", 7: "hollow", 8: "footslope", 9: "valley", 10: "pit"}
HILLSLOPE = {3, 4, 5, 6, 7}          # positions where failure initiates
DEPOSITIONAL = {1, 8, 9, 10}         # transport and deposition

BASINS = [("06_rio_huasco", "Huasco"), ("09_rio_maipo", "Maipo"),
          ("11_rio_maule", "Maule")]


def sample(path, rows, cols):
    with rasterio.open(path) as src:
        return src.read(1)[rows, cols]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-n", type=int, default=15,
                    help="skip movement types with fewer positives than this")
    a = ap.parse_args()
    report = {}

    for slug, name in BASINS:
        rows, cols, y, X17, _ = build_dataset(slug, negatives="uniform")
        t = aligned_types(slug, int(y.sum()))
        gm = sample(basin_dir(slug) / "terrain" / "geomorphons.tif", rows, cols)
        fp = feature_paths(slug)
        acc = sample(fp["log1p_flow_accumulation_mfd"], rows, cols)
        hand = sample(fp["hand"], rows, cols)
        slope = sample(fp["slope"], rows, cols)
        pos = y == 1

        print(f"\n=== {name} ===")
        print(f"{'group':16s} {'n':>5s} {'hillslope%':>11s} {'deposit%':>9s} "
              f"{'slope':>6s} {'log1p acc':>10s} {'HAND':>7s}   dominant classes")
        rep = {}

        def describe(mask, label):
            g = gm[mask]
            n = int(mask.sum())
            if n == 0:
                return
            hs = 100 * np.isin(g, list(HILLSLOPE)).mean()
            dp = 100 * np.isin(g, list(DEPOSITIONAL)).mean()
            top = Counter(GEOMORPHON.get(int(v), str(v)) for v in g).most_common(3)
            print(f"{label:16s} {n:5d} {hs:10.1f}% {dp:8.1f}% "
                  f"{slope[mask].mean():6.1f} {acc[mask].mean():10.2f} "
                  f"{hand[mask].mean():7.1f}   "
                  + ", ".join(f"{k} {100*v/n:.0f}%" for k, v in top))
            rep[label] = {"n": n, "hillslope_pct": float(hs),
                          "depositional_pct": float(dp),
                          "slope_mean": float(slope[mask].mean()),
                          "log1p_acc_mean": float(acc[mask].mean()),
                          "hand_mean": float(hand[mask].mean()),
                          "top_classes": top}

        for typ, cnt in Counter(t).most_common():
            if cnt < a.min_n:
                continue
            m = pos.copy()
            m[pos] = (t == typ)
            describe(m, typ)
        describe(pos, "ALL positives")
        describe(~pos, "negatives")
        report[slug] = {"basin": name, "groups": rep}

    dst = RESULTS / "geomorphon_check.json"
    dst.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n[done] wrote {dst.name}")


if __name__ == "__main__":
    main()
