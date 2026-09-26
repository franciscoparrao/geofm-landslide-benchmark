"""Export the benchmark's evaluated point sets to CSV for the public deposit.

The point sets published alongside the paper were first written by hand, before
the negative sampler was corrected for the edge artefact of Section 2.5. The
uniform files therefore carried the pre-fix bounding-box draw -- realized class
ratios of 2.2:1 to 2.9:1 instead of the intended 5:1, because roughly half the
sampled negatives fell outside the basin and were then lost to NoData in the
feature stack. The constrained files were unaffected: that branch already
required DEM != 0 and so never saw the artefact.

Regenerating them by hand a second time would invite the same drift, so the
export now runs from build_dataset() itself. Whatever the benchmark evaluates is
what lands in the CSV, by construction.

Columns: row, col, x, y_coord (raster CRS), label, spatial_block_id.

Positives that fall in the same 30 m cell are kept as separate rows. They are
distinct catalogued events and the benchmark counts them as such; deduplicating
here would make the published file disagree with the n_+ of Table 1.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import rasterio

from config import BASINS, RESULTS, basin_dir
from point_probes import build_dataset

BENCH_BASINS = ["06_rio_huasco", "09_rio_maipo", "11_rio_maule"]


def export(basin: str, negatives: str, out_dir: Path) -> None:
    rows, cols, y, _feats, block_id = build_dataset(basin, negatives=negatives)
    with rasterio.open(basin_dir(basin) / "dem_30m.tif") as src:
        transform = src.transform
    xs, ys = rasterio.transform.xy(transform, rows, cols, offset="center")

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{basin}_points_{negatives}.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "col", "x", "y_coord", "label", "spatial_block_id"])
        for r, c, x, yc, lab, blk in zip(rows, cols, xs, ys, y, block_id):
            w.writerow([int(r), int(c), f"{x:.1f}", f"{yc:.1f}", int(lab), int(blk)])

    n_pos = int(y.sum())
    n_neg = int((y == 0).sum())
    keys = rows.astype(np.int64) * (10 ** 6) + cols
    n_dup = len(keys) - len(np.unique(keys))
    print(f"[export] {path.name}: pos={n_pos} neg={n_neg} "
          f"ratio={n_neg / n_pos:.2f}:1 coincident_cells={n_dup} "
          f"blocks={len(np.unique(block_id))}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--basin", action="append", choices=BASINS,
                    help="repeatable; default is the three benchmarked basins")
    ap.add_argument("--negatives", action="append",
                    choices=["uniform", "constrained"])
    ap.add_argument("--out", default=str(RESULTS / "benchmark_points"))
    args = ap.parse_args()
    for basin in (args.basin or BENCH_BASINS):
        for neg in (args.negatives or ["uniform", "constrained"]):
            export(basin, neg, Path(args.out))


if __name__ == "__main__":
    main()
