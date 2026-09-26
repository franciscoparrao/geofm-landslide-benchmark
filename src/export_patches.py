"""Export the benchmark's Prithvi input patches to a portable memmap.

The layer sweep and any fine-tuning arm run on a GPU host that does not carry
the basin rasters, the inventories or the feature stacks. Rather than replicate
the sampling logic there -- where it could silently drift from the published
point set -- this script reproduces the benchmark's points locally with the very
same build_dataset() call the paper uses, extracts the six HLS-equivalent bands
exactly as _encode_prithvi consumes them (raw uint16 scale, PRITHVI_HLS_INDICES
order), and writes:

    {basin}_prithvi_patches.u16    memmap, shape (n, 6, 224, 224), uint16
    {basin}_prithvi_patches.npz    rows, cols, y, fingerprint, shape, band order

Storing uint16 rather than float32 halves the transfer and is lossless: with
scale=1.0 the extractor returns the raw reflectance DNs unchanged, and the
model's own mean/std normalisation happens downstream on the GPU host.

The fingerprint is the same one the embedding cache uses, so the returned
embeddings can be checked against the point set they claim to describe instead
of being trusted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import rasterio

from config import BASINS, S2_COMPOSITE_BASE
from point_probes import build_dataset
from terramind_linprobe import (
    PATCH_SIZE, PRITHVI_HLS_INDICES, PRITHVI_NAME, embedding_fingerprint,
    extract_patches_s2,
)

CHUNK = 128


def export(basin: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, cols, y, _pix, _blk = build_dataset(basin, negatives="uniform")
    n = len(rows)
    fp = embedding_fingerprint(PRITHVI_NAME, "pretrained", ["S2L2A"], rows, cols, y)

    s2_path = S2_COMPOSITE_BASE / f"{basin}_s2l2a_2023.tif"
    if not s2_path.exists():
        raise SystemExit(f"missing composite: {s2_path}")

    bin_path = out_dir / f"{basin}_prithvi_patches.u16"
    shape = (n, len(PRITHVI_HLS_INDICES), PATCH_SIZE, PATCH_SIZE)
    mm = np.lib.format.open_memmap(bin_path, mode="w+", dtype=np.uint16, shape=shape)

    t0 = perf_counter()
    with rasterio.open(s2_path) as src:
        for i in range(0, n, CHUNK):
            sl = slice(i, min(i + CHUNK, n))
            patches = extract_patches_s2(
                src, rows[sl], cols[sl],
                band_indices=PRITHVI_HLS_INDICES, scale=1.0,
            )
            # scale=1.0 leaves raw DNs; clip only guards against the fill value
            # of a malformed composite, and is a no-op on well-formed input.
            np.clip(patches, 0, np.iinfo(np.uint16).max, out=patches)
            mm[sl] = patches.astype(np.uint16)
            print(f"  [export] {min(i + CHUNK, n)}/{n}  "
                  f"elapsed={perf_counter() - t0:.0f}s", flush=True)
    mm.flush()
    del mm

    np.savez(
        out_dir / f"{basin}_prithvi_patches.npz",
        rows=rows, cols=cols, y=y,
        fingerprint=np.array(fp),
        shape=np.array(shape),
        band_indices=np.array(PRITHVI_HLS_INDICES),
        basin=np.array(basin),
    )
    size_gb = bin_path.stat().st_size / 1e9
    print(f"[done] {basin}: n={n} pos={int(y.sum())} neg={int((y == 0).sum())} "
          f"-> {bin_path.name} ({size_gb:.2f} GB), fingerprint {fp[:12]}...")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--basin", action="append", choices=BASINS,
                    help="repeatable; default is the three benchmarked basins")
    ap.add_argument("--out", default="../results/_patches")
    args = ap.parse_args()
    basins = args.basin or ["06_rio_huasco", "09_rio_maipo", "11_rio_maule"]
    for b in basins:
        export(b, Path(args.out))


if __name__ == "__main__":
    main()
