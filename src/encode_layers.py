"""Encode every Prithvi block in one forward pass (GPU host).

The benchmark probes only the final block: forward_features() already returns
all 24, and the pipeline discards 23 of them. Last-layer mean pooling is a weak
default for spatially structured tasks, and reporting no layer search is a fair
criticism of the study -- so this script keeps them all. Because they come from
the same forward pass, a full 24-layer sweep costs exactly one encoding run.

Runs standalone on a host that has torch, timm, einops and the Prithvi weights,
but none of the basin rasters: it consumes the memmap written by
export_patches.py, so the point set cannot drift from the published one. The
fingerprint travels with the patches and is copied into the output for the
evaluation step to verify.

Output: {basin}_prithvi_layers.npz with an (n_layers, n, 1024) float32 array of
mean-pooled embeddings (CLS dropped), matching _encode_prithvi's pooling exactly.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

PRITHVI_EMBED_DIM = 1024

# The published embeddings were computed in full fp32 on CPU. PyTorch enables
# TF32 matmuls by default on Ampere and later, which keeps only 10 mantissa bits
# and shifts the last layer by ~1e-2 per dimension -- enough that the sweep would
# no longer reproduce the benchmark cell it is supposed to extend. Disabling it
# costs some speed and buys an exact-in-fp32 comparison.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def build_prithvi(prithvi_path: str):
    """Same construction path as terramind_linprobe.build_encoder()."""
    p = Path(prithvi_path)
    sys.path.insert(0, str(p))
    from prithvi_mae import PrithviMAE  # type: ignore

    cfg = json.loads((p / "config.json").read_text())["pretrained_cfg"]
    cfg["num_frames"] = 1
    norm = {
        "mean": np.array(cfg["mean"], dtype=np.float32)[None, :, None, None],
        "std": np.array(cfg["std"], dtype=np.float32)[None, :, None, None],
    }
    for k in ("mean", "std"):
        cfg.pop(k, None)
    model = PrithviMAE(**cfg)
    ckpt = torch.load(p / "Prithvi_EO_V2_300M.pt", map_location="cpu",
                      weights_only=True)
    for k in list(ckpt.keys()):
        if "pos_embed" in k:
            del ckpt[k]
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    print(f"[encoder] prithvi loaded; missing={len(missing)} unexpected={len(unexpected)}")
    return model, norm


def encode(basin: str, patch_dir: Path, out_dir: Path, prithvi_path: str,
           batch_size: int, device: str, limit: int = 0) -> None:
    meta = np.load(patch_dir / f"{basin}_prithvi_patches.npz", allow_pickle=False)
    patches = np.load(patch_dir / f"{basin}_prithvi_patches.u16", mmap_mode="r")
    n = patches.shape[0]
    if limit:
        # Smoke test: exercise the whole path (weights, GPU, pooling, output
        # shape) on a handful of points. Written to a separate file so a probe
        # run can never be mistaken for, or overwrite, a real one.
        n = min(limit, n)
        print(f"[smoke] limiting to {n} points")
    if tuple(meta["shape"]) != patches.shape:
        raise SystemExit(f"shape mismatch: {tuple(meta['shape'])} vs {patches.shape}")

    model, norm = build_prithvi(prithvi_path)
    model.eval().to(device)
    mean = torch.from_numpy(norm["mean"]).to(device)
    std = torch.from_numpy(norm["std"]).to(device)

    out = None
    t0 = perf_counter()
    with torch.no_grad():
        for i in range(0, n, batch_size):
            sl = slice(i, min(i + batch_size, n))
            # uint16 -> float32 on device, then the model's own normalisation
            x = torch.from_numpy(np.ascontiguousarray(patches[sl])).to(device)
            x = x.to(torch.float32)
            # BOA_ADD_OFFSET removed from valid pixels (PRITHVI_INPUT in
            # terramind_linprobe.py): Prithvi is pretrained on offset-free HLS.
            x = torch.where((x > 0).any(dim=1, keepdim=True), x - 1000.0, x)
            x = (x - mean) / std
            feats = model.forward_features(x)
            if not isinstance(feats, (list, tuple)):
                feats = [feats]
            if out is None:
                out = np.zeros((len(feats), n, PRITHVI_EMBED_DIM), dtype=np.float32)
                print(f"[encode] {len(feats)} layers x {n} points x {PRITHVI_EMBED_DIM} dims")
            for li, f in enumerate(feats):
                # drop CLS, mean-pool spatial tokens -- identical to the benchmark
                out[li, sl] = f[:, 1:, :].mean(dim=1).to(torch.float32).cpu().numpy()
            del feats, x
            if (i // batch_size) % 5 == 0:
                done = min(i + batch_size, n)
                rate = done / max(perf_counter() - t0, 1e-9)
                print(f"  [encode] {done}/{n}  {rate:.1f} patches/s  "
                      f"elapsed={perf_counter() - t0:.0f}s", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_smoke{n}" if limit else ""
    dst = out_dir / f"{basin}_prithvi_layers{suffix}.npz"
    np.savez_compressed(
        dst, embeddings=out,
        fingerprint=meta["fingerprint"] if not limit else np.array("SMOKE-PARTIAL"),
        y=meta["y"][:n], rows=meta["rows"][:n], cols=meta["cols"][:n],
        basin=np.array(basin),
    )
    print(f"[done] {basin}: {out.shape} -> {dst.name} "
          f"({dst.stat().st_size / 1e6:.0f} MB) in {perf_counter() - t0:.0f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--basin", action="append", required=True)
    ap.add_argument("--patch-dir", default="~/geofm/patches")
    ap.add_argument("--out-dir", default="~/geofm/layers")
    ap.add_argument("--prithvi-path", default="~/models/prithvi-300m")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=0,
                    help="encode only the first N points (smoke test)")
    a = ap.parse_args()
    for b in a.basin:
        encode(b, Path(a.patch_dir).expanduser(), Path(a.out_dir).expanduser(),
               str(Path(a.prithvi_path).expanduser()), a.batch_size, a.device,
               a.limit)


if __name__ == "__main__":
    main()
