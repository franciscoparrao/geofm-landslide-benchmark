"""Fase 2 — Geospatial-foundation-model linear probe on susceptibility.

For one basin: build dataset (positives from SERNAGEOMIN inventory, negatives
sampled with buffer), extract 224x224 patches centered on each point, encode
with a chosen backbone (TerraMind v1-tiny by default, or Prithvi-EO-2.0-300M),
mean-pool last-layer tokens to obtain an embedding per point, train RF, report
AUC ROC / PR with bootstrap CI.

Same protocol as src/susceptibility_h33.py to make Δ-AUC comparable.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from time import perf_counter

warnings.filterwarnings("ignore")

import numpy as np
import pyproj
import rasterio
import torch
from rasterio.windows import Window
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

from config import BASINS, DEFAULT_BASIN, RESULTS, basin_dir

INVENTORY_BASE = Path(
    "/mnt/kingston/proyectos/postdoc/papers/paper1_susceptibilidad/basin_inventory"
)
ML_DATASET_BASE = Path(
    "/mnt/kingston/proyectos/postdoc/papers/paper1_susceptibilidad/ml_dataset"
)
S2_COMPOSITE_BASE = Path(
    "/home/franciscoparrao/proyectos/no_supervisado_superficie/paper/data/s2_composites"
)
PATCH_SIZE = 224
NEG_RATIO = 5
BUFFER_PX = PATCH_SIZE // 2  # 112 px = 3360 m, prevents patch-content leak
N_FOLDS = 5
N_BOOTSTRAP = 1000
N_TREES = 300
SEED = 42
BATCH_SIZE = 16
SPATIAL_BLOCK_PX = 333  # 333 * 30 m ≈ 10 km blocks for spatial holdout
S2_BAND_COUNT = 12   # composite has 12 S2L2A bands in PRETRAINED_BANDS order

# Encoder choices and their characteristics.
TERRAMIND_NAME = "terramind_v1_tiny"
TERRAMIND_EMBED_DIM = 192

PRITHVI_NAME = "prithvi-300m"
PRITHVI_EMBED_DIM = 1024
DEFAULT_PRITHVI_PATH = "/root/models/prithvi-300m"

# HLS band order Prithvi was pretrained on: Blue, Green, Red, NIR-narrow, SWIR1, SWIR2.
# Mapping from our 12-band S2L2A composite (B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B11,B12):
#   HLS B02 (Blue)        <- S2 B02  (composite index 1)
#   HLS B03 (Green)       <- S2 B03  (composite index 2)
#   HLS B04 (Red)         <- S2 B04  (composite index 3)
#   HLS B05 (NIR narrow)  <- S2 B8A  (composite index 8)
#   HLS B06 (SWIR1)       <- S2 B11  (composite index 10)
#   HLS B07 (SWIR2)       <- S2 B12  (composite index 11)
PRITHVI_HLS_INDICES = [1, 2, 3, 8, 10, 11]


def load_events_xy(basin, raster_crs, raster_transform):
    """Return (rows, cols) of positive events in raster pixel coordinates.

    Prefers ml_dataset/{basin}.csv (positives + negatives pre-processed in UTM
    by the project's pipeline; we use only positives with label==1). Falls back
    to basin_inventory/{basin}.csv (raw SERNAGEOMIN catastro in lat/lon) if the
    ml_dataset file is missing. Uses csv.reader to handle quoted strings.
    """
    import csv
    ml_path = ML_DATASET_BASE / f"{basin}.csv"
    if ml_path.exists():
        xs, ys = [], []
        with open(ml_path) as f:
            reader = csv.DictReader(f)
            for r in reader:
                if int(r.get("label", "0")) != 1:
                    continue
                xs.append(float(r["x"])); ys.append(float(r["y"]))
        xs_arr = np.array(xs); ys_arr = np.array(ys)
        print(f"[load_events] using ml_dataset/{basin}.csv n_positives={len(xs)}")
    else:
        csv_path = INVENTORY_BASE / f"{basin}.csv"
        lats, lons = [], []
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for r in reader:
                lats.append(float(r["lat"])); lons.append(float(r["lon"]))
        transformer = pyproj.Transformer.from_crs(
            "EPSG:4326", raster_crs, always_xy=True
        )
        xs_arr, ys_arr = transformer.transform(lons, lats)
        xs_arr = np.array(xs_arr); ys_arr = np.array(ys_arr)
        print(f"[load_events] using basin_inventory/{basin}.csv n_positives={len(lats)}")
    inv = ~raster_transform
    cols, rrows = inv * (xs_arr, ys_arr)
    return np.array(rrows, dtype=np.int64), np.array(cols, dtype=np.int64)


def extract_patches(dem_src, rows, cols, size=PATCH_SIZE):
    """Extract size×size DEM patches centered at each (row, col)."""
    half = size // 2
    out = np.zeros((len(rows), size, size), dtype=np.float32)
    for i, (r, c) in enumerate(zip(rows, cols)):
        win = Window(c - half, r - half, size, size)
        arr = dem_src.read(1, window=win, boundless=True, fill_value=0.0)
        if arr.shape != (size, size):
            pad_r = size - arr.shape[0]
            pad_c = size - arr.shape[1]
            arr = np.pad(arr, ((0, max(0, pad_r)), (0, max(0, pad_c))),
                         constant_values=0.0)[:size, :size]
        out[i] = arr
    out = np.where(np.isfinite(out), out, 0.0)
    return out


def extract_patches_s2(s2_src, rows, cols, size=PATCH_SIZE,
                      band_indices=None, scale=10000.0):
    """Extract size×size Sentinel-2 multi-band patches centered at each (row, col).

    Args:
        band_indices: list of 0-based band indices to keep. None = all bands.
        scale: divisor applied to raw values. 10000.0 yields reflectance in
            [0, 1]; 1.0 keeps the raw uint16 scale that Prithvi expects (with
            its own mean/std normalization downstream).
    Returns float32 array of shape (n, n_bands_selected, size, size).
    """
    half = size // 2
    n_bands_src = s2_src.count
    sel = list(range(n_bands_src)) if band_indices is None else list(band_indices)
    out = np.zeros((len(rows), len(sel), size, size), dtype=np.float32)
    for i, (r, c) in enumerate(zip(rows, cols)):
        win = Window(c - half, r - half, size, size)
        arr = s2_src.read(window=win, boundless=True, fill_value=0)
        if arr.shape[1:] != (size, size):
            pad_r = size - arr.shape[1]
            pad_c = size - arr.shape[2]
            arr = np.pad(
                arr,
                ((0, 0), (0, max(0, pad_r)), (0, max(0, pad_c))),
                constant_values=0,
            )[:, :size, :size]
        out[i] = arr[sel].astype(np.float32) / scale
    out = np.where(np.isfinite(out), out, 0.0)
    return out


def build_encoder(name, pretrained, modalities, prithvi_path=None):
    """Build a backbone and return (model, kind, embed_dim, norm_stats).

    kind is "terramind" or "prithvi". For Prithvi, norm_stats is a dict with
    'mean' and 'std' (per-band, raw-reflectance scale) used at encode time.
    For TerraMind, norm_stats is None (the model handles normalization).
    """
    if name == TERRAMIND_NAME:
        from terratorch import BACKBONE_REGISTRY
        model = BACKBONE_REGISTRY.build(
            name, pretrained=pretrained, modalities=modalities,
        )
        return model, "terramind", TERRAMIND_EMBED_DIM, None

    if name == PRITHVI_NAME:
        if prithvi_path is None:
            prithvi_path = DEFAULT_PRITHVI_PATH
        import sys
        sys.path.insert(0, prithvi_path)
        from prithvi_mae import PrithviMAE  # type: ignore
        cfg = json.loads((Path(prithvi_path) / "config.json").read_text())["pretrained_cfg"]
        cfg["num_frames"] = 1  # single-timestep inference
        norm_stats = {
            "mean": np.array(cfg["mean"], dtype=np.float32)[None, :, None, None],
            "std":  np.array(cfg["std"],  dtype=np.float32)[None, :, None, None],
        }
        model = PrithviMAE(**cfg)
        if pretrained:
            ckpt = torch.load(
                Path(prithvi_path) / "Prithvi_EO_V2_300M.pt",
                map_location="cpu", weights_only=True,
            )
            for k in list(ckpt.keys()):
                if "pos_embed" in k:
                    del ckpt[k]
            missing, unexpected = model.load_state_dict(ckpt, strict=False)
            print(f"[encoder] prithvi pretrained loaded; missing={len(missing)} unexpected={len(unexpected)}")
        return model, "prithvi", PRITHVI_EMBED_DIM, norm_stats

    raise ValueError(f"Unknown encoder: {name}")


def _encode_terramind(model, dem_patches, s2_patches, embed_dim, batch_size):
    n = dem_patches.shape[0]
    embeddings = np.zeros((n, embed_dim), dtype=np.float32)
    use_s2 = s2_patches is not None
    t0 = perf_counter()
    for i in range(0, n, batch_size):
        batch_dem = dem_patches[i:i + batch_size]
        inputs = {"DEM": torch.from_numpy(batch_dem[:, None, :, :])}
        if use_s2:
            inputs["S2L2A"] = torch.from_numpy(s2_patches[i:i + batch_size])
        with torch.no_grad():
            out = model(inputs)
        last = out if isinstance(out, torch.Tensor) else out[-1]
        embeddings[i:i + batch_size] = last.mean(dim=1).cpu().numpy().astype(np.float32)
        if (i // batch_size) % 5 == 0:
            print(f"  [encode] {i + batch_dem.shape[0]}/{n}  elapsed={perf_counter() - t0:.1f}s")
    return embeddings


def _encode_prithvi(model, s2_hls_patches, embed_dim, batch_size,
                    norm_stats, use_bf16=True):
    """Encode raw-reflectance S2 HLS-equivalent patches with Prithvi.

    Mean-pools the spatial tokens of the final encoder layer (drops CLS).
    Uses bf16 by default for ~2.5x speedup on CPU.
    """
    n = s2_hls_patches.shape[0]
    embeddings = np.zeros((n, embed_dim), dtype=np.float32)
    mean = norm_stats["mean"]; std = norm_stats["std"]
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    model = model.to(dtype)
    t0 = perf_counter()
    for i in range(0, n, batch_size):
        batch = (s2_hls_patches[i:i + batch_size] - mean) / std
        x = torch.from_numpy(batch).to(dtype)
        with torch.no_grad():
            feats = model.forward_features(x)
        last = feats[-1] if isinstance(feats, (list, tuple)) else feats
        # drop CLS token (index 0), mean-pool spatial tokens
        emb = last[:, 1:, :].mean(dim=1).to(torch.float32).cpu().numpy()
        embeddings[i:i + batch_size] = emb.astype(np.float32)
        if (i // batch_size) % 5 == 0:
            print(f"  [encode] {i + batch.shape[0]}/{n}  elapsed={perf_counter() - t0:.1f}s")
    return embeddings


def encode_patches(kind, model, embed_dim, dem_patches=None,
                   s2_patches=None, s2_hls_patches=None,
                   norm_stats=None, batch_size=BATCH_SIZE):
    """Dispatch to the encoder-specific implementation."""
    if kind == "terramind":
        return _encode_terramind(model, dem_patches, s2_patches, embed_dim, batch_size)
    if kind == "prithvi":
        assert s2_hls_patches is not None and norm_stats is not None
        return _encode_prithvi(model, s2_hls_patches, embed_dim, batch_size, norm_stats)
    raise ValueError(f"Unknown encoder kind: {kind}")


def auc_with_paired_bootstrap(y, sA, sB, rng, n=N_BOOTSTRAP):
    n_obs = y.size
    boot_deltas_roc = np.empty(n)
    boot_deltas_pr = np.empty(n)
    for b in range(n):
        ix = rng.integers(0, n_obs, size=n_obs)
        if y[ix].sum() == 0 or y[ix].sum() == n_obs:
            boot_deltas_roc[b] = np.nan; boot_deltas_pr[b] = np.nan; continue
        boot_deltas_roc[b] = (
            roc_auc_score(y[ix], sB[ix]) - roc_auc_score(y[ix], sA[ix])
        )
        boot_deltas_pr[b] = (
            average_precision_score(y[ix], sB[ix])
            - average_precision_score(y[ix], sA[ix])
        )
    boot_deltas_roc = boot_deltas_roc[~np.isnan(boot_deltas_roc)]
    boot_deltas_pr = boot_deltas_pr[~np.isnan(boot_deltas_pr)]
    return {
        "delta_roc_mean": float(boot_deltas_roc.mean()),
        "delta_roc_ci_95": [float(np.percentile(boot_deltas_roc, 2.5)),
                            float(np.percentile(boot_deltas_roc, 97.5))],
        "delta_pr_mean": float(boot_deltas_pr.mean()),
        "delta_pr_ci_95": [float(np.percentile(boot_deltas_pr, 2.5)),
                           float(np.percentile(boot_deltas_pr, 97.5))],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--basin", default=DEFAULT_BASIN, choices=BASINS)
    parser.add_argument("--cv", choices=("stratified", "spatial"),
                        default="stratified")
    parser.add_argument("--init", choices=("pretrained", "random"),
                        default="pretrained")
    parser.add_argument(
        "--encoder", choices=(TERRAMIND_NAME, PRITHVI_NAME),
        default=TERRAMIND_NAME,
        help="Backbone to evaluate. Default: TerraMind v1-tiny.",
    )
    parser.add_argument(
        "--prithvi-path", default=DEFAULT_PRITHVI_PATH,
        help="Path to the local Prithvi-EO-2.0-300M model directory.",
    )
    parser.add_argument(
        "--modalities", nargs="+", default=None,
        choices=["DEM", "S2L2A"],
        help="Input modalities. Default: DEM for TerraMind, S2L2A for Prithvi. "
             "Prithvi only accepts S2L2A.",
    )
    args = parser.parse_args()
    basin = args.basin
    cv_mode = args.cv
    init_mode = args.init
    encoder_name = args.encoder

    if args.modalities is None:
        modalities = ["S2L2A"] if encoder_name == PRITHVI_NAME else ["DEM"]
    else:
        modalities = list(args.modalities)
    if encoder_name == PRITHVI_NAME and modalities != ["S2L2A"]:
        raise SystemExit(
            "Prithvi-EO-2.0 was pretrained on HLS only; "
            "use --modalities S2L2A (DEM is not natively supported)."
        )

    use_s2 = "S2L2A" in modalities
    if use_s2:
        s2_path = S2_COMPOSITE_BASE / f"{basin}_s2l2a_2023.tif"
        if not s2_path.exists():
            raise SystemExit(f"S2 composite not found: {s2_path}. "
                             "Run paper/scripts/download_s2_composite.py first.")

    rng = np.random.default_rng(SEED)

    dem_path = basin_dir(basin) / "dem_30m.tif"
    dem_src = rasterio.open(dem_path)
    height, width = dem_src.shape
    transform = dem_src.transform
    crs = dem_src.crs
    print(f"[fase2.0] basin={basin} DEM shape={dem_src.shape}")

    pos_rows, pos_cols = load_events_xy(basin, crs, transform)
    in_bounds = (
        (pos_rows >= PATCH_SIZE // 2) & (pos_rows < height - PATCH_SIZE // 2)
        & (pos_cols >= PATCH_SIZE // 2) & (pos_cols < width - PATCH_SIZE // 2)
    )
    pos_rows = pos_rows[in_bounds]; pos_cols = pos_cols[in_bounds]
    n_pos = pos_rows.size
    print(f"[fase2.0] positives_valid_in_bounds={n_pos}")
    if n_pos < 30:
        raise SystemExit("too few positive events for modeling")

    excluded = np.zeros(height * width, dtype=bool)
    for r, c in zip(pos_rows, pos_cols):
        for dr in range(-BUFFER_PX, BUFFER_PX + 1):
            for dc in range(-BUFFER_PX, BUFFER_PX + 1):
                r2 = r + dr; c2 = c + dc
                if 0 <= r2 < height and 0 <= c2 < width:
                    excluded[r2 * width + c2] = True

    n_neg_target = NEG_RATIO * n_pos
    half = PATCH_SIZE // 2
    neg_rows = []; neg_cols = []
    while len(neg_rows) < n_neg_target:
        r = int(rng.integers(half, height - half))
        c = int(rng.integers(half, width - half))
        if not excluded[r * width + c]:
            neg_rows.append(r); neg_cols.append(c)
    neg_rows = np.array(neg_rows); neg_cols = np.array(neg_cols)
    n_neg = neg_rows.size
    print(f"[fase2.0] negatives_sampled={n_neg}")

    all_rows = np.concatenate([pos_rows, neg_rows])
    all_cols = np.concatenate([pos_cols, neg_cols])
    y = np.concatenate([np.ones(n_pos), np.zeros(n_neg)]).astype(np.int8)

    stack_path = RESULTS / f"{basin}_stack.npz"
    print(f"[fase2.0] loading pixel features from {stack_path.name}")
    with np.load(stack_path, allow_pickle=False) as npz:
        X_all = npz["X"].astype(np.float32)
        valid_idx = npz["valid_idx"]
        feature_names_pixel = list(npz["feature_names"])
    flat_to_row = {int(idx): i for i, idx in enumerate(valid_idx)}
    flat_positions = all_rows * width + all_cols
    pixel_features = np.full(
        (len(all_rows), X_all.shape[1]), np.nan, dtype=np.float32
    )
    valid_pixel_mask = np.zeros(len(all_rows), dtype=bool)
    for i, fp in enumerate(flat_positions):
        if int(fp) in flat_to_row:
            pixel_features[i] = X_all[flat_to_row[int(fp)]]
            valid_pixel_mask[i] = True
    n_valid_pixel = int(valid_pixel_mask.sum())
    print(f"[fase2.0] pixel_features valid for {n_valid_pixel}/{len(all_rows)} points")
    if n_valid_pixel < len(all_rows):
        keep = valid_pixel_mask
        all_rows = all_rows[keep]; all_cols = all_cols[keep]
        pixel_features = pixel_features[keep]
        y = y[keep]
        n_pos = int(y.sum()); n_neg = int((y == 0).sum())
        print(f"[fase2.0] after pixel-validity filter: pos={n_pos} neg={n_neg}")

    needs_dem = encoder_name == TERRAMIND_NAME
    dem_patches = None
    if needs_dem:
        print(f"[fase2] extracting {len(all_rows)} DEM patches (size={PATCH_SIZE})")
        t0 = perf_counter()
        dem_patches = extract_patches(dem_src, all_rows, all_cols)
        print(f"[fase2] DEM patches.shape={dem_patches.shape} in {perf_counter() - t0:.1f}s")
        print(f"[fase2] elev range raw: min={dem_patches.min():.1f}m max={dem_patches.max():.1f}m mean={dem_patches.mean():.1f}m")

    s2_patches = None
    s2_hls_patches = None
    if use_s2:
        s2_src = rasterio.open(s2_path)
        if encoder_name == TERRAMIND_NAME:
            print(f"[fase2] extracting {len(all_rows)} S2L2A patches (12 bands) from {s2_path.name}")
            t0 = perf_counter()
            s2_patches = extract_patches_s2(s2_src, all_rows, all_cols,
                                            band_indices=None, scale=10000.0)
            print(f"[fase2] S2 patches.shape={s2_patches.shape} in {perf_counter() - t0:.1f}s")
            print(f"[fase2] S2 reflectance range: min={s2_patches.min():.3f} max={s2_patches.max():.3f} mean={s2_patches.mean():.3f}")
        else:  # prithvi: HLS-equivalent 6 bands, keep raw uint16 scale
            print(f"[fase2] extracting {len(all_rows)} S2 HLS-equivalent patches (6 bands) from {s2_path.name}")
            t0 = perf_counter()
            s2_hls_patches = extract_patches_s2(
                s2_src, all_rows, all_cols,
                band_indices=PRITHVI_HLS_INDICES, scale=1.0,
            )
            print(f"[fase2] S2-HLS patches.shape={s2_hls_patches.shape} in {perf_counter() - t0:.1f}s")
            print(f"[fase2] raw range: min={s2_hls_patches.min():.0f} max={s2_hls_patches.max():.0f} mean={s2_hls_patches.mean():.0f}")

    print(f"[fase2] loading {encoder_name}  init={init_mode}  modalities={modalities}")
    pretrained = init_mode == "pretrained"
    model, encoder_kind, embed_dim, norm_stats = build_encoder(
        encoder_name, pretrained=pretrained, modalities=modalities,
        prithvi_path=args.prithvi_path,
    )
    if not pretrained:
        torch.manual_seed(SEED)
        for p in model.parameters():
            if p.requires_grad and p.ndim >= 2:
                torch.nn.init.xavier_uniform_(p)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[fase2] params={n_params:,}  embed_dim={embed_dim}  kind={encoder_kind}")

    print(f"[fase2] encoding patches with batch_size={BATCH_SIZE}")
    t0 = perf_counter()
    embeddings = encode_patches(
        encoder_kind, model, embed_dim,
        dem_patches=dem_patches, s2_patches=s2_patches,
        s2_hls_patches=s2_hls_patches, norm_stats=norm_stats,
        batch_size=BATCH_SIZE,
    )
    print(f"[fase2] embeddings.shape={embeddings.shape} elapsed={perf_counter() - t0:.1f}s")

    if cv_mode == "spatial":
        n_cols_blocks = (width + SPATIAL_BLOCK_PX - 1) // SPATIAL_BLOCK_PX
        block_id = (all_rows // SPATIAL_BLOCK_PX) * n_cols_blocks + (all_cols // SPATIAL_BLOCK_PX)
        unique_blocks = np.unique(block_id)
        pos_blocks = np.unique(block_id[y == 1])
        neg_blocks = np.unique(block_id[y == 0])
        print(f"[fase2.0] spatial CV  block_size={SPATIAL_BLOCK_PX}px (~{SPATIAL_BLOCK_PX * 30 / 1000:.1f} km)")
        print(f"[fase2.0]   unique blocks: {len(unique_blocks)}  with pos: {len(pos_blocks)}  with neg: {len(neg_blocks)}")
        cv_iter = GroupKFold(n_splits=N_FOLDS).split(embeddings, y, groups=block_id)
    else:
        cv_iter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(embeddings, y)

    fold_results = []
    sA_all = np.empty(len(y), dtype=np.float64)
    sB_all = np.empty(len(y), dtype=np.float64)
    for fi, (tr, te) in enumerate(cv_iter):
        t0 = perf_counter()
        rfA = RandomForestClassifier(
            n_estimators=N_TREES, max_depth=None, min_samples_leaf=5,
            n_jobs=-1, random_state=SEED + fi, class_weight="balanced",
        ).fit(pixel_features[tr], y[tr])
        sA = rfA.predict_proba(pixel_features[te])[:, 1]
        sA_all[te] = sA
        rocA = roc_auc_score(y[te], sA); prA = average_precision_score(y[te], sA)

        rfB = RandomForestClassifier(
            n_estimators=N_TREES, max_depth=None, min_samples_leaf=5,
            n_jobs=-1, random_state=SEED + fi, class_weight="balanced",
        ).fit(embeddings[tr], y[tr])
        sB = rfB.predict_proba(embeddings[te])[:, 1]
        sB_all[te] = sB
        rocB = roc_auc_score(y[te], sB); prB = average_precision_score(y[te], sB)

        elapsed = perf_counter() - t0
        print(
            f"[fase2.0] fold={fi}  A(pix17): ROC={rocA:.3f} PR={prA:.3f}  "
            f"B(tm192): ROC={rocB:.3f} PR={prB:.3f}  "
            f"ΔROC={rocB - rocA:+.3f} ΔPR={prB - prA:+.3f}  time={elapsed:.1f}s"
        )
        fold_results.append({
            "fold": fi,
            "roc_A": float(rocA), "pr_A": float(prA),
            "roc_B": float(rocB), "pr_B": float(prB),
            "delta_roc": float(rocB - rocA), "delta_pr": float(prB - prA),
        })

    rocsA = np.array([r["roc_A"] for r in fold_results])
    prsA = np.array([r["pr_A"] for r in fold_results])
    rocsB = np.array([r["roc_B"] for r in fold_results])
    prsB = np.array([r["pr_B"] for r in fold_results])
    delta_rocs = rocsB - rocsA
    delta_prs = prsB - prsA
    print(f"\n[fase2] A (17 pixel feats):  ROC={rocsA.mean():.3f}±{rocsA.std():.3f}  PR={prsA.mean():.3f}±{prsA.std():.3f}")
    print(f"[fase2] B ({encoder_name} {embed_dim}-dim): ROC={rocsB.mean():.3f}±{rocsB.std():.3f}  PR={prsB.mean():.3f}±{prsB.std():.3f}")
    print(f"[fase2.0] Δ-ROC = {delta_rocs.mean():+.3f} ± {delta_rocs.std():.3f}")
    print(f"[fase2.0] Δ-PR  = {delta_prs.mean():+.3f} ± {delta_prs.std():.3f}")

    boot_rng = np.random.default_rng(SEED + 1)
    n = len(y)
    boot_droc = np.empty(N_BOOTSTRAP); boot_dpr = np.empty(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        ix = boot_rng.integers(0, n, size=n)
        if y[ix].sum() == 0 or y[ix].sum() == n:
            boot_droc[b] = np.nan; boot_dpr[b] = np.nan; continue
        boot_droc[b] = roc_auc_score(y[ix], sB_all[ix]) - roc_auc_score(y[ix], sA_all[ix])
        boot_dpr[b] = average_precision_score(y[ix], sB_all[ix]) - average_precision_score(y[ix], sA_all[ix])
    boot_droc = boot_droc[~np.isnan(boot_droc)]
    boot_dpr = boot_dpr[~np.isnan(boot_dpr)]
    ci_roc = (float(np.percentile(boot_droc, 2.5)), float(np.percentile(boot_droc, 97.5)))
    ci_pr = (float(np.percentile(boot_dpr, 2.5)), float(np.percentile(boot_dpr, 97.5)))
    print(f"[fase2.0] paired bootstrap N={N_BOOTSTRAP}")
    print(f"[fase2.0]   Δ-ROC 95% CI = [{ci_roc[0]:+.3f}, {ci_roc[1]:+.3f}]")
    print(f"[fase2.0]   Δ-PR  95% CI = [{ci_pr[0]:+.3f}, {ci_pr[1]:+.3f}]")

    summary = {
        "basin": basin, "model": encoder_name,
        "init_mode": init_mode, "modalities": modalities,
        "patch_size": PATCH_SIZE, "buffer_px": BUFFER_PX,
        "cv_mode": cv_mode, "spatial_block_px": SPATIAL_BLOCK_PX,
        "n_pos": int(n_pos), "n_neg": int(n_neg),
        "embedding_dim": int(embeddings.shape[1]),
        "fold_results": fold_results,
        "A_roc_mean": float(rocsA.mean()), "A_roc_std": float(rocsA.std()),
        "A_pr_mean": float(prsA.mean()), "A_pr_std": float(prsA.std()),
        "B_roc_mean": float(rocsB.mean()), "B_roc_std": float(rocsB.std()),
        "B_pr_mean": float(prsB.mean()), "B_pr_std": float(prsB.std()),
        "delta_roc_mean": float(delta_rocs.mean()),
        "delta_roc_ci_95": list(ci_roc),
        "delta_pr_mean": float(delta_prs.mean()),
        "delta_pr_ci_95": list(ci_pr),
    }
    encoder_slug = "terramind" if encoder_name == TERRAMIND_NAME else encoder_name
    cv_suffix = "spatial" if cv_mode == "spatial" else "fair"
    init_suffix = "" if init_mode == "pretrained" else "_randinit"
    if encoder_name == TERRAMIND_NAME:
        modality_suffix = "" if modalities == ["DEM"] else "_" + "+".join(sorted(modalities)).lower()
    else:
        modality_suffix = ""  # Prithvi is S2-only, no need to disambiguate
    out_json = RESULTS / f"{basin}_{encoder_slug}_linprobe_{cv_suffix}{init_suffix}{modality_suffix}.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"[fase2] wrote {out_json.name}")


if __name__ == "__main__":
    main()
