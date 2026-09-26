"""Evaluate every Prithvi block against the manual baseline (local host).

Consumes the per-layer embeddings produced by encode_layers.py on the GPU host
and scores each one with the folds, classifier and baseline of the published
benchmark, so the sweep is directly comparable to Table 3 rather than to a
re-derived analysis.

The embeddings arrive from another machine, so the first thing this does is
recompute the sampling fingerprint locally and refuse to proceed unless it
matches the one that travelled with them: that is what rules out having encoded
a different point set. Layer -1 reproduces the published cell, which is a second,
independent check that the remote path agrees with the local one.

Writes {basin}_prithvi_layer_sweep.json with per-layer fold metrics.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.stats import t as student_t
from sklearn.model_selection import GroupKFold

from config import RESULTS
from point_probes import build_dataset, rf_fold_metrics
from terramind_linprobe import N_FOLDS, PRITHVI_NAME, embedding_fingerprint


def fold_tci(values):
    n = len(values)
    m = sum(values) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in values) / (n - 1))
    h = float(student_t.ppf(0.975, n - 1)) * sd / math.sqrt(n)
    return m, m - h, m + h, not (m - h <= 0 <= m + h)


def evaluate(basin: str, layer_dir: Path, folds: int) -> dict:
    d = np.load(layer_dir / f"{basin}_prithvi_layers.npz", allow_pickle=False)
    emb = d["embeddings"]                      # (n_layers, n, 1024)
    n_layers = emb.shape[0]

    rows, cols, y, X17, block_id = build_dataset(basin, negatives="uniform")
    fp_local = embedding_fingerprint(PRITHVI_NAME, "pretrained", ["S2L2A"],
                                     rows, cols, y)
    fp_remote = str(d["fingerprint"])
    if fp_local != fp_remote:
        raise SystemExit(
            f"fingerprint mismatch for {basin}: the embeddings were computed on a "
            f"different point set (local {fp_local[:12]}, remote {fp_remote[:12]}). "
            "Re-export the patches."
        )
    if emb.shape[1] != len(y):
        raise SystemExit(f"{basin}: {emb.shape[1]} embeddings vs {len(y)} points")
    print(f"[{basin}] fingerprint OK ({fp_local[:12]}), {n_layers} layers, n={len(y)}")

    cv = list(GroupKFold(n_splits=folds).split(X17, y, groups=block_id))
    a_roc, a_pr = rf_fold_metrics(X17, y, cv)
    print(f"[{basin}] baseline A: ROC={np.mean(a_roc):.3f} PR={np.mean(a_pr):.3f}")

    out = {"basin": basin, "n_folds": folds, "n_layers": n_layers,
           "fingerprint": fp_local,
           "A_roc": a_roc, "A_pr": a_pr, "layers": []}

    t0 = perf_counter()
    for li in range(n_layers):
        b_roc, b_pr = rf_fold_metrics(np.ascontiguousarray(emb[li]), y, cv)
        droc = [b - a for b, a in zip(b_roc, a_roc)]
        m, lo, hi, sig = fold_tci(droc)
        out["layers"].append({
            "layer": li, "roc": b_roc, "pr": b_pr,
            "roc_mean": float(np.mean(b_roc)), "pr_mean": float(np.mean(b_pr)),
            "delta_roc_mean": m, "delta_roc_ci": [lo, hi], "significant": sig,
        })
        print(f"  layer {li:2d}/{n_layers - 1}: ROC={np.mean(b_roc):.3f} "
              f"dROC={m:+.3f} [{lo:+.3f}, {hi:+.3f}]{'*' if sig else ''}  "
              f"({perf_counter() - t0:.0f}s)", flush=True)

    best = max(out["layers"], key=lambda r: r["delta_roc_mean"])
    last = out["layers"][-1]
    out["best_layer"] = best["layer"]
    out["last_layer_delta"] = last["delta_roc_mean"]
    out["best_layer_delta"] = best["delta_roc_mean"]
    out["gain_over_last"] = best["delta_roc_mean"] - last["delta_roc_mean"]
    print(f"[{basin}] published cell (last layer {last['layer']}): "
          f"dROC={last['delta_roc_mean']:+.3f}")
    print(f"[{basin}] best layer {best['layer']}: "
          f"dROC={best['delta_roc_mean']:+.3f} "
          f"(gain over last: {out['gain_over_last']:+.3f})")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--basin", action="append", required=True)
    ap.add_argument("--layer-dir", default="../results/_layers")
    ap.add_argument("--folds", type=int, default=N_FOLDS)
    a = ap.parse_args()
    for b in a.basin:
        res = evaluate(b, Path(a.layer_dir), a.folds)
        dst = RESULTS / f"{b}_prithvi_layer_sweep.json"
        dst.write_text(json.dumps(res, indent=2))
        print(f"[done] wrote {dst.name}\n")


if __name__ == "__main__":
    main()
