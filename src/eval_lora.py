"""Score the LoRA-adapted embeddings with the paper's classifier and folds.

Each fold has its own adapted backbone, so each fold must be scored with the
embeddings produced by that fold's backbone -- pooling them into a single matrix
would evaluate fold k's test points under a model that saw them during
adaptation elsewhere.

The head used during adaptation is discarded here on purpose: the numbers come
from the same RandomForest the benchmark uses, so the only thing that differs
from the published Prithvi cell is the representation.

Reports the adapted arm against two references on identical folds: the manual
baseline A (the paper's headline comparison) and the frozen linear probe (the
question the fine-tuning arm exists to answer).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from config import RESULTS
from point_probes import build_dataset, rf_fold_metrics
from terramind_linprobe import (
    N_FOLDS, N_TREES, PRITHVI_NAME, SEED, embedding_fingerprint,
)


def tci(v):
    n = len(v)
    m = sum(v) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1))
    h = float(student_t.ppf(0.975, n - 1)) * sd / math.sqrt(n)
    return m, m - h, m + h, not (m - h <= 0 <= m + h)


def fmt(v):
    m, lo, hi, s = tci(v)
    return f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]{'*' if s else ''}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--basin", required=True)
    ap.add_argument("--lora-dir", default="../results/_lora")
    a = ap.parse_args()

    rows, cols, y, X17, _blk = build_dataset(a.basin, negatives="uniform")
    fp = embedding_fingerprint(PRITHVI_NAME, "pretrained", ["S2L2A"],
                               rows, cols, y)
    folds_npz = np.load(RESULTS / "_patches" / f"{a.basin}_folds.npz",
                        allow_pickle=False)
    if str(folds_npz["fingerprint"]) != fp:
        raise SystemExit("fold file describes a different point set")
    cv = [(folds_npz[f"train_{k}"], folds_npz[f"test_{k}"])
          for k in range(int(folds_npz["n_folds"]))]

    a_roc, a_pr = rf_fold_metrics(X17, y, cv)
    print(f"[{a.basin}] baseline A: ROC={np.mean(a_roc):.3f} PR={np.mean(a_pr):.3f}")

    frozen = json.loads(
        (RESULTS / f"{a.basin}_prithvi-300m_linprobe_spatial.json").read_text())
    f_roc = [r["roc_B"] for r in frozen["fold_results"]]
    print(f"[{a.basin}] frozen probe (published): ROC={np.mean(f_roc):.3f}")

    l_roc, l_pr, val_aucs = [], [], []
    for k, (tr, te) in enumerate(cv):
        d = np.load(Path(a.lora_dir) / f"{a.basin}_prithvi_lora_fold{k}.npz",
                    allow_pickle=False)
        if str(d["fingerprint"]) != fp:
            raise SystemExit(f"fold {k} embeddings describe a different point set")
        emb = d["embeddings"]
        val_aucs.append(float(d["val_auc"]))
        clf = RandomForestClassifier(
            n_estimators=N_TREES, min_samples_leaf=5, n_jobs=-1,
            random_state=SEED + k, class_weight="balanced",
        ).fit(emb[tr], y[tr])
        s = clf.predict_proba(emb[te])[:, 1]
        l_roc.append(float(roc_auc_score(y[te], s)))
        l_pr.append(float(average_precision_score(y[te], s)))
        print(f"  fold {k}: LoRA ROC={l_roc[-1]:.3f}  frozen={f_roc[k]:.3f}  "
              f"A={a_roc[k]:.3f}  (val_auc during adaptation {val_aucs[-1]:.3f})")

    d_vs_a = [l - b for l, b in zip(l_roc, a_roc)]
    d_vs_f = [l - b for l, b in zip(l_roc, f_roc)]
    print(f"\n[{a.basin}] LoRA-adapted ROC = {np.mean(l_roc):.3f} "
          f"(frozen {np.mean(f_roc):.3f}, baseline A {np.mean(a_roc):.3f})")
    print(f"  dROC vs baseline A   : {fmt(d_vs_a)}")
    print(f"  dROC vs frozen probe : {fmt(d_vs_f)}")

    out = {
        "basin": a.basin, "n_folds": len(cv), "fingerprint": fp,
        "lora_roc": l_roc, "lora_pr": l_pr, "val_auc_per_fold": val_aucs,
        "frozen_roc": f_roc, "A_roc": a_roc,
        "delta_vs_A": {"mean": tci(d_vs_a)[0], "ci": list(tci(d_vs_a)[1:3]),
                       "significant": tci(d_vs_a)[3]},
        "delta_vs_frozen": {"mean": tci(d_vs_f)[0], "ci": list(tci(d_vs_f)[1:3]),
                            "significant": tci(d_vs_f)[3]},
    }
    dst = RESULTS / f"{a.basin}_prithvi_lora.json"
    dst.write_text(json.dumps(out, indent=2))
    print(f"[done] wrote {dst.name}")


if __name__ == "__main__":
    main()
