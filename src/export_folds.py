"""Export the benchmark's spatial folds so the GPU host trains on the same
partition the paper evaluates on. Recomputing them remotely would risk a silent
drift; shipping the indices makes the split auditable."""
import sys
import numpy as np
from sklearn.model_selection import GroupKFold
from point_probes import build_dataset
from terramind_linprobe import N_FOLDS, PRITHVI_NAME, embedding_fingerprint

for basin in sys.argv[1:]:
    rows, cols, y, X17, blk = build_dataset(basin, negatives="uniform")
    fp = embedding_fingerprint(PRITHVI_NAME, "pretrained", ["S2L2A"], rows, cols, y)
    folds = list(GroupKFold(n_splits=N_FOLDS).split(X17, y, groups=blk))
    d = {"fingerprint": np.array(fp), "y": y, "n_folds": np.array(N_FOLDS)}
    for i, (tr, te) in enumerate(folds):
        d[f"train_{i}"] = tr
        d[f"test_{i}"] = te
        print(f"  {basin} fold {i}: train={len(tr)} (pos {int(y[tr].sum())}) "
              f"test={len(te)} (pos {int(y[te].sum())})")
    np.savez(f"../results/_patches/{basin}_folds.npz", **d)
    print(f"[ok] {basin}_folds.npz  fingerprint {fp[:12]}")
