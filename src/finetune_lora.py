"""LoRA fine-tuning arm for Prithvi-EO-2.0 (GPU host).

The benchmark evaluates frozen linear probes. A fair criticism is that this does
not test what the foundation-model literature actually does, which is to adapt
the backbone. Full fine-tuning of a 300M ViT-L does not fit in 6 GB of VRAM, and
with 156-178 training positives per fold it would be the wrong choice anyway:
that is roughly 1.8 M parameters per positive example. Low-rank adaptation is
both what fits and what the sample size warrants.

Protocol, chosen so the result stays comparable to Table 3:

  * Adaptation happens INSIDE each spatial fold. Adapting once and then
    cross-validating would leak the test fold into the backbone.
  * The folds are the published ones, shipped from the host that produced the
    paper rather than recomputed here.
  * Training uses a differentiable linear head, but the head is then discarded:
    the reported numbers come from re-fitting the paper's RandomForest on the
    adapted embeddings. Otherwise the comparison would move two things at once
    (representation and classifier) and neither effect would be attributable.
  * An inner validation split, carved from the training folds only, drives early
    stopping. The test fold is never seen during adaptation.

Writes {basin}_prithvi_lora_fold{k}.npz with embeddings for all points under the
backbone adapted on that fold.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

EMBED_DIM = 1024
LORA_TARGETS = r".*encoder\.blocks\.\d+\.(attn\.(qkv|proj)|mlp\.(fc1|fc2))"


def build_backbone(prithvi_path: str):
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
    model.load_state_dict(ckpt, strict=False)
    return model, norm


class Probe(nn.Module):
    """Adapted backbone + linear head on mean-pooled final-layer tokens."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(EMBED_DIM, 1)

    def embed(self, x):
        feats = self.backbone.forward_features(x)
        last = feats[-1] if isinstance(feats, (list, tuple)) else feats
        return last[:, 1:, :].mean(dim=1)      # drop CLS, as in the benchmark

    def forward(self, x):
        return self.head(self.embed(x)).squeeze(-1)


def batches(idx, bs, shuffle, rng=None):
    order = rng.permutation(idx) if shuffle else idx
    for i in range(0, len(order), bs):
        yield order[i:i + bs]


def run_fold(fold, patches, y, folds, norm, args, device):
    from peft import LoraConfig, get_peft_model

    tr_all = folds[f"train_{fold}"]
    te = folds[f"test_{fold}"]

    rng = np.random.default_rng(args.seed + fold)
    perm = rng.permutation(tr_all)
    n_val = max(1, int(round(len(perm) * args.val_frac)))
    val, tr = perm[:n_val], perm[n_val:]

    backbone, _ = build_backbone(args.prithvi_path)
    cfg = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
                     target_modules=LORA_TARGETS, bias="none")
    backbone = get_peft_model(backbone, cfg)
    model = Probe(backbone).to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_tr = sum(p.numel() for p in trainable)
    print(f"[fold {fold}] trainable {n_tr / 1e6:.2f} M of "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f} M  "
          f"| train {len(tr)} val {len(val)} test {len(te)}")

    pos_w = torch.tensor([(y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)],
                         dtype=torch.float32, device=device)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler(device)
    mean = torch.from_numpy(norm["mean"]).to(device)
    std = torch.from_numpy(norm["std"]).to(device)

    def prep(ix):
        x = torch.from_numpy(np.ascontiguousarray(patches[ix])).to(device)
        x = x.to(torch.float32)
        # BOA_ADD_OFFSET removed from valid pixels, as in _encode_prithvi.
        x = torch.where((x > 0).any(dim=1, keepdim=True), x - 1000.0, x)
        return (x - mean) / std

    best_auc, best_state, bad = -1.0, None, 0
    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        for ix in batches(tr, args.batch_size, True, rng):
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device, dtype=torch.float16):
                loss = lossf(model(prep(ix)),
                             torch.from_numpy(y[ix]).to(device, torch.float32))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += float(loss) * len(ix)

        model.eval()
        scores = []
        with torch.no_grad():
            for ix in batches(val, args.batch_size, False):
                with torch.autocast(device, dtype=torch.float16):
                    scores.append(model(prep(ix)).float().cpu().numpy())
        auc = roc_auc_score(y[val], np.concatenate(scores))
        print(f"  ep {ep:2d} loss={tot / len(tr):.4f} val_auc={auc:.4f}"
              f"{'  *' if auc > best_auc else ''}", flush=True)
        if auc > best_auc:
            best_auc, bad = auc, 0
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()
                          if v.requires_grad or "lora" in k}
        else:
            bad += 1
            if bad >= args.patience:
                print(f"  early stop at epoch {ep} (best val_auc={best_auc:.4f})")
                break

    if best_state:
        model.load_state_dict(best_state, strict=False)

    # embeddings for every point under this fold's adapted backbone, in fp32
    model.eval()
    emb = np.zeros((len(y), EMBED_DIM), dtype=np.float32)
    with torch.no_grad():
        for ix in batches(np.arange(len(y)), args.batch_size, False):
            emb[ix] = model.embed(prep(ix)).float().cpu().numpy()

    del model, backbone
    torch.cuda.empty_cache()
    return emb, best_auc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--basin", required=True)
    ap.add_argument("--patch-dir", default="~/geofm/patches")
    ap.add_argument("--out-dir", default="~/geofm/lora")
    ap.add_argument("--prithvi-path", default="~/models/prithvi-300m")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", default="")
    a = ap.parse_args()

    pd_ = Path(a.patch_dir).expanduser()
    a.prithvi_path = str(Path(a.prithvi_path).expanduser())
    patches = np.load(pd_ / f"{a.basin}_prithvi_patches.u16", mmap_mode="r")
    folds = np.load(pd_ / f"{a.basin}_folds.npz", allow_pickle=False)
    meta = np.load(pd_ / f"{a.basin}_prithvi_patches.npz", allow_pickle=False)
    if str(folds["fingerprint"]) != str(meta["fingerprint"]):
        raise SystemExit("folds and patches describe different point sets")
    y = folds["y"].astype(np.int64)
    _, norm = None, None
    _, norm = build_backbone(a.prithvi_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(a.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    want = [int(x) for x in a.folds.split(",")] if a.folds else \
        list(range(int(folds["n_folds"])))

    for k in want:
        t0 = perf_counter()
        emb, auc = run_fold(k, patches, y, folds, norm, a, device)
        dst = out_dir / f"{a.basin}_prithvi_lora_fold{k}.npz"
        np.savez_compressed(dst, embeddings=emb, fold=np.array(k),
                            val_auc=np.array(auc),
                            fingerprint=meta["fingerprint"])
        print(f"[ok] fold {k} -> {dst.name} val_auc={auc:.4f} "
              f"({perf_counter() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
