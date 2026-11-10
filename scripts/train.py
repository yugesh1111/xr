"""Train EchoNet.

    # E1 raw landmarks          E2 canonical features
    python scripts/train.py --mode raw --fusion flat --out runs/e1
    python scripts/train.py --mode canonical --fusion flat --out runs/e2
    # E3 factorized             E4 + quality gate          E5 + confusion margin
    python scripts/train.py --fusion sum --out runs/e3
    python scripts/train.py --fusion gated --out runs/e4
    python scripts/train.py --fusion gated --confusion-margin 1.0 --out runs/e5
    # E6 = E5 + stronger view/occlusion augmentation
    python scripts/train.py --fusion gated --confusion-margin 1.0 --aug strong --out runs/e6
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from echogesture.data import ClassBalancedSampler, ClipDataset, load_split  # noqa: E402
from echogesture.inputs import group_slices  # noqa: E402
from echogesture.losses import ConfusionMarginLoss, SupConLoss  # noqa: E402
from echogesture.model import EchoNet  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", default="data/wlasl/manifest.csv")
ap.add_argument("--landmarks", default="data/wlasl/landmarks")
ap.add_argument("--out", default="runs/e4")
ap.add_argument("--mode", choices=["raw", "canonical"], default="canonical")
ap.add_argument("--fusion", choices=["flat", "sum", "gated"], default="gated")
ap.add_argument("--confusion-margin", type=float, default=0.0)
ap.add_argument("--aug", choices=["default", "strong"], default="default")
ap.add_argument("--supcon", type=float, default=0.0, help="weight of the supervised contrastive loss")
ap.add_argument("--balanced", action="store_true", help="P x K class-balanced batches (on by default with --supcon)")
ap.add_argument("--dropout", type=float, default=0.2)
ap.add_argument("--n-conv", type=int, default=4)
ap.add_argument("--n-attn", type=int, default=2)
ap.add_argument("--alt-landmarks", nargs="*", default=[],
                help="folders with landmarks of the same clips from degraded video (make_degraded.py)")
ap.add_argument("--init", default=None, help="checkpoint to start from (e.g. runs/pretrain/best.pt); head is re-initialised")
ap.add_argument("--extra-manifest", default=None, help="extra training clips (e.g. Kaggle ASL Signs) for glosses in our vocabulary")
ap.add_argument("--extra-landmarks", default=None)
ap.add_argument("--extra-per-class", type=int, default=20, help="max extra clips per gloss, keeps classes balanced")
ap.add_argument("--length", type=int, default=32)
ap.add_argument("--p-close", type=float, default=0.0,
                help="share of training samples turned into phone close-ups (shoulders off-screen)")
ap.add_argument("--p-far", type=float, default=0.0,
                help="share of training samples turned into a small, distant signer (noisier landmarks)")
ap.add_argument("--dim", type=int, default=192)
ap.add_argument("--epochs", type=int, default=120)
ap.add_argument("--batch-size", type=int, default=64)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--weight-decay", type=float, default=0.05)
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--device", default="auto")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
if args.device == "auto":
    args.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

tr_paths, tr_y, vocab, _ = load_split(args.manifest, args.landmarks, "train")
va_paths, va_y, _, _ = load_split(args.manifest, args.landmarks, "val", vocab)
if args.extra_manifest:
    import pandas as pd
    ex = pd.read_csv(args.extra_manifest, dtype={"video_id": str})
    ex = ex[(ex.split == "train") & ex.gloss.isin(vocab)]
    ex = ex.sample(frac=1, random_state=0).groupby("gloss").head(args.extra_per_class)
    index = {g: i for i, g in enumerate(vocab)}
    tr_paths = list(tr_paths) + [Path(args.extra_landmarks) / f"{v}.npz" for v in ex.video_id]
    tr_y = np.concatenate([tr_y, [index[g] for g in ex.gloss]])
    print(f"+ {len(ex)} extra training clips for {ex.gloss.nunique()} glosses")
print(f"train {len(tr_paths)} clips, val {len(va_paths)} clips, {len(vocab)} classes, device {args.device}")
train_ds = ClipDataset(tr_paths, tr_y, args.mode, args.length, train=True, aug=args.aug,
                       alt_dirs=args.alt_landmarks, p_close=args.p_close, p_far=args.p_far)
if args.alt_landmarks:
    print(f"degraded-video landmarks for {sum(1 for a in train_ds.alts if a)} of {len(train_ds)} training clips")
val_ds = ClipDataset(va_paths, va_y, args.mode, args.length)
if args.balanced or args.supcon > 0:
    train_dl = DataLoader(train_ds, batch_sampler=ClassBalancedSampler(tr_y, p=16, k=4, seed=args.seed),
                          num_workers=args.workers, persistent_workers=args.workers > 0)
else:
    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=args.workers,
                          drop_last=len(train_ds) > args.batch_size, persistent_workers=args.workers > 0)
val_dl = DataLoader(val_ds, 256, num_workers=args.workers)

slices = group_slices(args.mode)
model = EchoNet(slices, len(vocab), d=args.dim, fusion=args.fusion, length=args.length,
                n_conv=args.n_conv, n_attn=args.n_attn, dropout=args.dropout)

if args.init:
    # keep the pre-trained encoder and its input normalization; new classifier head
    ck = torch.load(args.init, map_location="cpu")
    state = {k: v for k, v in ck["model"].items() if not k.startswith("head.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"initialised from {args.init}; new: {missing}")
sample = None if args.init else np.stack([ClipDataset(tr_paths[:3000], tr_y[:3000], args.mode, args.length)[i][0].numpy()
                   for i in range(min(len(tr_paths), 3000))]).reshape(-1, model.mu.numel())
if sample is not None:
    model.set_normalization(sample.mean(0), sample.std(0))
model.to(args.device)

loss_fn = ConfusionMarginLoss(len(vocab), margin=args.confusion_margin).to(args.device)
supcon = SupConLoss()
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
steps = args.epochs * len(train_dl)
warm = max(int(0.05 * steps), 1)
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s / steps, 1.0))))


@torch.no_grad()
def evaluate():
    model.eval()
    logits, ys = [], []
    for x, y in val_dl:
        logits.append(model(x.to(args.device)).cpu())
        ys.append(y)
    logits, ys = torch.cat(logits), torch.cat(ys)
    top5 = logits.topk(min(5, logits.shape[1]), 1).indices
    return {
        "val_loss": torch.nn.functional.cross_entropy(logits, ys).item(),
        "top1": (top5[:, 0] == ys).float().mean().item(),
        "top5": (top5 == ys[:, None]).any(1).float().mean().item(),
    }, logits.argmax(1).numpy(), ys.numpy()


config = vars(args) | {"slices": slices, "n_classes": len(vocab)}
(out / "vocab.json").write_text(json.dumps(vocab))
best, log = -1.0, open(out / "metrics.jsonl", "w")
for epoch in range(args.epochs):
    model.train()
    loss_fn.new_epoch()
    t0, total, correct, seen = time.time(), 0.0, 0, 0
    for x, y in train_dl:
        x, y = x.to(args.device), y.to(args.device)
        emb = model.embed(x)
        logits = model.head(model.drop(emb))
        loss = loss_fn(logits, y)
        if args.supcon > 0:
            loss = loss + args.supcon * supcon(emb, y)
        correct += (logits.argmax(1) == y).sum().item()
        seen += len(y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        total += loss.item()
    m, pred, ys = evaluate()
    m |= {"epoch": epoch + 1, "train_loss": total / len(train_dl), "train_top1": correct / max(seen, 1),
          "sec": round(time.time() - t0, 1)}
    log.write(json.dumps(m) + "\n")
    log.flush()
    print(f"ep {epoch + 1:3d}  loss {m['train_loss']:.3f}  train top1 {m['train_top1']:.3f}  val top1 {m['top1']:.3f}  top5 {m['top5']:.3f}  ({m['sec']}s)")
    if m["top1"] > best:
        best = m["top1"]
        torch.save({"model": model.state_dict(), "config": config, "vocab": vocab}, out / "best.pt")
        # most confused pairs on validation, for error analysis
        pairs = {}
        for p, t in zip(pred, ys):
            if p != t:
                pairs[(vocab[t], vocab[p])] = pairs.get((vocab[t], vocab[p]), 0) + 1
        top = sorted(pairs.items(), key=lambda kv: -kv[1])[:30]
        (out / "confusions.json").write_text(json.dumps([[a, b, n] for (a, b), n in top], indent=1))
print(f"best val top1 {best:.3f} -> {out / 'best.pt'}")
