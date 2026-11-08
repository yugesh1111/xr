"""Pre-train EchoNet on Kaggle ASL Signs (250 signs, ~94k clips, 21 signers).

Inputs are computed once (no augmentation: with 94k clips the data itself
supplies the variety) and kept in memory, so an epoch takes seconds of
data work instead of minutes. The encoder is then fine-tuned on WLASL with
`train.py --init runs/pretrain/best.pt`.

    python scripts/pretrain.py --out runs/pretrain
"""
import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from echogesture.inputs import group_slices, make_input  # noqa: E402
from echogesture.model import EchoNet  # noqa: E402


def load_input(vid, root, length):
    x = np.load(root / f"{vid}.npz")["xyz"].astype(np.float32)
    try:
        return make_input(x, "canonical", length).astype(np.float16)
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/kaggle_asl")
    ap.add_argument("--out", default="runs/pretrain")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--dim", type=int, default=192)
    ap.add_argument("--length", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    m = pd.read_csv(root / "manifest_split.csv", dtype={"video_id": str})
    vocab = sorted(m.gloss.unique())
    idx = {g: i for i, g in enumerate(vocab)}
    with ProcessPoolExecutor(args.workers) as pool:
        X = list(tqdm(pool.map(partial(load_input, root=root / "landmarks_fixed", length=args.length),
                               m.video_id, chunksize=64), total=len(m), desc="inputs"))
    keep = np.array([x is not None for x in X])
    X = torch.from_numpy(np.stack([x for x in X if x is not None]))
    y = torch.tensor([idx[g] for g in m.gloss[keep]])
    split = m.split[keep].to_numpy()
    tr, va = np.flatnonzero(split == "train"), np.flatnonzero(split == "val")
    print(f"train {len(tr)}, val {len(va)} (unseen signers), {len(vocab)} signs")

    slices = group_slices("canonical")
    model = EchoNet(slices, len(vocab), d=args.dim, fusion="gated", length=args.length)
    Xtr = X[tr].float()
    model.set_normalization(Xtr.mean((0, 1)).numpy(), Xtr.std((0, 1)).numpy())
    del Xtr
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    steps = args.epochs * math.ceil(len(tr) / args.batch_size)
    warm = max(steps // 20, 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s / steps, 1.0))))

    config = {"mode": "canonical", "fusion": "gated", "dim": args.dim, "length": args.length,
              "slices": slices, "n_classes": len(vocab), "n_conv": 4, "n_attn": 2, "dropout": 0.2,
              "pretrain": "kaggle-asl-signs"}
    best, log = -1, open(out / "metrics.jsonl", "w")
    for epoch in range(args.epochs):
        model.train()
        t0, total = time.time(), 0.0
        perm = tr[torch.randperm(len(tr)).numpy()]
        for i in range(0, len(perm), args.batch_size):
            b = perm[i:i + args.batch_size]
            xb = X[b].float()
            xb = xb + 0.02 * torch.randn_like(xb) * (xb != 0)  # light noise on observed values
            loss = torch.nn.functional.cross_entropy(model(xb), y[b], label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.item() * len(b)
        model.eval()
        with torch.no_grad():
            pred = torch.cat([model(X[va[i:i + 1024]].float()).argmax(1) for i in range(0, len(va), 1024)])
        acc = (pred == y[va]).float().mean().item()
        rec = {"epoch": epoch + 1, "train_loss": total / len(tr), "val_top1": acc, "sec": round(time.time() - t0, 1)}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(f"ep {epoch + 1:2d}  loss {rec['train_loss']:.3f}  val top1 {acc:.3f}  ({rec['sec']}s)")
        if acc > best:
            best = acc
            torch.save({"model": model.state_dict(), "config": config, "vocab": vocab}, out / "best.pt")
    print(f"best val top1 {best:.3f} -> {out / 'best.pt'}")


if __name__ == "__main__":
    main()
