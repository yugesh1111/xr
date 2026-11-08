"""Kaggle ASL Signs: fix aspect ratio, split by signer, map glosses to WLASL.

    python scripts/prepare_kaggle_asl.py

Writes data/kaggle_asl/landmarks_fixed/*.npz and data/kaggle_asl/manifest_split.csv
with columns video_id, gloss, signer_id, split (train/val), in_wlasl.
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from echogesture.kaggle_asl import fix_aspect  # noqa: E402


def job(vid, src, dst):
    out = dst / f"{vid}.npz"
    if not out.exists():
        x = np.load(src / f"{vid}.npz")["xyz"]
        np.savez_compressed(out, xyz=fix_aspect(x).astype(np.float16))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/kaggle_asl")
    ap.add_argument("--wlasl-manifest", default="data/wlasl/manifest.csv")
    ap.add_argument("--val-signers", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    root = Path(args.root)
    src, dst = root / "landmarks", root / "landmarks_fixed"
    dst.mkdir(exist_ok=True)
    m = pd.read_csv(root / "manifest.csv", dtype={"video_id": str})
    with ProcessPoolExecutor(args.workers) as pool:
        list(tqdm(pool.map(partial(job, src=src, dst=dst), m.video_id, chunksize=64), total=len(m)))
    signers = sorted(m.signer_id.unique())
    val = set(np.random.default_rng(0).choice(signers, args.val_signers, replace=False))
    m["split"] = np.where(m.signer_id.isin(val), "val", "train")
    m["in_wlasl"] = m.gloss.isin(set(pd.read_csv(args.wlasl_manifest).gloss))
    m[["video_id", "gloss", "signer_id", "split", "in_wlasl"]].to_csv(root / "manifest_split.csv", index=False)
    print(m.groupby("split").agg(clips=("video_id", "size"), signers=("signer_id", "nunique"),
                                 signs=("gloss", "nunique")).to_string())


if __name__ == "__main__":
    main()
