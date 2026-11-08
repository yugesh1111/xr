"""Convert Kaggle ASL Signs parquet files into our .npz format.

Download first (needs a Kaggle account and accepting the competition rules):
    kaggle competitions download -c asl-signs -p data/kaggle_asl && unzip ...

    python scripts/convert_kaggle_asl.py --root data/kaggle_asl
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
from echogesture.kaggle_asl import parquet_to_array  # noqa: E402

def job(path, root, out_dir):
    out = out_dir / f"{Path(path).stem}.npz"
    if not out.exists():
        np.savez_compressed(out, xyz=parquet_to_array(root / path).astype(np.float16))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/kaggle_asl")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    root = Path(args.root)
    out_dir = root / "landmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(root / "train.csv")  # path, participant_id, sequence_id, sign
    with ProcessPoolExecutor(args.workers) as pool:
        list(tqdm(pool.map(partial(job, root=root, out_dir=out_dir), train.path, chunksize=16), total=len(train)))
    manifest = train.rename(columns={"participant_id": "signer_id", "sign": "gloss"})
    manifest["video_id"] = manifest.sequence_id.astype(str)
    manifest["gloss"] = manifest.gloss.str.upper()
    manifest.to_csv(root / "manifest.csv", index=False)
    print(f"converted {len(train)} sequences; manifest at {root / 'manifest.csv'}")


if __name__ == "__main__":
    main()
