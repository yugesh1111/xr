"""Video-level augmentation: extract landmarks from randomly degraded copies
of the *training* videos (blur, low resolution, darkness, or a mix).

    python scripts/make_degraded.py --copies 2
    python scripts/train.py --fusion gated --alt-landmarks data/wlasl/landmarks_aug1 data/wlasl/landmarks_aug2 --out runs/e13

Each copy uses a different random degradation per clip, seeded by clip id
and copy number, so the result is reproducible.
"""
import argparse
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from echogesture.corruptions import random_degradation  # noqa: E402


def job(row, root, copy):
    from echogesture.extract import extract_video
    out = root / f"landmarks_aug{copy}" / f"{row['video_id']}.npz"
    if out.exists():
        return "exists"
    src = next((root / "raw").glob(f"{row['video_id']}.*"), None)
    if src is None:
        return "missing"
    seed = zlib.crc32(f"{row['video_id']}-{copy}".encode())
    xyz = extract_video(src, int(row["frame_start"]), int(row["frame_end"]), transform=random_degradation(seed))
    np.savez_compressed(out, xyz=xyz.astype(np.float16))
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/wlasl")
    ap.add_argument("--copies", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    root = Path(args.root)
    df = pd.read_csv(root / "manifest.csv", dtype={"video_id": str})
    rows = df[df.split == "train"][["video_id", "frame_start", "frame_end"]].to_dict("records")
    for c in range(1, args.copies + 1):
        (root / f"landmarks_aug{c}").mkdir(exist_ok=True)
        with ProcessPoolExecutor(args.workers) as pool:
            res = list(tqdm(pool.map(partial(job, root=root, copy=c), rows), total=len(rows), desc=f"copy {c}"))
        print(f"copy {c}:", pd.Series(res).value_counts().to_dict())


if __name__ == "__main__":
    main()
