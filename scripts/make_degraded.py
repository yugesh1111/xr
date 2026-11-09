"""Video-level augmentation: extract landmarks from randomly degraded copies
of the *training* videos (blur, low resolution, darkness, or a mix).

    python scripts/make_degraded.py --copies 2
    python scripts/train.py --fusion gated --alt-landmarks data/wlasl/landmarks_aug1 data/wlasl/landmarks_aug2 --out runs/e13

    # E14: low-resolution copies, through the deployed pipeline (adaptive
    # sharpening), for every clip so the held-out-signer runs can use them too
    python scripts/make_degraded.py --kind low_res --prefix landmarks_lr --enhance --splits train val test

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
from echogesture.corruptions import random_degradation, random_low_res  # noqa: E402

KINDS = {"mixed": random_degradation, "low_res": random_low_res}


def job(row, root, copy, kind="mixed", prefix="landmarks_aug", enhance=False):
    from echogesture.extract import extract_video
    out = root / f"{prefix}{copy}" / f"{row['video_id']}.npz"
    if out.exists():
        return "exists"
    src = next((root / "raw").glob(f"{row['video_id']}.*"), None)
    if src is None:
        return "missing"
    seed = zlib.crc32(f"{row['video_id']}-{copy}{'' if kind == 'mixed' else '-' + kind}".encode())
    xyz = extract_video(src, int(row["frame_start"]), int(row["frame_end"]), transform=KINDS[kind](seed),
                        enhance=enhance)
    np.savez_compressed(out, xyz=xyz.astype(np.float16))
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/wlasl")
    ap.add_argument("--copies", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--kind", choices=list(KINDS), default="mixed")
    ap.add_argument("--prefix", default="landmarks_aug")
    ap.add_argument("--enhance", action="store_true", help="adaptive sharpening, as the deployed pipeline")
    ap.add_argument("--splits", nargs="+", default=["train"])
    args = ap.parse_args()
    root = Path(args.root)
    df = pd.read_csv(root / "manifest.csv", dtype={"video_id": str})
    rows = df[df.split.isin(args.splits)][["video_id", "frame_start", "frame_end"]].to_dict("records")
    for c in range(1, args.copies + 1):
        (root / f"{args.prefix}{c}").mkdir(exist_ok=True)
        with ProcessPoolExecutor(args.workers) as pool:
            res = list(tqdm(pool.map(partial(job, root=root, copy=c, kind=args.kind, prefix=args.prefix,
                                             enhance=args.enhance), rows), total=len(rows), desc=f"copy {c}"))
        print(f"copy {c}:", pd.Series(res).value_counts().to_dict())


if __name__ == "__main__":
    main()
