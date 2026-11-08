"""Run MediaPipe Holistic over every downloaded clip -> <root>/landmarks/<id>.npz

    python scripts/extract_landmarks.py --workers 4

Each .npz holds `xyz` (T, 88, 3) float16 with NaN where nothing was detected.
Resumable: existing files are skipped.
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
from echogesture.extract import ensure_model, extract_video  # noqa: E402


def job(row, root, out_dir):
    out = out_dir / f"{row['video_id']}.npz"
    if out.exists():
        return "exists"
    src = next((root / "raw").glob(f"{row['video_id']}.*"), None)
    if src is None:
        return "missing"
    try:
        xyz = extract_video(src, int(row["frame_start"]), int(row["frame_end"]))
        np.savez_compressed(out, xyz=xyz.astype(np.float16))
        return "ok" if len(xyz) else "empty"
    except Exception as e:
        return f"error: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/wlasl")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    root = Path(args.root)
    out_dir = root / "landmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    ensure_model()
    df = pd.read_csv(root / "manifest.csv", dtype={"video_id": str}).drop_duplicates("video_id")
    rows = df[["video_id", "frame_start", "frame_end"]].to_dict("records")
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(tqdm(pool.map(partial(job, root=root, out_dir=out_dir), rows, chunksize=4), total=len(rows)))
    print(pd.Series([r.split(":")[0] for r in results]).value_counts().to_string())


if __name__ == "__main__":
    main()
