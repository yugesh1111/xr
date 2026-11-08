"""Copy the clips we need out of the Kaggle WLASL mirror into <root>/raw.

The mirror (kaggle: risangbaskoro/wlasl-processed) is the original WLASL
videos that were still online in 2021, named <video_id>.mp4.

    curl -L -o data/kaggle_wlasl/archive.zip \\
        https://www.kaggle.com/api/v1/datasets/download/risangbaskoro/wlasl-processed
    python scripts/unpack_kaggle_wlasl.py
"""
import argparse
import shutil
import zipfile
from pathlib import Path

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--zip", default="data/kaggle_wlasl/archive.zip")
ap.add_argument("--root", default="data/wlasl")
args = ap.parse_args()

root = Path(args.root)
(root / "raw").mkdir(parents=True, exist_ok=True)
want = set(pd.read_csv(root / "manifest.csv", dtype={"video_id": str}).video_id)
n = 0
with zipfile.ZipFile(args.zip) as z:
    for name in z.namelist():
        vid = Path(name).stem
        if name.startswith("videos/") and vid in want and not any((root / "raw").glob(f"{vid}.*")):
            with z.open(name) as src, open(root / "raw" / f"{vid}.mp4", "wb") as dst:
                shutil.copyfileobj(src, dst)
            n += 1
print(f"copied {n} clips into {root / 'raw'}")
