"""Kaggle ASL Signs landmarks via a public mirror, streamed and converted.

The competition's own per-file API has a daily request quota (we hit it after
a few hundred of the ~94k files). This mirror has the same landmarks as one
.npy per sequence (frames, 543, 3) in the competition's order
(face 0-467, left hand 468-488, pose 489-521, right hand 522-542):
https://www.kaggle.com/datasets/kaerunantoka/asl-signs-npy (~9.3 GB zip)

The zip is streamed and each sequence is cut down to our 88-point layout on
the fly, so the 9 GB never has to be on disk (~1 GB of .npz instead).
Labels come from the competition's train.csv (download it once with
download_kaggle_asl.py, or from the competition page).

    python scripts/download_gislr_mirror.py
"""
import argparse
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from stream_unzip import stream_unzip
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from echogesture import layout as L  # noqa: E402
from scripts.download_kaggle_wlasl import credentials  # noqa: E402

URL = "https://www.kaggle.com/api/v1/datasets/download/kaerunantoka/asl-signs-npy"
# competition order -> our 88-point layout
SRC = np.r_[489:522, 468:489, 522:543, np.array(L.FACE_IDX)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/kaggle_asl")
    args = ap.parse_args()
    root = Path(args.root)
    out_dir = root / "landmarks"
    out_dir.mkdir(parents=True, exist_ok=True)

    r = requests.get(URL, auth=credentials(), stream=True, timeout=120)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    bar = tqdm(total=total, unit="B", unit_scale=True)

    def chunks():
        for c in r.iter_content(1 << 20):
            bar.update(len(c))
            yield c

    n = 0
    for name, _, data in stream_unzip(chunks()):
        name = name.decode()
        body = b"".join(data)  # always drain the entry before moving on
        if not name.endswith(".npy"):
            continue
        seq = Path(name).stem.split("_")[-1]
        out = out_dir / f"{seq}.npz"
        if out.exists():
            continue
        x = np.load(io.BytesIO(body))
        if x.ndim != 3 or x.shape[1] != 543:
            continue
        np.savez_compressed(out, xyz=x[:, SRC, :3].astype(np.float16))
        n += 1
    bar.close()
    print(f"converted {n} sequences")

    have = {p.stem for p in out_dir.glob("*.npz")}
    m = pd.read_csv(root / "train.csv")
    m = m[m.sequence_id.astype(str).isin(have)].rename(columns={"participant_id": "signer_id", "sign": "gloss"})
    m["video_id"] = m.sequence_id.astype(str)
    m["gloss"] = m.gloss.str.upper()
    m.to_csv(root / "manifest.csv", index=False)
    print(f"manifest: {len(m)} sequences, {m.gloss.nunique()} signs, {m.signer_id.nunique()} signers")


if __name__ == "__main__":
    main()
