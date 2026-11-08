"""Download Kaggle ASL Signs (Google Isolated Sign Language Recognition)
landmark files one by one and convert each straight to our .npz format.

Competition: https://www.kaggle.com/competitions/asl-signs
Needs the Kaggle key (see README) and the competition rules accepted on
kaggle.com with that account. The full set is ~94k files; converting on the
fly keeps disk use around 1 GB instead of ~50 GB.

    python scripts/download_kaggle_asl.py --only-wlasl-overlap   # 39 shared signs first
    python scripts/download_kaggle_asl.py                        # everything
"""
import argparse
import io
import sys
import urllib.parse
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from echogesture.kaggle_asl import parquet_to_array  # noqa: E402
from scripts.download_kaggle_wlasl import credentials  # noqa: E402

BASE = "https://www.kaggle.com/api/v1/competitions/data/download/asl-signs/"


def fetch(path, out_dir, session):
    out = out_dir / f"{Path(path).stem}.npz"
    if out.exists():
        return "exists"
    for _ in range(3):
        try:
            r = session.get(BASE + urllib.parse.quote(path, safe=""), timeout=60)
            if r.status_code == 429:
                continue
            r.raise_for_status()
            data = r.content
            if data[:2] == b"PK":  # Kaggle zips single files
                with zipfile.ZipFile(io.BytesIO(data)) as z:
                    data = z.read(z.namelist()[0])
            xyz = parquet_to_array(io.BytesIO(data))
            np.savez_compressed(out, xyz=xyz.astype(np.float16))
            return "ok"
        except Exception:
            continue
    return "fail"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/kaggle_asl")
    ap.add_argument("--wlasl-manifest", default="data/wlasl/manifest.csv")
    ap.add_argument("--only-wlasl-overlap", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    root = Path(args.root)
    out_dir = root / "landmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(root / "train.csv")
    wlasl = set(pd.read_csv(args.wlasl_manifest).gloss)
    overlap = train.sign.str.upper().isin(wlasl)
    if args.only_wlasl_overlap:
        train = train[overlap]
    else:  # shared signs first, so a partial download is still the most useful part
        train = pd.concat([train[overlap], train[~overlap]])
    if args.limit:
        train = train.head(args.limit)

    session = requests.Session()
    session.auth = credentials()
    with ThreadPoolExecutor(args.workers) as pool:
        res = list(tqdm(pool.map(lambda p: fetch(p, out_dir, session), train.path), total=len(train)))
    print(pd.Series(res).value_counts().to_string())

    have = {p.stem for p in out_dir.glob("*.npz")}
    m = pd.read_csv(root / "train.csv")
    m = m[m.sequence_id.astype(str).isin(have)].rename(columns={"participant_id": "signer_id", "sign": "gloss"})
    m["video_id"] = m.sequence_id.astype(str)
    m["gloss"] = m.gloss.str.upper()
    m.to_csv(root / "manifest.csv", index=False)
    print(f"manifest: {len(m)} sequences, {m.gloss.nunique()} signs, {m.signer_id.nunique()} signers")


if __name__ == "__main__":
    main()
