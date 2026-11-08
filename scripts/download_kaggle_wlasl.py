"""Download the WLASL video mirror from Kaggle (~5.4 GB zip).

Dataset: https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed

Credentials (never commit them):
  - env vars KAGGLE_USERNAME and KAGGLE_KEY, or
  - ~/.kaggle/kaggle.json  ({"username": "...", "key": "..."})
Get a key at kaggle.com -> Settings -> API -> Create New Token.

    python scripts/download_kaggle_wlasl.py
    python scripts/unpack_kaggle_wlasl.py
"""
import argparse
import json
import os
from pathlib import Path

import requests
from tqdm import tqdm

URL = "https://www.kaggle.com/api/v1/datasets/download/risangbaskoro/wlasl-processed"


def credentials():
    user, key = os.environ.get("KAGGLE_USERNAME"), os.environ.get("KAGGLE_KEY")
    cfg = Path.home() / ".kaggle" / "kaggle.json"
    if not (user and key) and cfg.exists():
        c = json.loads(cfg.read_text())
        user, key = c["username"], c["key"]
    return (user, key) if user and key else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/kaggle_wlasl/archive.zip")
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    auth = credentials()
    if auth is None:
        print("no Kaggle credentials found; trying without (works only if a proxy adds them)")
    with requests.get(URL, auth=auth, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(out, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
