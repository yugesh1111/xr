"""Download the raw WLASL videos listed in the manifest (resumable).

Expect a fair number of dead links - WLASL is scraped from many sites and
some are gone. The script logs failures to data/wlasl/failed.txt; ask the
organisers for their copy of the videos if coverage is too low.

    python scripts/download_wlasl.py --workers 8
"""
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="data/wlasl")
ap.add_argument("--workers", type=int, default=8)
args = ap.parse_args()

root = Path(args.root)
raw = root / "raw"
raw.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(root / "manifest.csv", dtype={"video_id": str})
UA = {"User-Agent": "Mozilla/5.0"}


def fetch(row):
    vid, url = row.video_id, row.url
    if any(raw.glob(f"{vid}.*")):
        return vid, "exists"
    try:
        if "youtube.com" in url or "youtu.be" in url:
            # YouTube entries are full videos; extraction trims them with frame_start/frame_end
            r = subprocess.run(["yt-dlp", "-q", "-f", "mp4/best", "-o", str(raw / f"{vid}.%(ext)s"), url],
                               capture_output=True, timeout=300)
            return vid, "ok" if r.returncode == 0 else "fail"
        if url.endswith(".swf"):
            return vid, "fail"  # flash videos, not worth converting
        r = requests.get(url, headers=UA, timeout=30)
        r.raise_for_status()
        ext = Path(url.split("?")[0]).suffix or ".mp4"
        (raw / f"{vid}{ext}").write_bytes(r.content)
        return vid, "ok"
    except Exception:
        return vid, "fail"


rows = list(df.drop_duplicates("video_id").itertuples())
with ThreadPoolExecutor(args.workers) as pool:
    results = list(tqdm(pool.map(fetch, rows), total=len(rows)))
failed = [v for v, s in results if s == "fail"]
(root / "failed.txt").write_text("\n".join(failed))
print(f"ok/existing: {len(rows) - len(failed)}, failed: {len(failed)} (see {root / 'failed.txt'})")
