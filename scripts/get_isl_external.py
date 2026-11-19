"""External ISL test set: short single-word YouTube videos of the INCLUDE-50 words
by people who are not in INCLUDE (found by title, list in yt_candidates.json).
Used ONLY for testing. Writes data/isl_external/raw/<id>.mp4 and manifest.csv
(split = test), then: python scripts/extract_landmarks.py --root data/isl_external

    python scripts/get_isl_external.py --candidates yt_candidates.json --per-word 3
"""
import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--candidates", required=True)
ap.add_argument("--per-word", type=int, default=3)
ap.add_argument("--out", default=str(ROOT / "data" / "isl_external"))
ap.add_argument("--ytdlp", default="yt-dlp")
a = ap.parse_args()
out = Path(a.out)
(out / "raw").mkdir(parents=True, exist_ok=True)
jobs = [(w, h) for w, hits in json.load(open(a.candidates)).items() for h in hits[:a.per_word]]


def get(job):
    w, h = job
    dst = out / "raw" / f"yt_{h['id']}.mp4"
    if not dst.exists():
        subprocess.run([a.ytdlp, "-q", "--no-warnings", "-f", "bv*[height<=720][ext=mp4]/b[height<=720]/bv*+ba/b",
                        "--merge-output-format", "mp4", "--retries", "20", "-o", str(dst), f"https://www.youtube.com/watch?v={h['id']}"])
    return {"video_id": f"yt_{h['id']}", "gloss": w, "split": "test", "frame_start": 1, "frame_end": -1,
            "source": "youtube", "signer_id": "youtube", "title": h["title"], "url": f"https://youtu.be/{h['id']}"} if dst.exists() else None


with ThreadPoolExecutor(8) as ex:
    rows = [r for r in ex.map(get, jobs) if r]
pd.DataFrame(rows).to_csv(out / "manifest.csv", index=False)
print(f"{len(rows)} of {len(jobs)} clips, {len({r['gloss'] for r in rows})} words -> {out / 'manifest.csv'}")
