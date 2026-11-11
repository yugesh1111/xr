"""Cut the clips listed in models/sentence_clips.json to their sign (WLASL
frame_start..frame_end; some source videos are long compilations) so the app's
sentence test plays exactly what the Python pipeline evaluates.

    python scripts/cut_sentence_clips.py     # -> data/wlasl/sentence_clips/<id>.mp4
"""
import json
import subprocess
from pathlib import Path

import pandas as pd

root = Path("data/wlasl")
out = root / "sentence_clips"
out.mkdir(exist_ok=True)
m = pd.read_csv(root / "manifest.csv", dtype={"video_id": str}).set_index("video_id")
for sent in json.loads(Path("models/sentence_clips.json").read_text()):
    for c in sent:
        v = c["video_id"]
        a, b = int(m.loc[v, "frame_start"]), int(m.loc[v, "frame_end"])
        sel = f"gte(n\\,{a - 1})" + (f"*lte(n\\,{b - 1})" if b > 0 else "")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(root / "raw" / f"{v}.mp4"),
                        "-vf", f"select='{sel}',setpts=N/FRAME_RATE/TB", "-an", "-c:v", "libx264", "-crf", "18",
                        "-pix_fmt", "yuv420p", str(out / f"{v}.mp4")], check=True)
        print(v, c["gloss"], a, b)
