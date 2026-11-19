"""Built-in ISL test for the app: 10 INCLUDE-50 test clips (10 different words),
re-encoded small (480p H.264) into data/include50/test_clips/, and
models/isl_test_clips.json with each clip's true word and what the Python
pipeline (the exported INT8 model, like main.py) predicts for it, so the app can
show "same answer as the computer".

    python scripts/make_isl_test_clips.py --model models/isl_model_int8.onnx
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from echogesture.data import ClipDataset  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=str(ROOT / "data" / "include50"))
ap.add_argument("--model", default=str(ROOT / "models" / "isl_model_int8.onnx"))
ap.add_argument("--n", type=int, default=10)
ap.add_argument("--out", default=str(ROOT / "models" / "isl_test_clips.json"))
a = ap.parse_args()

root = Path(a.root)
meta = json.loads(Path(a.model).with_name(Path(a.model).name.replace("model_int8.onnx", "model_meta.json")).read_text())
vocab = meta["vocab"]
df = pd.read_csv(root / "manifest.csv", dtype={"video_id": str})
df = df[(df.split == "test") & np.array([(root / "landmarks" / f"{v}.npz").exists() for v in df.video_id])]
words = sorted(df.gloss.unique())
pick_words = list(np.random.default_rng(0).choice(words, min(a.n, len(words)), replace=False))
rows = [df[df.gloss == w].sample(1, random_state=0).iloc[0] for w in sorted(pick_words)]
sess = ort.InferenceSession(a.model, providers=["CPUExecutionProvider"])
out_dir = root / "test_clips"
out_dir.mkdir(exist_ok=True)
clips = []
for r in rows:
    ds = ClipDataset([root / "landmarks" / f"{r.video_id}.npz"], [0], meta["mode"], meta["length"])
    logits = sess.run(None, {"x": ds[0][0].numpy()[None]})[0][0]
    p = np.exp(logits - logits.max()); p /= p.sum()
    src = next((root / "raw").glob(f"{r.video_id}.*"))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vf", "scale=-2:480", "-an", "-c:v", "libx264",
                    "-crf", "26", "-preset", "medium", "-pix_fmt", "yuv420p", str(out_dir / f"{r.video_id}.mp4")], check=True)
    clips.append({"video_id": r.video_id, "gloss": r.gloss, "python": vocab[int(p.argmax())],
                  "python_conf": round(float(p.max()), 4)})
    print(f"{r.gloss:12s} -> {clips[-1]['python']:12s} {p.max():.2f}")
Path(a.out).write_text(json.dumps(clips, indent=1))
print(f"{sum(c['gloss'] == c['python'] for c in clips)} of {len(clips)} right; {a.out}")
