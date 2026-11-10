"""Refresh models/test_clips.json: the word main.py predicts for each built-in
app test clip with the model in models/, so the app's "Same as Python" check
compares against the shipped model.

    python scripts/update_test_clips.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from main import predict  # noqa: E402

path = Path("models/test_clips.json")
clips = json.loads(path.read_text())
for c in clips:
    src = next(Path("data/wlasl/raw").glob(f"{c['video_id']}.*"))
    p = predict(str(src))
    c["python"], c["python_conf"] = p["prediction"], p["confidence"]
    print(f"{c['video_id']} {c['gloss']:10s} -> {p['prediction']:10s} {p['confidence']:.2f}")
path.write_text(json.dumps(clips, indent=1) + "\n")
right = sum(c["python"] == c["gloss"] for c in clips)
print(f"{right}/{len(clips)} correct")
