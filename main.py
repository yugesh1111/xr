"""Submission entry point.

    from main import predict
    predict("clips/test_0412.mp4")
    # {"clip_id": "test_0412", "prediction": "HAPPY", "top5": [...], "confidence": 0.81, "latency_ms": 23.4}

Batch mode, writes one JSON object per line:
    python main.py --videos path/to/test_clips --out results/predictions.jsonl

Everything runs locally: MediaPipe on the frames, then the INT8 ONNX model.
No network access is needed or used after the models are on disk.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from echogesture.extract import extract_video
from echogesture.inputs import make_input

MODEL_DIR = Path(os.environ.get("ECHOGESTURE_MODEL_DIR", Path(__file__).resolve().parent / "models"))
_session, _meta = None, None


def _load():
    global _session, _meta
    if _session is None:
        import onnxruntime as ort
        _meta = json.loads((MODEL_DIR / "model_meta.json").read_text())
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        _session = ort.InferenceSession(str(MODEL_DIR / "model_int8.onnx"), so, providers=["CPUExecutionProvider"])
    return _session, _meta


def _softmax(z):
    z = np.exp(z - z.max())
    return z / z.sum()


def predict(video_path: str) -> dict:
    """Return the predicted gloss/word for one video clip. Must run fully on-device."""
    sess, meta = _load()
    clip_id = Path(video_path).stem
    xyz = extract_video(video_path)
    if len(xyz) == 0:
        return {"clip_id": clip_id, "prediction": meta["vocab"][0], "top5": meta["vocab"][:5],
                "confidence": 0.0, "latency_ms": 0.0}
    t = time.perf_counter()
    x = make_input(xyz, meta["mode"], meta["length"])[None].astype(np.float32)
    p = _softmax(sess.run(None, {"x": x})[0][0])
    latency = (time.perf_counter() - t) * 1000  # features + model; landmark extraction is timed separately
    top = np.argsort(-p)[:5]
    return {"clip_id": clip_id, "prediction": meta["vocab"][top[0]],
            "top5": [meta["vocab"][i] for i in top], "confidence": round(float(p[top[0]]), 4),
            "latency_ms": round(latency, 2)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, help="folder of clips, or a text file with one path per line")
    ap.add_argument("--out", default="results/predictions.jsonl")
    args = ap.parse_args()
    src = Path(args.videos)
    if src.is_dir():
        paths = sorted(p for p in src.iterdir() if p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm"})
    else:
        paths = [Path(line) for line in src.read_text().split() if line]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for p in paths:
            f.write(json.dumps(predict(str(p))) + "\n")
    print(f"{len(paths)} predictions -> {args.out}")
