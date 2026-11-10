"""Demo GIF: test clips drawn as the landmarks the model sees (no video pixels,
so no faces), with the word the shipped INT8 model predicts.

    python scripts/make_demo_gif.py --ids 06335 06360 06845 --out docs/media/pipeline.gif
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from echogesture import layout as L  # noqa: E402
from echogesture.inputs import make_input  # noqa: E402

POSE_EDGES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24)]
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
              (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
W, H = 360, 400


def predict(sess, meta, xyz):
    z = sess.run(None, {"x": make_input(xyz, meta["mode"], meta["length"])[None].astype(np.float32)})[0][0]
    p = np.exp(z - z.max())
    p /= p.sum()
    i = int(p.argmax())
    return meta["vocab"][i], float(p[i])


def draw(xyz, box, caption=None, truth=""):
    img = np.full((H, W, 3), (24, 24, 28), np.uint8)
    x0, y0, s = box

    def pt(p):
        return int((p[0] - x0) * s), int((p[1] - y0) * s)

    def lines(block, edges, color, width):
        pts = xyz[block]
        for a, b in edges:
            if not (np.isnan(pts[a]).any() or np.isnan(pts[b]).any()):
                cv2.line(img, pt(pts[a]), pt(pts[b]), color, width, cv2.LINE_AA)
        for p in pts:
            if not np.isnan(p).any():
                cv2.circle(img, pt(p), 2, color, -1, cv2.LINE_AA)

    lines(L.POSE, POSE_EDGES, (150, 150, 160), 3)
    f = xyz[L.FACE]
    if not np.isnan(f[[L.F_FOREHEAD, L.F_CHIN]]).any():
        c = (f[L.F_FOREHEAD] + f[L.F_CHIN]) / 2
        r = np.linalg.norm(f[L.F_CHIN, :2] - f[L.F_FOREHEAD, :2]) * s * 0.62
        cv2.ellipse(img, pt(c), (int(r * 0.8), int(r)), 0, 0, 360, (150, 150, 160), 2, cv2.LINE_AA)
    for p in f:
        if not np.isnan(p).any():
            cv2.circle(img, pt(p), 2, (200, 200, 200), -1, cv2.LINE_AA)
    lines(L.LHAND, HAND_EDGES, (90, 200, 255), 2)
    lines(L.RHAND, HAND_EDGES, (255, 170, 80), 2)
    cv2.putText(img, f"signed: {truth}", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (170, 170, 170), 1, cv2.LINE_AA)
    if caption:
        word, prob = caption
        cv2.rectangle(img, (0, H - 64), (W, H), (0, 0, 0), -1)
        cv2.putText(img, word, (16, H - 22), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, f"{100 * prob:.0f}%", (W - 80, H - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (120, 230, 140), 2, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="+", required=True)
    ap.add_argument("--landmarks", default="data/wlasl/landmarks")
    ap.add_argument("--manifest", default="data/wlasl/manifest.csv")
    ap.add_argument("--models", default="models")
    ap.add_argument("--out", default="docs/media/pipeline.gif")
    args = ap.parse_args()

    import onnxruntime as ort
    import pandas as pd
    meta = json.loads((Path(args.models) / "model_meta.json").read_text())
    sess = ort.InferenceSession(str(Path(args.models) / "model_int8.onnx"), providers=["CPUExecutionProvider"])
    gloss = dict(pd.read_csv(args.manifest, dtype={"video_id": str})[["video_id", "gloss"]].values)

    frames = []
    for vid in args.ids:
        xyz = np.load(Path(args.landmarks) / f"{vid}.npz")["xyz"].astype(np.float32)
        cap = predict(sess, meta, xyz)
        upper = np.concatenate([xyz[:, L.FACE], xyz[:, L.LHAND], xyz[:, L.RHAND], xyz[:, 11:17]], 1).reshape(-1, 3)
        upper = upper[~np.isnan(upper).any(1)]
        lo, hi = np.percentile(upper[:, :2], 1, 0), np.percentile(upper[:, :2], 99, 0)
        span = max(hi[0] - lo[0], (hi[1] - lo[1]) * W / (H - 70)) * 1.25
        cx, cy = (lo + hi) / 2
        s = W / span
        box = (cx - W / s / 2, cy - (H - 70) / s / 2 - 0.02, s)
        for f in xyz[::2]:
            frames.append(draw(f, box, truth=gloss.get(vid, "")))
        frames += [draw(xyz[-1], box, cap, gloss.get(vid, ""))] * 12
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    imgs = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
    imgs[0].save(args.out, save_all=True, append_images=imgs[1:], duration=80, loop=0, optimize=True)
    print(f"{len(frames)} frames -> {args.out} ({Path(args.out).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
