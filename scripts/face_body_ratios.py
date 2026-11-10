"""Measure where the shoulders sit relative to the face, in face heights.

Prints the medians to put in echogesture/features.py (SHOULDER_DROP,
SHOULDER_HALF). Uses training clips with both shoulders in view.

    python scripts/face_body_ratios.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from echogesture import layout as L  # noqa: E402
from echogesture.features import shoulders_in_frame  # noqa: E402

df = pd.read_csv("data/wlasl/manifest.csv", dtype={"video_id": str})
df = df[df.split == "train"]
drop, half, lateral = [], [], []
for v in df.video_id:
    p = Path(f"data/wlasl/landmarks/{v}.npz")
    if not p.exists():
        continue
    x = np.load(p)["xyz"]
    ok = shoulders_in_frame(x)
    f = x[:, L.FACE]
    ok &= ~np.isnan(f[:, [1, 2, 3, 4], :2]).any(axis=(1, 2))
    if ok.sum() < 5:
        continue
    x, f = x[ok], f[ok]
    er, el = f[:, 3, :2], f[:, 4, :2]
    fh = np.linalg.norm(f[:, L.F_FOREHEAD, :2] - f[:, L.F_CHIN, :2], axis=-1)
    u = (el - er) / np.linalg.norm(el - er, axis=-1, keepdims=True)
    down = np.stack([-u[:, 1], u[:, 0]], 1)
    down *= np.sign(down[:, 1:2])
    mid = (x[:, L.L_SHOULDER, :2] + x[:, L.R_SHOULDER, :2]) / 2 - (el + er) / 2
    drop.append(np.median((mid * down).sum(1) / fh))
    lateral.append(np.median((mid * u).sum(1) / fh))
    half.append(np.median(np.linalg.norm(x[:, L.L_SHOULDER, :2] - x[:, L.R_SHOULDER, :2], axis=-1) / 2 / fh))
print(f"{len(drop)} clips")
for name, v in (("SHOULDER_DROP", drop), ("SHOULDER_HALF", half), ("lateral offset", lateral)):
    v = np.array(v)
    print(f"{name:15s} median {np.median(v):.3f}  IQR {np.percentile(v, 25):.3f}-{np.percentile(v, 75):.3f}")
