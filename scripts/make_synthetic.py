"""Synthetic signers for testing the training/export plumbing without data.

NOT for reporting results. Each "sign" is a hand shape + a path relative to
the body; each "signer" has their own size, position, camera roll and
handedness. Writes data/synthetic/{landmarks/*.npz, manifest.csv}.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from echogesture import layout as L  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="data/synthetic")
ap.add_argument("--classes", type=int, default=20)
ap.add_argument("--signers", type=int, default=12)
ap.add_argument("--reps", type=int, default=4)
args = ap.parse_args()
rng = np.random.default_rng(0)


def hand(curl, scale):
    """21 hand points in a right-hand frame: x to thumb side, y to fingers."""
    pts = np.zeros((21, 3))
    bases = [(0.35, 0.3), (0.2, 1.0), (0.0, 1.05), (-0.18, 1.0), (-0.34, 0.9)]
    for f, (bx, by) in enumerate(bases):
        ang, p = 0.0, np.array([bx, by, 0.0])
        pts[1 + 4 * f] = p
        for j in range(3):
            ang += curl[f] * 0.9
            p = p + 0.35 * np.array([0.6 * bx if f == 0 else 0, np.cos(ang), -np.sin(ang)])
            pts[2 + 4 * f + j] = p
    return pts * scale


signs = [{
    "curl": rng.uniform(0, 1.4, 5),
    "center": rng.uniform([-0.6, -0.3], [0.6, 1.4]),  # in shoulder widths, relative to chest
    "amp": rng.uniform(0.05, 0.4, 2), "freq": rng.integers(1, 3, 2), "phase": rng.uniform(0, 6.3),
    "palm": rng.uniform(-1.2, 1.2),
} for _ in range(args.classes)]
signers = [{
    "width": rng.uniform(0.18, 0.3), "pos": rng.uniform([0.35, 0.55], [0.65, 0.7]),
    "roll": rng.normal(0, 0.08), "left": rng.random() < 0.25, "hand": rng.uniform(0.04, 0.06),
} for _ in range(args.signers)]

out = Path(args.out)
(out / "landmarks").mkdir(parents=True, exist_ok=True)
rows = []
for si, s in enumerate(signers):
    for ci, c in enumerate(signs):
        for r in range(args.reps):
            T = int(rng.integers(30, 70))
            t = np.linspace(0, 2 * np.pi, T)
            x = np.full((T, L.N_POINTS, 3), np.nan)
            body = np.zeros((T, L.N_POINTS, 3))  # body frame: x right->left, y up, unit shoulder width
            body[:, L.L_SHOULDER] = [0.5, 0, 0]
            body[:, L.R_SHOULDER] = [-0.5, 0, 0]
            body[:, L.NOSE] = [0, 0.9, 0.1]
            body[:, L.FACE] = [0, 0.9, 0.1] + rng.normal(0, 0.05, (len(L.FACE_IDX), 3))
            path = np.stack([c["center"][0] + c["amp"][0] * np.sin(c["freq"][0] * t + c["phase"]),
                             c["center"][1] + c["amp"][1] * np.sin(c["freq"][1] * t), 0.2 + 0 * t], 1)
            path += rng.normal(0, 0.02, path.shape)
            hp = hand(c["curl"] + rng.normal(0, 0.05, 5), s["hand"] / s["width"] * 4)
            cr, sr = np.cos(c["palm"]), np.sin(c["palm"])
            hp = hp @ np.array([[cr, 0, sr], [0, 1, 0], [-sr, 0, cr]]).T
            # dominant = signer's right (camera-left, negative x) unless left-handed
            body[:, L.RHAND] = path[:, None] * [-1, 1, 1] + hp * [-1, 1, 1] - [0, 0, 0]
            body[:, L.R_WRIST] = body[:, L.RHAND.start]
            body[:, L.LHAND] = np.array([0.4, -0.8, 0.1]) + hand(np.full(5, 0.3), s["hand"] / s["width"] * 4)
            body[:, L.L_WRIST] = body[:, L.LHAND.start]
            body[:, L.R_ELBOW] = (body[:, L.R_SHOULDER] + body[:, L.R_WRIST]) / 2 + [0, -0.3, 0]
            body[:, L.L_ELBOW] = (body[:, L.L_SHOULDER] + body[:, L.L_WRIST]) / 2 + [0, -0.3, 0]
            if s["left"]:
                body = body[:, L.mirror_permutation()]
                body[..., 0] *= -1
            # body frame -> image: scale, roll, flip y, place
            cr, sr = np.cos(s["roll"]), np.sin(s["roll"])
            img = body * s["width"] @ np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]]).T
            img[..., 1] *= -1
            img[..., :2] += s["pos"]
            # signer's right appears on camera's left: body x already points right->left
            keep = np.zeros(L.N_POINTS, bool)
            keep[[L.NOSE, L.L_SHOULDER, L.R_SHOULDER, L.L_ELBOW, L.R_ELBOW, L.L_WRIST, L.R_WRIST]] = True
            keep[L.LHAND] = keep[L.RHAND] = keep[L.FACE] = True
            x[:, keep] = img[:, keep]
            x[:3, L.LHAND.start:L.FACE.start] = np.nan  # hands enter after a few frames
            vid = f"s{si:02d}c{ci:03d}r{r}"
            np.savez_compressed(out / "landmarks" / f"{vid}.npz", xyz=x.astype(np.float16))
            split = "val" if si >= args.signers - 3 else "train"
            rows.append({"video_id": vid, "gloss": f"SIGN{ci:03d}", "signer_id": si, "split": split})
pd.DataFrame(rows).to_csv(out / "manifest.csv", index=False)
print(f"{len(rows)} clips -> {out}")
