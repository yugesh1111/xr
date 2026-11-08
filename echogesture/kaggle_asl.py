"""Convert Kaggle "Google - Isolated Sign Language Recognition" (ASL Signs)
parquet files into our (T, 88, 3) layout.

Each parquet has long-format rows: frame, type (face/left_hand/pose/right_hand),
landmark_index, x, y, z. Those landmarks were made with MediaPipe Holistic,
the same model family we use, so the two sources line up.

Note: the Kaggle videos' aspect ratio is unknown, so x/z are not rescaled the
way our extractor does. The canonicalizer's shoulder-width scaling absorbs
most of that; pass `aspect` if you know it.
"""
import numpy as np
import pandas as pd

from . import layout as L

_OFFSET = {"pose": 0, "left_hand": 33, "right_hand": 54}
_FACE_MAP = {f: 75 + i for i, f in enumerate(L.FACE_IDX)}


def parquet_to_array(path, aspect=1.0):
    df = pd.read_parquet(path, columns=["frame", "type", "landmark_index", "x", "y", "z"])
    frames = np.sort(df.frame.unique())
    t_of = {f: i for i, f in enumerate(frames)}
    out = np.full((len(frames), L.N_POINTS, 3), np.nan, dtype=np.float32)

    face = df[df.type == "face"]
    face = face[face.landmark_index.isin(_FACE_MAP)]
    rest = df[df.type.isin(_OFFSET)]
    for part, col in ((rest, rest.type.map(_OFFSET) + rest.landmark_index),
                      (face, face.landmark_index.map(_FACE_MAP))):
        t = part.frame.map(t_of).to_numpy()
        n = col.to_numpy().astype(int)
        out[t, n] = part[["x", "y", "z"]].to_numpy(dtype=np.float32)
    out[..., [0, 2]] *= aspect
    return out


# eye width / face height measured on aspect-corrected WLASL landmarks
FACE_RATIO_REF = 0.499
DEFAULT_ASPECT = 0.695  # median over Kaggle clips, used when no face is found


def estimate_aspect(xyz):
    """Width/height of the source video, estimated from face proportions.
    Kaggle coordinates are normalised by width and height separately, so x
    must be multiplied by this to match our extractor (which does it at
    extraction time)."""
    F = xyz[:, L.FACE]
    eye = np.abs(F[:, 4, 0] - F[:, 3, 0])
    h = np.abs(F[:, 2, 1] - F[:, 1, 1])
    ok = ~(np.isnan(eye) | np.isnan(h)) & (h > 0)
    if ok.sum() < 3:
        return DEFAULT_ASPECT
    return float(np.clip(FACE_RATIO_REF / np.median(eye[ok] / h[ok]), 0.45, 1.1))


def fix_aspect(xyz):
    out = np.asarray(xyz, dtype=np.float32).copy()
    out[..., [0, 2]] *= estimate_aspect(out)
    return out
