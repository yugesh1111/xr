"""Raw landmarks -> the fixed-size tensor the model sees.

Used by training, export calibration and predict(), so the three can't drift
apart.
"""
import numpy as np

from . import layout as L
from .features import GROUPS, clip_features
from .sampling import adaptive_indices, sample_clip

# upper body only: legs carry no sign information and are often out of frame
RAW_POINTS = np.r_[0:25, L.LHAND.start:L.FACE.stop]


def raw_input(xyz, length=32):
    """E1 baseline input: landmarks centred and scaled per clip, nothing else."""
    p = np.asarray(xyz, dtype=np.float32)[:, RAW_POINTS]
    center = np.nanmean(p.reshape(-1, 3), axis=0)
    scale = np.nanstd(p[..., :2]) + 1e-6
    p = np.nan_to_num((p - center) / scale).reshape(len(p), -1)
    idx = adaptive_indices(np.ones(len(p)), length, uniform_weight=1.0)
    return p[idx]


def canonical_input(xyz, length=32, uniform_weight=0.5):
    f = sample_clip(clip_features(xyz), length, uniform_weight)
    return np.concatenate([f[g] for g in GROUPS], axis=1)


def make_input(xyz, mode="canonical", length=32, uniform_weight=0.5):
    if len(xyz) == 0:
        raise ValueError("clip has no frames")
    if mode == "raw":
        return raw_input(xyz, length)
    return canonical_input(xyz, length, uniform_weight)


def group_slices(mode="canonical"):
    """{group: (start, stop)} column ranges inside the model input."""
    if mode == "raw":
        return {"raw": (0, len(RAW_POINTS) * 3)}
    from .features import group_dims
    out, start = {}, 0
    for g, d in group_dims().items():
        out[g] = (start, start + d)
        start += d
    return out
