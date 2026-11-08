"""Motion-adaptive temporal sampling.

A fixed-length model input (e.g. 32 frames) is built by spending more samples
where the hands move and fewer on holds and the rest pose at the start/end.
"""
import numpy as np


def active_span(present, pad=2):
    """First/last frame (inclusive) where any hand is visible, padded."""
    idx = np.flatnonzero(present)
    if len(idx) == 0:
        return 0, len(present) - 1
    return max(idx[0] - pad, 0), min(idx[-1] + pad, len(present) - 1)


def adaptive_indices(speed, length, uniform_weight=0.5):
    """Pick `length` frame indices; density follows hand speed.

    uniform_weight=1 gives plain uniform sampling, 0 follows motion only.
    Short clips simply repeat frames."""
    T = len(speed)
    if T == 0:
        raise ValueError("empty clip")
    s = np.nan_to_num(np.asarray(speed, dtype=np.float64))
    s = s / s.sum() if s.sum() > 0 else np.full(T, 1.0 / T)
    w = uniform_weight / T + (1 - uniform_weight) * s
    cdf = np.cumsum(w) / w.sum()
    q = (np.arange(length) + 0.5) / length
    return np.clip(np.searchsorted(cdf, q), 0, T - 1)


def sample_clip(features, length=32, uniform_weight=0.5):
    """Apply span trimming + adaptive sampling to a feature dict from
    features.clip_features. Returns a dict of (length, D) arrays."""
    q = features["quality"]
    present = (q[:, 0] > 0) | (q[:, 2] > 0)  # dominant or non-dominant hand seen
    a, b = active_span(present)
    m = features["motion"]
    speed = m[a:b + 1, 3] + m[a:b + 1, 11]  # dominant + non-dominant wrist speed
    idx = a + adaptive_indices(speed, length, uniform_weight)
    return {k: v[idx] for k, v in features.items()}
