"""Training-time augmentation on raw landmarks (image coords, y down).

Everything here is something the test videos can actually do to us:
a different camera angle, a different body, faster signing, a hand that
leaves the frame or a tracker that gives up for a few frames in poor light.
"""
import numpy as np

from . import layout as L


def _rotation(yaw, pitch, roll):
    cy, sy, cp, sp, cr, sr = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return rz @ rx @ ry


def rotate_view(x, rng, max_yaw=25, max_pitch=10, max_roll=10):
    c = np.nanmean(x[:, [L.L_SHOULDER, L.R_SHOULDER]].reshape(-1, 3), axis=0)
    if np.isnan(c).any():
        c = np.nanmean(x.reshape(-1, 3), axis=0)
    R = _rotation(*np.radians(rng.uniform(-1, 1, 3) * [max_yaw, max_pitch, max_roll]))
    return ((x - c) @ R.T + c).astype(np.float32)


def vary_anatomy(x, rng):
    """Different hand sizes and arm lengths, keeping the articulation."""
    x = x.copy()
    for hand, wrist, elbow, shoulder in ((L.LHAND, L.L_WRIST, L.L_ELBOW, L.L_SHOULDER),
                                         (L.RHAND, L.R_WRIST, L.R_ELBOW, L.R_SHOULDER)):
        h = x[:, hand]
        w = h[:, :1]
        x[:, hand] = w + (h - w) * rng.uniform(0.85, 1.15)
        arm = rng.uniform(0.9, 1.1) - 1
        sh = x[:, shoulder:shoulder + 1]
        shift = (x[:, wrist:wrist + 1] - sh) * arm
        x[:, hand] += shift
        x[:, wrist:wrist + 1] += shift
        x[:, elbow:elbow + 1] += (x[:, elbow:elbow + 1] - sh) * arm
    x[..., 0] *= rng.uniform(0.9, 1.1)  # camera aspect / lens differences
    return x


def change_speed(x, rng, lo=0.8, hi=1.25):
    T = len(x)
    n = max(int(round(T / rng.uniform(lo, hi))), 2)
    return x[np.round(np.linspace(0, T - 1, n)).astype(int)]


def occlude(x, rng, p_hand=0.3, p_frames=0.2, max_span=0.4):
    x = x.copy()
    T = len(x)
    if rng.random() < p_hand and T > 4:  # one hand lost for a stretch
        span = int(T * rng.uniform(0.05, max_span))
        start = rng.integers(0, T - span + 1)
        x[start:start + span, rng.choice([L.LHAND, L.RHAND])] = np.nan
    if rng.random() < p_frames:  # detector drops whole frames
        x[rng.random(T) < 0.1] = np.nan
    return x


def jitter(x, rng):
    x = x + rng.normal(0, rng.uniform(0.001, 0.005), x.shape).astype(np.float32)
    x[:, L.LHAND.start:L.FACE.start] += rng.normal(0, 0.002, x[:, L.LHAND.start:L.FACE.start].shape)
    return x


# "strong" targets the weak spots found by scripts/robustness.py:
# signers turned sideways and a hand lost for a long stretch.
STRENGTH = {
    "default": {"max_yaw": 25, "p_hand": 0.3, "max_span": 0.4},
    "strong": {"max_yaw": 50, "p_hand": 0.5, "max_span": 0.5},
}


def augment(xyz, rng, strength="default"):
    cfg = STRENGTH[strength]
    x = np.asarray(xyz, dtype=np.float32)
    x = rotate_view(x, rng, max_yaw=cfg["max_yaw"])
    x = vary_anatomy(x, rng)
    x = change_speed(x, rng)
    x = occlude(x, rng, p_hand=cfg["p_hand"], max_span=cfg["max_span"])
    return jitter(x, rng).astype(np.float32)
