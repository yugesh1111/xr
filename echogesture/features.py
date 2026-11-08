"""Landmarks -> signer-invariant, factorized features.

Pipeline for one clip (T, 88, 3):
  1. fix_hand_slots   – put each hand next to the pose wrist it belongs to
  2. canonicalize     – body frame: origin at the chest, unit = shoulder width,
                        shoulders level and facing the camera
  3. mirror to a right-dominant convention (left-handed signers)
  4. fill short tracking gaps, remembering which values are filled in
  5. compute four articulation groups + a quality group:
       handshape   – finger configuration in the hand's own frame
       orientation – palm normal and finger direction in the body frame
       location    – where the hands are relative to body and face
       motion      – velocities in the body frame
       quality     – which hands were observed / interpolated
"""
import numpy as np

from . import layout as L

GROUPS = ("handshape", "orientation", "location", "motion", "nonmanual", "quality")


# ---------------------------------------------------------------- geometry

def _unit(v, eps=1e-8):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + eps)


def _rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def _rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def fix_hand_slots(xyz):
    """Swap the two hand blocks in frames where the 'left' hand sits closer to
    the pose right wrist (and vice versa). Handedness classifiers flip labels
    on rotated or mirrored views; the pose skeleton is the steadier reference."""
    xyz = xyz.copy()
    lw, rw = xyz[:, L.L_WRIST, :2], xyz[:, L.R_WRIST, :2]
    lh, rh = xyz[:, L.LHAND.start, :2], xyz[:, L.RHAND.start, :2]

    def d(a, b):
        return np.linalg.norm(a - b, axis=-1)

    keep = np.nansum([d(lh, lw), d(rh, rw)], axis=0)
    swap = np.nansum([d(lh, rw), d(rh, lw)], axis=0)
    # only decide when we actually have evidence
    known = ~np.isnan(lw).any(1) & ~np.isnan(rw).any(1) & ~(np.isnan(lh).any(1) & np.isnan(rh).any(1))
    flip = known & (swap < keep)
    if flip.any():
        l, r = xyz[flip, L.LHAND].copy(), xyz[flip, L.RHAND].copy()
        xyz[flip, L.LHAND], xyz[flip, L.RHAND] = r, l
    return xyz



def canonicalize(xyz, use_yaw=True, max_yaw_deg=60.0):
    """Express all points in a clip-level body frame.

    Frame is estimated once per clip (median over time) so that real body
    movement inside the sign - leaning, head turns - is preserved.
    x: signer's right -> left as seen by the camera, y: up, z: towards camera.
    """
    pts = xyz.astype(np.float64)
    pts[..., 1] *= -1  # image y points down
    ls, rs = pts[:, L.L_SHOULDER], pts[:, L.R_SHOULDER]
    if np.isnan(ls).all() or np.isnan(rs).all():
        # no body: fall back to centring on whatever we have
        center = np.nanmean(pts.reshape(-1, 3), axis=0)
        scale = np.nanstd(pts[..., :2]) * 4 + 1e-6
        return ((pts - center) / scale).astype(np.float32)

    l, r = np.nanmedian(ls, axis=0), np.nanmedian(rs, axis=0)
    center = (l + r) / 2
    across = l - r  # points from the right shoulder to the left one
    out = pts - center
    rot = _rot_z(-np.arctan2(across[1], across[0]))
    if use_yaw:
        a = rot @ across
        yaw = np.clip(np.arctan2(a[2], a[0]), -np.radians(max_yaw_deg), np.radians(max_yaw_deg))
        rot = _rot_y(yaw) @ rot
    out = out @ rot.T
    scale = np.linalg.norm(across[:2]) + 1e-6
    return (out / scale).astype(np.float32)


def motion_energy(xyz, block):
    p = xyz[:, block]
    v = np.linalg.norm(np.diff(p, axis=0), axis=-1)
    return float(np.nansum(v)) + 0.05 * float((~np.isnan(p[:, 0, 0])).sum())


def to_right_dominant(xyz):
    """Mirror left-handed signing so the dominant hand is always in RHAND.
    In ASL the mirrored sign is the same sign, so this is label-preserving."""
    if motion_energy(xyz, L.LHAND) <= motion_energy(xyz, L.RHAND):
        return xyz, False
    m = xyz[:, L.mirror_permutation()].copy()
    m[..., 0] *= -1
    return m, True


def fill_gaps(xyz, max_gap=6):
    """Linearly interpolate per-point gaps of at most `max_gap` frames.
    Returns (filled, interpolated_mask[T, N]). Longer gaps stay NaN: making up
    long stretches of motion is worse than admitting we didn't see it."""
    xyz = xyz.copy()
    T, N, _ = xyz.shape
    interp = np.zeros((T, N), dtype=bool)
    seen = ~np.isnan(xyz[..., 0])
    for n in range(N):
        idx = np.flatnonzero(seen[:, n])
        if len(idx) < 2:
            continue
        for a, b in zip(idx[:-1], idx[1:]):
            gap = b - a - 1
            if 0 < gap <= max_gap:
                w = (np.arange(1, gap + 1) / (gap + 1))[:, None]
                xyz[a + 1:b, n] = xyz[a, n] * (1 - w) + xyz[b, n] * w
                interp[a + 1:b, n] = True
    return xyz, interp


# ---------------------------------------------------------------- per-hand features

def _palm_frame(h):
    """Rotation matrix (T, 3, 3) whose rows are the hand's x, y, z axes."""
    y = _unit(h[:, L.MIDDLE_MCP] - h[:, L.WRIST])
    across = h[:, L.INDEX_MCP] - h[:, L.PINKY_MCP]
    z = _unit(np.cross(across, y))  # palm normal
    x = np.cross(y, z)
    return np.stack([x, y, z], axis=1)


def _angles(h):
    out = []
    for a, b, c in L.FINGER_JOINTS:
        u, v = _unit(h[:, a] - h[:, b]), _unit(h[:, c] - h[:, b])
        out.append(np.arccos(np.clip((u * v).sum(-1), -1, 1)))
    return np.stack(out, axis=1) / np.pi


def hand_features(h, mirror_x=False):
    """h: (T, 21, 3) canonical coords. Returns handshape (T, 85), orientation (T, 6)."""
    R = _palm_frame(h)
    size = np.linalg.norm(h[:, L.MIDDLE_MCP] - h[:, L.WRIST], axis=-1)[:, None, None] + 1e-6
    local = np.einsum("tij,tnj->tni", R, h[:, 1:] - h[:, :1]) / size
    if mirror_x:  # make the left (non-dominant) hand look like a right hand
        local[..., 0] *= -1
    tips = h[:, L.TIPS]
    dist = np.linalg.norm(tips[:, :, None] - tips[:, None], axis=-1)
    iu = np.triu_indices(5, 1)
    dist = dist[:, iu[0], iu[1]] / size[:, :, 0]
    shape = np.concatenate([local.reshape(len(h), -1), dist, _angles(h)], axis=1)
    orient = np.concatenate([R[:, 2], R[:, 1]], axis=1)  # palm normal, finger direction
    return shape, orient


def _vel(p):
    v = np.zeros_like(p)
    v[1:] = np.diff(p, axis=0)
    return v


# ---------------------------------------------------------------- main entry

def clip_features(xyz, max_gap=6):
    """(T, 88, 3) raw landmarks -> dict of (T, D) float32 arrays with no NaNs."""
    xyz = np.asarray(xyz, dtype=np.float32)
    xyz = fix_hand_slots(xyz)
    xyz = canonicalize(xyz)
    xyz, mirrored = to_right_dominant(xyz)
    xyz, interp = fill_gaps(xyz, max_gap)

    T = len(xyz)
    nose = xyz[:, L.FACE.start + L.F_NOSE]
    nose = np.where(np.isnan(nose), xyz[:, L.NOSE], nose)
    mouth = (xyz[:, L.FACE.start + L.F_UPPER_LIP] + xyz[:, L.FACE.start + L.F_LOWER_LIP]) / 2

    groups = {g: [] for g in GROUPS}
    present = []
    for block, mirror_x in ((L.RHAND, False), (L.LHAND, True)):  # dominant first
        h = xyz[:, block]
        ok = ~np.isnan(h).any(axis=(1, 2))
        present.append(ok)
        shape, orient = hand_features(h, mirror_x)
        wrist = h[:, L.WRIST]
        tips_rel = h[:, L.TIPS].mean(1) - wrist
        groups["handshape"].append(shape)
        groups["orientation"].append(orient)
        groups["location"].append(np.concatenate([
            wrist, wrist - nose, wrist - mouth,
            np.linalg.norm(wrist - nose, axis=-1, keepdims=True)], axis=1))
        v = _vel(wrist)
        groups["motion"].append(np.concatenate([
            v, np.linalg.norm(v, axis=-1, keepdims=True),
            np.linalg.norm(_vel(v), axis=-1, keepdims=True),
            _vel(tips_rel)], axis=1))
        groups["quality"].append(np.stack([ok, interp[:, block].any(1)], axis=1))

    dom, non = xyz[:, L.RHAND.start], xyz[:, L.LHAND.start]
    groups["location"].append(np.concatenate([dom - non, np.linalg.norm(dom - non, axis=-1, keepdims=True)], 1))
    groups["nonmanual"].append(np.concatenate([
        xyz[:, L.L_ELBOW], xyz[:, L.R_ELBOW], nose,
        np.linalg.norm(xyz[:, L.FACE.start + L.F_UPPER_LIP] - xyz[:, L.FACE.start + L.F_LOWER_LIP], axis=-1)[:, None],
        np.linalg.norm(xyz[:, L.FACE.start + L.F_MOUTH_L] - xyz[:, L.FACE.start + L.F_MOUTH_R], axis=-1)[:, None],
    ], axis=1))
    pose_ok = ~np.isnan(xyz[:, [L.L_SHOULDER, L.R_SHOULDER]]).any(axis=(1, 2))
    groups["quality"].append(np.stack([pose_ok, np.full(T, mirrored)], axis=1))

    out = {}
    for g, parts in groups.items():
        a = np.concatenate(parts, axis=1).astype(np.float32)
        out[g] = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def group_dims():
    """Feature size of each group (computed from a dummy clip)."""
    f = clip_features(np.random.default_rng(0).normal(size=(4, L.N_POINTS, 3)).astype(np.float32))
    return {g: a.shape[1] for g, a in f.items()}
