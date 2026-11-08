import numpy as np
import pytest

from echogesture import layout as L
from echogesture.features import clip_features, fill_gaps
from echogesture.sampling import adaptive_indices, sample_clip


def fake_clip(T=40, seed=0):
    """A rough upright signer in image coords (y down): shoulders level,
    right hand (camera left) moving, left hand resting low."""
    rng = np.random.default_rng(seed)
    x = np.full((T, L.N_POINTS, 3), np.nan, dtype=np.float32)
    x[:, :L.N_POSE] = rng.normal([0.5, 0.6, 0.0], 0.05, size=(L.N_POSE, 3))
    x[:, L.L_SHOULDER] = [0.6, 0.6, 0.0]
    x[:, L.R_SHOULDER] = [0.4, 0.6, 0.0]
    x[:, L.NOSE] = [0.5, 0.4, -0.05]
    t = np.linspace(0, 2 * np.pi, T)[:, None]
    hand = rng.normal(0, 0.03, size=(L.N_HAND, 3))
    hand[0] = 0
    x[:, L.RHAND] = hand + np.stack([0.4 + 0.05 * np.sin(t), 0.5 + 0.05 * np.cos(t), 0 * t], -1)
    x[:, L.LHAND] = hand * [-1, 1, 1] + [0.62, 0.85, 0.0]
    x[:, L.R_WRIST] = x[:, L.RHAND.start]
    x[:, L.L_WRIST] = x[:, L.LHAND.start]
    x[:, L.FACE] = rng.normal([0.5, 0.42, -0.05], 0.02, size=(len(L.FACE_IDX), 3))
    return x


def similarity(x, angle, scale, shift):
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    return (x @ R.T) * scale + np.array([*shift, 0], dtype=np.float32)


@pytest.mark.parametrize("angle,scale,shift", [(0.3, 1.7, (0.2, -0.1)), (-0.5, 0.6, (-0.3, 0.4))])
def test_invariant_to_camera_roll_distance_and_position(angle, scale, shift):
    a = clip_features(fake_clip())
    b = clip_features(similarity(fake_clip(), angle, scale, shift))
    for g in a:
        np.testing.assert_allclose(a[g], b[g], atol=1e-4, err_msg=g)


def test_left_handed_signer_gives_same_features():
    x = fake_clip()
    mirrored = x[:, L.mirror_permutation()].copy()
    mirrored[..., 0] = 1.0 - mirrored[..., 0]  # flip the image horizontally
    a, b = clip_features(x), clip_features(mirrored)
    for g in ("handshape", "orientation", "location", "motion"):
        np.testing.assert_allclose(a[g], b[g], atol=1e-4, err_msg=g)


def test_swapped_hand_labels_are_repaired():
    x = fake_clip()
    swapped = x.copy()
    swapped[:, L.LHAND], swapped[:, L.RHAND] = x[:, L.RHAND], x[:, L.LHAND]
    a, b = clip_features(x), clip_features(swapped)
    np.testing.assert_allclose(a["handshape"], b["handshape"], atol=1e-5)


def test_missing_data_never_produces_nan_and_is_flagged():
    x = fake_clip()
    x[10:13, L.RHAND] = np.nan   # short dropout -> interpolated
    x[20:35, L.LHAND] = np.nan   # long dropout -> marked absent
    f = clip_features(x)
    assert all(np.isfinite(v).all() for v in f.values())
    q = f["quality"]  # [dom_seen, dom_interp, non_seen, non_interp, pose_seen, mirrored]
    assert q[10:13, 0].all() and q[10:13, 1].all()
    assert not q[20:35, 2].any()


def test_fill_gaps_only_fills_short_gaps():
    x = np.zeros((20, 1, 3), dtype=np.float32)
    x[:, 0, 0] = np.arange(20)
    x[3:5] = np.nan
    x[8:18] = np.nan
    filled, interp = fill_gaps(x, max_gap=6)
    np.testing.assert_allclose(filled[3:5, 0, 0], [3, 4])
    assert np.isnan(filled[8:18]).all() and interp[3:5, 0].all()


def test_adaptive_sampling_concentrates_on_motion():
    speed = np.r_[np.zeros(40), np.ones(20), np.zeros(40)]
    idx = adaptive_indices(speed, 32, uniform_weight=0.2)
    assert len(idx) == 32 and np.all(np.diff(idx) >= 0)
    assert ((idx >= 40) & (idx < 60)).mean() > 0.6


def test_sample_clip_shapes():
    f = sample_clip(clip_features(fake_clip(T=17)), length=32)
    assert all(v.shape[0] == 32 for v in f.values())
