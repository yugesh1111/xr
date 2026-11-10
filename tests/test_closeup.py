import numpy as np

from echogesture import layout as L
from echogesture.augment import close_up
from echogesture.features import (SHOULDER_DROP, SHOULDER_HALF, clip_features, ensure_shoulders,
                                  shoulders_in_frame)


def body_clip(T=40, fh=0.12):
    """Upright signer whose face and shoulders follow the measured proportions."""
    x = np.full((T, L.N_POINTS, 3), np.nan, dtype=np.float32)
    eyes = np.array([0.5, 0.30])
    mid = eyes + [0, SHOULDER_DROP * fh]
    x[:, :L.N_POSE] = [0.5, 0.5, 0.0]
    x[:, L.L_SHOULDER] = [mid[0] + SHOULDER_HALF * fh, mid[1], 0]
    x[:, L.R_SHOULDER] = [mid[0] - SHOULDER_HALF * fh, mid[1], 0]
    x[:, L.L_ELBOW] = [0.70, 0.75, 0]
    x[:, L.R_ELBOW] = [0.30, 0.75, 0]
    f = L.FACE.start
    x[:, f:L.FACE.stop] = [0.5, 0.36, -0.05]
    x[:, f + 3] = [0.46, 0.30, -0.05]
    x[:, f + 4] = [0.54, 0.30, -0.05]
    x[:, f + L.F_FOREHEAD] = [0.5, 0.30 - 0.4 * fh, -0.05]
    x[:, f + L.F_CHIN] = [0.5, 0.30 + 0.6 * fh, -0.05]
    t = np.linspace(0, 2 * np.pi, T)[:, None]
    hand = np.random.default_rng(0).normal(0, 0.02, size=(L.N_HAND, 3))
    hand[0] = 0
    x[:, L.RHAND] = hand + np.stack([0.49 + 0.02 * np.sin(t), 0.42 + 0.02 * np.cos(t), 0 * t], -1)
    x[:, L.R_WRIST] = x[:, L.RHAND.start]
    return x


def test_shoulders_in_view_are_untouched():
    x = body_clip()
    y, real = ensure_shoulders(x)
    assert real and y is x


def test_offscreen_shoulders_are_placed_from_the_face():
    x = body_clip()
    true_l = x[0, L.L_SHOULDER, :2].copy()
    x[:, [L.L_SHOULDER, L.R_SHOULDER, L.L_ELBOW, L.R_ELBOW], 1] += 0.8  # MediaPipe guesses, below the image
    assert not shoulders_in_frame(x).any()
    y, real = ensure_shoulders(x)
    assert not real
    np.testing.assert_allclose(y[0, L.L_SHOULDER, :2], true_l, atol=1e-3)
    assert np.isnan(y[:, L.L_ELBOW]).all()  # off-screen elbows are dropped


def test_close_up_features_match_the_full_view():
    x = body_clip()
    full = clip_features(x)
    rng = np.random.default_rng(1)
    c = close_up(x, rng, aspect=9 / 16, face_frac=0.5)
    assert not shoulders_in_frame(c).any()
    assert not np.isnan(c[:, L.RHAND]).any()  # the signing hand stays in view
    near = clip_features(c)
    for g in ("handshape", "orientation", "location", "motion"):
        np.testing.assert_allclose(near[g], full[g], atol=2e-3, err_msg=g)
    assert near["quality"][:, 4].max() == 0  # flagged: shoulders not seen
    assert not any(np.isnan(a).any() for a in near.values())
