"""Test-time corruptions for the robustness benchmark.

Video corruptions change the pixels before MediaPipe runs (what a real dark
room or a shaky camera does). Landmark corruptions change the extracted
points directly (cheap, and isolate the model's own robustness).
Everything is seeded so every model sees exactly the same corruption.
"""
import cv2
import numpy as np

from . import layout as L


# ---------------------------------------------------------------- video (pixels)

def dark(rgb, factor=0.25):
    """Low light: dim the image and add sensor noise, like a phone camera indoors at night."""
    x = rgb.astype(np.float32) * factor
    x += np.random.default_rng(0).normal(0, 4, x.shape)
    return np.clip(x, 0, 255).astype(np.uint8)


def blur(rgb, k=9):
    """Motion / focus blur."""
    return cv2.GaussianBlur(rgb, (k, k), 0)


def rotate(rgb, deg=15):
    """Tilted camera (e.g. a head-mounted device on a tilted head)."""
    h, w = rgb.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(rgb, m, (w, h), borderMode=cv2.BORDER_REFLECT)


def low_res(rgb, scale=0.35):
    """Signer far from the camera / cheap sensor."""
    h, w = rgb.shape[:2]
    small = cv2.resize(rgb, (max(int(w * scale), 1), max(int(h * scale), 1)), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


class PassingOccluder:
    """Occlusion: something passes between camera and signer (a person walking
    by, an arm, a coffee cup). A vertical bar 22% of the frame wide, in a dull
    clothing colour, sweeps left to right across the frame every 60 frames
    (~2 s at 30 fps), so for part of every sign one or both hands are hidden.
    Stateful per clip: call .fresh() for each new video."""

    def __init__(self, width=0.22, period=60, color=(70, 62, 58)):
        self.width, self.period, self.color, self.i = width, period, color, 0

    def fresh(self):
        return PassingOccluder(self.width, self.period, self.color)

    def __call__(self, rgb):
        h, w = rgb.shape[:2]
        bw = int(w * self.width)
        x0 = int((self.i % self.period) / self.period * (w + bw)) - bw
        self.i += 1
        out = rgb.copy()
        out[:, max(x0, 0):max(min(x0 + bw, w), 0)] = self.color
        return out


VIDEO = {"dark": dark, "blur": blur, "rotate15": rotate, "low_res": low_res, "occlusion": PassingOccluder()}


def random_degradation(seed):
    """A random pixel degradation for *training* (video-level augmentation).
    Strengths are drawn from ranges around - not equal to - the fixed test
    settings above, so the robustness test still measures generalisation."""
    rng = np.random.default_rng(seed)
    kind = rng.choice(["blur", "low_res", "dark", "blur+low_res"])
    k = int(rng.choice([5, 7, 11, 13]))
    scale = float(rng.uniform(0.3, 0.6))
    light = float(rng.uniform(0.15, 0.5))
    if kind == "blur":
        return lambda rgb: blur(rgb, k)
    if kind == "low_res":
        return lambda rgb: low_res(rgb, scale)
    if kind == "dark":
        return lambda rgb: dark(rgb, light)
    return lambda rgb: blur(low_res(rgb, scale), 5)


def random_low_res(seed):
    """A random low-resolution degradation for training (E14): the signer far
    away or a cheap sensor. Scale 25-50% (the test uses a fixed 35%), half of
    the time with a light blur on top."""
    rng = np.random.default_rng(seed)
    scale = float(rng.uniform(0.25, 0.5))
    k = int(rng.choice([0, 3, 5]))
    if k == 0:
        return lambda rgb: low_res(rgb, scale)
    return lambda rgb: blur(low_res(rgb, scale), k)


# ---------------------------------------------------------------- landmarks

def _rot(x, yaw_deg):
    c = np.nanmean(x[:, [L.L_SHOULDER, L.R_SHOULDER]].reshape(-1, 3), axis=0)
    a = np.radians(yaw_deg)
    R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    return ((x - c) @ R.T + c).astype(np.float32)


def side_view(x, rng, deg=40):
    """Signer turned sideways (exocentric view at an angle)."""
    return _rot(x, deg * rng.choice([-1, 1]))


def hide_dominant_hand(x, rng, frac=0.3):
    """The more active hand is lost for a contiguous 30% of the clip."""
    x = x.copy()
    T = len(x)
    e = [np.nansum(np.abs(np.diff(x[:, b, :2], axis=0))) for b in (L.LHAND, L.RHAND)]
    block = (L.LHAND, L.RHAND)[int(np.argmax(e))]
    span = max(int(T * frac), 1)
    s = rng.integers(0, max(T - span, 0) + 1)
    x[s:s + span, block] = np.nan
    return x


def drop_frames(x, rng, p=0.2):
    """Tracker fails on 20% of frames."""
    x = x.copy()
    x[rng.random(len(x)) < p] = np.nan
    return x


def jitter(x, rng, sigma=0.01):
    """Noisy landmarks (poor light, motion blur) - 1% of image height."""
    return (x + rng.normal(0, sigma, x.shape)).astype(np.float32)


LANDMARK = {"side_view_40": side_view, "hide_hand_30pct": hide_dominant_hand,
            "drop_20pct_frames": drop_frames, "jitter_1pct": jitter}
