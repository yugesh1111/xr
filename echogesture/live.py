"""The live path, shared by the laptop app (live.py), the simulator
(scripts/live_sim.py) and mirrored in the Android app (core/*.kt):

    camera frame -> 88 landmarks -> LandmarkFilter -> Segmenter -> Recognizer
                 -> caption rule -> Sentence

Every class here is a straight port of its Kotlin twin, so a behaviour seen
on the laptop is the behaviour on the phone.
"""
import json
import math
import time
from pathlib import Path

import numpy as np

from . import layout as L
from .inputs import make_input

TIPS = [4, 8, 12, 16, 20]

# caption rule (MainActivity.kt): a word is shown and spoken when the model is sure,
# shown grey as "word?" when it is a fair guess, and not shown at all below that
SHOW_MIN_PROB, SHOW_MIN_MARGIN, GUESS_MIN_PROB = 0.45, 0.20, 0.25


# ------------------------------------------------------------------ filtering

class LandmarkFilter:
    """One Euro filter (Casiez et al., CHI 2012) on every landmark coordinate.

    Live trackers jitter by a few pixels from frame to frame even when the hand
    is still. Jitter makes resting hands look like they move (false signs) and
    adds noise to the motion features. A fixed low-pass would also blur real,
    fast signing. The One Euro filter adapts: when a point moves slowly the
    cut-off is low (strong smoothing), when it moves fast the cut-off rises
    (little lag). A point that disappears resets, so a hand coming back into
    view is never dragged from where it was.

    Units: coordinates in image heights, time in seconds.
    """

    def __init__(self, min_cutoff=1.5, beta=8.0, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = self.dx = None
        self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self):
        self.x = self.dx = self.t = None

    def __call__(self, frame, t_ms):
        f = np.asarray(frame, dtype=np.float32)
        t = t_ms / 1000.0
        if self.x is None:
            self.x, self.dx, self.t = f.copy(), np.zeros_like(f), t
            return f.copy()
        dt = min(max(t - self.t, 1e-3), 0.5)
        self.t = t
        seen = ~np.isnan(f)
        fresh = seen & np.isnan(self.x)          # point came back: start again from it
        dx = np.where(seen & ~fresh, (f - self.x) / dt, 0.0)
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx = np.where(seen & ~fresh, a_d * dx + (1 - a_d) * np.nan_to_num(self.dx), 0.0)
        # cut-off from the speed of the whole point (x, y, z together), so a hand
        # moving along one axis is not smoothed more on the other axes
        speed = np.linalg.norm(self.dx.reshape(-1, 3), axis=1).repeat(3).reshape(f.shape)
        cutoff = self.min_cutoff + self.beta * speed
        a = 1.0 / (1.0 + 1.0 / (2 * math.pi * cutoff * dt))
        out = np.where(fresh, f, a * f + (1 - a) * self.x)
        out = np.where(seen, out, np.nan).astype(np.float32)
        self.x = out
        return out.copy()


# ------------------------------------------------------------------ segmenter

class Segmenter:
    """Cuts single signs out of a continuous landmark stream (Segmenter.kt).

    A sign starts once a hand is seen in `start` frames in a row and ends after
    `end_still_ms` of still hands, `end_missing_ms` without hands, or `max_ms`.
    It only counts when the wrist or a fingertip really moved (5-95% range of
    at least `min_wrist` / `min_finger` face heights), so resting or fidgeting
    hands never produce a word.
    """

    def __init__(self, old=False, start=2, end_missing_ms=500, end_still_ms=500, still_speed=0.1, max_ms=3500,
                 pre_roll=5, min_wrist=0.2, min_finger=0.25):
        self.old = old
        self.start, self.end_missing_ms, self.end_still_ms = start, end_missing_ms, end_still_ms
        self.still_speed, self.max_ms, self.pre_roll = still_speed, max_ms, pre_roll  # still_speed: image heights / s
        self.min_frames = 10 if old else 5
        self.min_wrist, self.min_finger = min_wrist, min_finger
        self.signing = False
        self.hist, self.seg, self.seen, self.last, self.last_t = [], [], 0, None, None
        self.last_range = (0.0, 0.0)

    @staticmethod
    def _hand(f, b):
        return not np.isnan(f[b.start, 0])

    def _speed(self, f):
        b = next((b for b in (L.RHAND, L.LHAND) if self._hand(f, b)), None)
        prev = self.last
        if b is None:
            self.last = None
            return 0.0
        self.last = f[b.start, :2].copy()
        return 0.0 if prev is None else float(np.hypot(*(self.last - prev)))

    def push(self, f, t_ms):
        """Feed one frame; returns the frames of a finished sign, else None."""
        hand = self._hand(f, L.LHAND) or self._hand(f, L.RHAND)
        dt = 40.0 if self.last_t is None else min(max(t_ms - self.last_t, 1.0), 500.0)
        self.last_t = t_ms
        dist = self._speed(f)
        sp = dist * 1000.0 / dt
        if not self.signing:
            self.hist = (self.hist + [f])[-(self.pre_roll + self.start):]
            self.seen = self.seen + 1 if hand else 0
            if self.seen >= self.start:
                self.signing, self.seg = True, list(self.hist)
                self.miss = self.still = 0
                self.moved, self.travel, self.t0 = False, 0.0, t_ms
            return None
        self.seg.append(f)
        self.travel += dist
        self.miss = 0 if hand else self.miss + dt
        self.moved |= sp > self.still_speed * 3
        self.still = self.still + dt if hand and sp < self.still_speed else 0
        if self.miss >= self.end_missing_ms or (self.moved and self.still >= self.end_still_ms) or t_ms - self.t0 >= self.max_ms:
            self.signing, self.hist, self.seen = False, [], 0
            out, self.seg = np.stack(self.seg), []
            return out if (self.old or t_ms - self.t0 >= 400) and self.is_sign(out) else None
        return None

    def is_sign(self, x):
        if len(x) < self.min_frames:
            return False
        fh = face_height(x)
        if self.old:
            return self.moved and self.travel / fh >= 0.8
        wrist, finger = movement_ranges(x)
        self.last_range = (wrist / fh, finger / fh)
        return wrist / fh >= self.min_wrist or finger / fh >= self.min_finger


def face_height(x):
    d = np.linalg.norm(x[:, L.FACE.start + L.F_FOREHEAD, :2] - x[:, L.FACE.start + L.F_CHIN, :2], axis=1)
    d = d[~np.isnan(d)]
    return float(np.median(d)) if len(d) else 0.25


def movement_ranges(x):
    """Largest 5-95% range of a wrist, and of a fingertip relative to its wrist."""
    def span(p):
        e = np.percentile(p, 95, 0) - np.percentile(p, 5, 0)
        return np.linalg.norm(e, axis=-1)
    wrist = finger = 0.0
    for b in (L.RHAND, L.LHAND):
        ok = ~np.isnan(x[:, b.start, 0])
        if ok.sum() < 3:
            continue
        h = x[ok][:, b.start:b.start + 21, :2]
        wrist = max(wrist, float(span(h[:, 0])))
        finger = max(finger, float(span(h[:, TIPS] - h[:, :1]).max()))
    return wrist, finger


# ------------------------------------------------------------------ recogniser

class Recognizer:
    """Sign gate + INT8 word model (Recognizer.kt)."""

    def __init__(self, models="models", gate=True, threads=1):
        import onnxruntime as ort
        models = Path(models)
        self.meta = json.loads((models / "model_meta.json").read_text())
        self.vocab = self.meta["vocab"]
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(models / "model_int8.onnx"), so, providers=["CPUExecutionProvider"])
        self.gate = self.thr = None
        if gate and (models / "gate_int8.onnx").exists():
            self.gate = ort.InferenceSession(str(models / "gate_int8.onnx"), so, providers=["CPUExecutionProvider"])
            self.thr = json.loads((models / "gate_meta.json").read_text())["threshold"]

    def __call__(self, frames):
        """Returns dict(words, probs, sign_prob, is_sign, ms) for one cut-out sign."""
        t0 = time.perf_counter()
        x = make_input(np.asarray(frames), self.meta["mode"], self.meta["length"])[None].astype(np.float32)
        sign_prob = 1.0
        if self.gate is not None:
            g = self.gate.run(None, {"x": x})[0][0]
            sign_prob = float(1 / (1 + np.exp(g[0] - g[1])))
        z = self.sess.run(None, {"x": x})[0][0]
        p = np.exp(z - z.max())
        p /= p.sum()
        top = np.argsort(-p)[:3]
        return {"words": [self.vocab[i] for i in top], "probs": [float(p[i]) for i in top], "p": p,
                "sign_prob": sign_prob, "is_sign": self.thr is None or sign_prob >= self.thr,
                "ms": (time.perf_counter() - t0) * 1000}


def caption(r):
    """'sure' (show + speak + add to the sentence), 'guess' (grey word?), or None."""
    if not r["is_sign"]:
        return None
    p = r["probs"]
    if p[0] >= SHOW_MIN_PROB and p[0] - p[1] >= SHOW_MIN_MARGIN:
        return "sure"
    return "guess" if p[0] >= GUESS_MIN_PROB else None


# ------------------------------------------------------------------ sentence

class Sentence:
    """Running sentence (Sentence.kt): a 4 s pause closes it, a word repeated
    within 2.5 s counts once, the last two sentences stay on screen."""

    def __init__(self, pause_ms=4000, repeat_ms=2500, keep=2, max_words=12, on_close=None):
        self.pause_ms, self.repeat_ms, self.keep, self.max_words = pause_ms, repeat_ms, keep, max_words
        self.on_close = on_close or (lambda s: None)
        self.closed, self.open, self.last_ms = [], [], 0

    def add(self, word, t_ms):
        self.tick(t_ms)
        w = word.lower()
        if self.open and self.open[-1] == w and t_ms - self.last_ms < self.repeat_ms:
            self.last_ms = t_ms
            return False
        self.open.append(w)
        self.last_ms = t_ms
        if len(self.open) >= self.max_words:
            self._close()
        return True

    def tick(self, t_ms):
        if self.open and t_ms - self.last_ms >= self.pause_ms:
            return self._close()
        return None

    def clear(self):
        self.closed, self.open = [], []

    def forming(self):
        return self._fmt(self.open, False) if self.open else None

    def _close(self):
        s = self._fmt(self.open, True)
        self.closed = (self.closed + [s])[-self.keep:]
        self.open = []
        self.on_close(s)
        return s

    @staticmethod
    def _fmt(words, end):
        s = " ".join(words)
        return s[:1].upper() + s[1:] + ("." if end else "")
