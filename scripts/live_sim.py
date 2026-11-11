"""Simulate the app's live path on WLASL test clips.

The offline test (main.py) gives the model each whole clip. The app instead
cuts signs out of a camera stream (Segmenter.kt) and only shows a word when
the model is sure (MainActivity.confident). This script runs both steps on
the test clips at phone frame rates, plus still-hands "idle" streams that
should never produce a word.

    python scripts/live_sim.py --every 2          # ~12 fps
    python scripts/live_sim.py --every 2 --old    # the 10 Nov rule (wrist travel >= 0.8 faces)
    python scripts/live_sim.py --every 3 --sentences 60   # also: 3 signs in a row, short pauses
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from echogesture import layout as L  # noqa: E402
from echogesture.inputs import make_input  # noqa: E402

TIPS = [4, 8, 12, 16, 20]


class Segmenter:
    """Python port of android/.../core/Segmenter.kt."""

    def __init__(self, old=False, start=3, end_missing_ms=500, end_still_ms=500, still_speed=0.1, max_ms=3500, pre_roll=5):
        self.old = old
        self.start, self.end_missing_ms, self.end_still_ms = start, end_missing_ms, end_still_ms
        self.still_speed, self.max_ms, self.pre_roll = still_speed, max_ms, pre_roll  # still_speed: image heights / s
        self.min_frames = 10 if old else 5
        self.signing = False
        self.hist, self.seg, self.seen, self.last, self.last_t = [], [], 0, None, None

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
        return wrist / fh >= 0.3 or finger / fh >= 0.4


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=2, help="keep every n-th frame (WLASL is ~25 fps)")
    ap.add_argument("--old", action="store_true")
    ap.add_argument("--models", default="models")
    ap.add_argument("--sentences", type=int, default=0, help="also test N three-sign sequences without dropping the hands")
    args = ap.parse_args()

    import onnxruntime as ort
    meta = json.loads((Path(args.models) / "model_meta.json").read_text())
    sess = ort.InferenceSession(str(Path(args.models) / "model_int8.onnx"), providers=["CPUExecutionProvider"])

    def probs(x):
        z = sess.run(None, {"x": make_input(x, meta["mode"], meta["length"])[None].astype(np.float32)})[0][0]
        p = np.exp(z - z.max())
        return p / p.sum()

    def shown(p):  # MainActivity: SHOW_MIN_PROB 0.45, SHOW_MIN_MARGIN 0.20
        s = np.sort(p)[::-1]
        return s[0] >= 0.45 and s[0] - s[1] >= 0.20

    m = pd.read_csv("data/wlasl/manifest.csv", dtype={"video_id": str})
    m = m[m.split == "test"]
    rng = np.random.default_rng(0)
    n = cut = on_screen = right = idle_n = idle_words = 0
    for _, r in m.iterrows():
        p = Path("data/wlasl/landmarks") / f"{r.video_id}.npz"
        if not p.exists():
            continue
        x = np.load(p)["xyz"].astype(np.float32)[::args.every]
        rest = x[:1].copy()
        rest[:, L.LHAND] = np.nan
        rest[:, L.RHAND] = np.nan
        stream = np.concatenate([np.repeat(rest, 10, 0), x, np.repeat(rest, 12, 0)])
        seg = Segmenter(old=args.old)
        dt = 40.0 * args.every
        segs = [s for s in (seg.push(f, i * dt) for i, f in enumerate(stream)) if s is not None]
        n += 1
        if segs:
            cut += 1
            best = max((probs(s) for s in segs), key=lambda q: q.max())
            if shown(best):
                on_screen += 1
                right += meta["vocab"][int(best.argmax())] == r.gloss
        # idle: hands held still in view with tracker jitter / slow sway, 8 s
        vis = np.flatnonzero(~np.isnan(x[:, L.RHAND.start, 0]))
        if len(vis):
            f0 = x[vis[np.nanargmax(x[vis, L.RHAND.start, 1])]]
            for noise, sway in [(0.002, 0.0), (0.004, 0.0), (0.002, 0.003)]:
                s = f0[None] + rng.normal(0, noise, (96,) + f0.shape).astype(np.float32)
                s[..., 0] += sway * np.sin(np.arange(96)[:, None] / 6.0)
                seg = Segmenter(old=args.old)
                idle_n += 1
                idle_words += sum(shown(probs(o)) for o in (seg.push(f, i * dt) for i, f in enumerate(s)) if o is not None)
    print(f"~{25 / args.every:.0f} fps, {'old' if args.old else 'current'} rule, {n} test signs")
    print(f"  sign found by the segmenter : {100 * cut / n:.1f}%")
    print(f"  word shown                  : {100 * on_screen / n:.1f}%  (right when shown {100 * right / max(on_screen, 1):.1f}%)")
    print(f"  correct word on screen      : {100 * right / n:.1f}%")
    print(f"  idle streams with a word    : {idle_words}/{idle_n}")
    if args.sentences:
        sentences(args, m, probs, shown, meta["vocab"])


def sentences(args, m, probs, shown, vocab):
    """Three test clips back to back, hands kept up: between signs the last
    pose is held for 0.7 s (with tracker jitter) instead of dropping the hands."""
    import difflib
    rng = np.random.default_rng(1)
    dt = 40.0 * args.every
    clips = []
    for _, r in m.iterrows():
        p = Path("data/wlasl/landmarks") / f"{r.video_id}.npz"
        if p.exists():
            x = np.load(p)["xyz"].astype(np.float32)[::args.every]
            vis = ~np.isnan(x[:, L.RHAND.start, 0]) | ~np.isnan(x[:, L.LHAND.start, 0])
            if vis.any():
                a, b = np.flatnonzero(vis)[[0, -1]]
                clips.append((r.gloss, x[a:b + 1]))
    hold = max(1, int(round(700 / dt)))
    exact = hits = total = 0
    for _ in range(args.sentences):
        pick = [clips[i] for i in rng.choice(len(clips), 3, replace=False)]
        parts = []
        for g, x in pick:
            parts.append(x)
            parts.append(x[-1:] + rng.normal(0, 0.002, (hold,) + x.shape[1:]).astype(np.float32))
        rest = parts[0][:1].copy()
        rest[:, L.LHAND] = np.nan
        rest[:, L.RHAND] = np.nan
        stream = np.concatenate([np.repeat(rest, 8, 0)] + parts + [np.repeat(rest, 20, 0)])
        seg = Segmenter(old=args.old)
        words = []
        for i, f in enumerate(stream):
            o = seg.push(f, i * dt)
            if o is not None:
                q = probs(o)
                if shown(q):
                    w = vocab[int(q.argmax())]
                    if not words or words[-1] != w:
                        words.append(w)
        truth = [g for g, _ in pick]
        sm = difflib.SequenceMatcher(a=truth, b=words)
        hits += sum(bl.size for bl in sm.get_matching_blocks())
        total += 3
        exact += words == truth
    print(f"  3-sign sequences ({args.sentences}): words right and in order {100 * hits / total:.1f}%, whole sequence exact {100 * exact / args.sentences:.1f}%")


if __name__ == "__main__":
    main()
