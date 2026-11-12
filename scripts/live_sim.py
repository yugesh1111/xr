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

from echogesture.live import LandmarkFilter, Segmenter  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=2, help="keep every n-th frame (WLASL is ~25 fps)")
    ap.add_argument("--old", action="store_true")
    ap.add_argument("--models", default="models")
    ap.add_argument("--no-gate", action="store_true", help="ignore models/gate_int8.onnx")
    ap.add_argument("--seg", default="{}", help="JSON overrides for the Segmenter, for tuning")
    ap.add_argument("--sentences", type=int, default=0, help="also test N three-sign sequences without dropping the hands")
    ap.add_argument("--jitter", type=float, default=0.0,
                    help="add live-tracker jitter: Gaussian noise of this many image heights per point and frame")
    ap.add_argument("--filter", default="", help='One Euro filter on the landmarks, JSON params, e.g. \'{}\' for defaults')
    args = ap.parse_args()

    import onnxruntime as ort
    meta = json.loads((Path(args.models) / "model_meta.json").read_text())
    sess = ort.InferenceSession(str(Path(args.models) / "model_int8.onnx"), providers=["CPUExecutionProvider"])

    # sign gate (scripts/train_gate.py), as in the app: no word at all when it says "not a sign"
    gate_path = Path(args.models) / "gate_int8.onnx"
    gate = thr = None
    if gate_path.exists() and not args.no_gate and not args.old:
        gate = ort.InferenceSession(str(gate_path), providers=["CPUExecutionProvider"])
        thr = json.loads((Path(args.models) / "gate_meta.json").read_text())["threshold"]

    def probs(x):
        f = make_input(x, meta["mode"], meta["length"])[None].astype(np.float32)
        if gate is not None:
            g = gate.run(None, {"x": f})[0][0]
            if 1 / (1 + np.exp(g[0] - g[1])) < thr:
                return np.full(len(meta["vocab"]), 1.0 / len(meta["vocab"]))  # nothing passes the caption rule
        z = sess.run(None, {"x": f})[0][0]
        p = np.exp(z - z.max())
        return p / p.sum()

    def shown(p):  # MainActivity: SHOW_MIN_PROB 0.45, SHOW_MIN_MARGIN 0.20
        s = np.sort(p)[::-1]
        return s[0] >= 0.45 and s[0] - s[1] >= 0.20

    m = pd.read_csv("data/wlasl/manifest.csv", dtype={"video_id": str})
    m = m[m.split == "test"]
    rng = np.random.default_rng(0)
    jrng = np.random.default_rng(7)
    n = cut = on_screen = right = idle_n = idle_words = 0
    for _, r in m.iterrows():
        p = Path("data/wlasl/landmarks") / f"{r.video_id}.npz"
        if not p.exists():
            continue
        x = np.load(p)["xyz"].astype(np.float32)[::args.every]
        rest = x[:1].copy()
        rest[:, L.LHAND] = np.nan
        rest[:, L.RHAND] = np.nan
        stream = prepare(args, np.concatenate([np.repeat(rest, 10, 0), x, np.repeat(rest, 12, 0)]), jrng)
        seg = Segmenter(old=args.old, **json.loads(args.seg))
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
                s = prepare(args, s, jrng)
                seg = Segmenter(old=args.old, **json.loads(args.seg))
                idle_n += 1
                idle_words += sum(shown(probs(o)) for o in (seg.push(f, i * dt) for i, f in enumerate(s)) if o is not None)
    print(f"~{25 / args.every:.0f} fps, {'old' if args.old else 'current'} rule, {n} test signs, "
          f"jitter {args.jitter}, filter {args.filter or 'off'}")
    print(f"  sign found by the segmenter : {100 * cut / n:.1f}%")
    print(f"  word shown                  : {100 * on_screen / n:.1f}%  (right when shown {100 * right / max(on_screen, 1):.1f}%)")
    print(f"  correct word on screen      : {100 * right / n:.1f}%")
    print(f"  idle streams with a word    : {idle_words}/{idle_n}")
    if args.sentences:
        sentences(args, m, probs, shown, meta["vocab"])


def prepare(args, stream, rng):
    """What the tracker would hand the app: jitter added, then the optional filter."""
    if args.jitter:
        stream = stream + rng.normal(0, args.jitter, stream.shape).astype(np.float32)
    if args.filter:
        f = LandmarkFilter(**json.loads(args.filter))
        dt = 40.0 * args.every
        stream = np.stack([f(x, i * dt) for i, x in enumerate(stream)])
    return stream


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
        stream = prepare(args, np.concatenate([np.repeat(rest, 8, 0)] + parts + [np.repeat(rest, 20, 0)]), rng)
        seg = Segmenter(old=args.old, **json.loads(args.seg))
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
