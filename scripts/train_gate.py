"""Sign gate: a small "is this a sign from our vocabulary?" model for the live app.

The word model always answers with one of its 100 words, so a scratch of the
nose or a sign it never learned comes out as some word. The gate sees the same
32 x 239 input and says sign / not a sign; the app shows a word only when the
gate agrees. predict() in main.py does not use it (every test clip is a sign).

Positives: WLASL100 training clips (clean, augmented, and cut the way the live
segmenter cuts them) and Kaggle ASL clips of the same words.
Negatives:
  rest     - the stretches before and after the sign inside WLASL clips
             (hands resting, coming up, going down)
  other    - Kaggle ASL clips of signs that are not in the 100 words
  fidget   - synthetic: a real pose whose hand drifts around, goes to the face
             and stays there, or just hovers

    python scripts/train_gate.py --out runs/gate
Writes gate_fp32.onnx, gate_int8.onnx, gate_meta.json, report.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from echogesture import layout as L  # noqa: E402
from echogesture.augment import augment  # noqa: E402
from echogesture.inputs import group_slices, make_input  # noqa: E402
from echogesture.model import EchoNet  # noqa: E402

LEN = 32


def hands_seen(x):
    return ~np.isnan(x[:, L.RHAND.start, 0]) | ~np.isnan(x[:, L.LHAND.start, 0])


def wrist_speed(x):
    sp = np.zeros(len(x))
    for b in (L.RHAND, L.LHAND):
        w = x[:, b.start, :2]
        d = np.linalg.norm(np.diff(w, axis=0), axis=1)
        sp[1:] = np.fmax(sp[1:], np.nan_to_num(d))
    return sp


def active_span(x):
    """Frames that hold the middle 90% of the wrist movement."""
    sp = wrist_speed(x)
    if sp.sum() <= 0:
        return 0, len(x) - 1
    c = np.cumsum(sp) / sp.sum()
    return int(np.searchsorted(c, 0.05)), int(np.searchsorted(c, 0.95))


def segment_like(x, rng):
    """Cut a clip roughly the way the live segmenter would: start a few frames
    before the hands show, end a little after the movement stops."""
    seen = np.flatnonzero(hands_seen(x))
    if len(seen) < 8:
        return x
    a, b = active_span(x)
    lo = max(min(seen[0], a) - rng.integers(0, 6), 0)
    hi = min(max(b + rng.integers(4, 16), a + 8), len(x) - 1)
    return x[lo:hi + 1]


def rest_fragments(x, rng):
    """Before / after the sign: hands resting, rising, falling."""
    a, b = active_span(x)
    out = []
    for lo, hi in ((0, a), (b, len(x))):
        if hi - lo >= 8 and hands_seen(x[lo:hi]).mean() >= 0.5:
            n = rng.integers(8, min(hi - lo, 40) + 1)
            s = rng.integers(lo, hi - n + 1)
            out.append(x[s:s + n])
    return out


def smooth_path(rng, T, amp):
    """(T, 2) random smooth 2-D path starting at 0, size ~amp."""
    k = rng.integers(2, 6)
    pts = np.concatenate([[[0, 0]], rng.normal(0, amp, (k, 2))])
    t = np.linspace(0, k, T)
    return np.stack([np.interp(t, np.arange(k + 1), pts[:, d]) for d in range(2)], 1)


def fidget(frame, rng):
    """A non-sign from one real frame: the hand drifts, goes to the face and
    stays, or hovers with tracker jitter."""
    T = int(rng.integers(10, 60))
    x = np.repeat(frame[None], T, 0).astype(np.float32)
    face = frame[L.FACE.start + L.F_FOREHEAD:L.FACE.start + L.F_CHIN + 1]
    fh = np.linalg.norm(frame[L.FACE.start + L.F_FOREHEAD, :2] - frame[L.FACE.start + L.F_CHIN, :2])
    fh = 0.25 if not np.isfinite(fh) or fh < 1e-3 else fh
    kind = rng.choice(["drift", "to_face", "hover", "one_move"])
    for b, pw, pe in ((L.LHAND, 15, 13), (L.RHAND, 16, 14)):
        if np.isnan(frame[b.start, 0]):
            continue
        if kind == "drift":
            off = smooth_path(rng, T, rng.uniform(0.15, 0.8) * fh)
        elif kind == "to_face" and np.isfinite(face).all():
            target = np.nanmean(face[:, :2], 0) + rng.normal(0, 0.3 * fh, 2) - frame[b.start, :2]
            ramp = np.clip(np.arange(T) / max(rng.integers(4, 15), 1), 0, 1) ** 0.7
            off = ramp[:, None] * target + smooth_path(rng, T, 0.04 * fh)
        elif kind == "one_move":
            ramp = np.clip((np.arange(T) - rng.integers(0, T // 2 + 1)) / max(rng.integers(3, 12), 1), 0, 1)
            off = ramp[:, None] * rng.normal(0, 0.8 * fh, 2)
        else:
            off = smooth_path(rng, T, 0.05 * fh)
        x[:, b, :2] += off[:, None]
        # slow finger wiggle + tracker jitter
        wig = np.stack([smooth_path(rng, T, 0.03 * fh) for _ in range(21)], 1)
        x[:, b, :2] += wig * rng.uniform(0, 1)
        x[:, pw, :2] += off
        x[:, pe, :2] += 0.5 * off
    x[..., :2] += rng.normal(0, 0.002, x[..., :2].shape)
    return x


def feats(clips):
    return np.stack([make_input(c, "canonical", LEN) for c in clips]).astype(np.float32)


def build(split_clips, kag_pos, kag_neg, rng, n_aug):
    pos, neg = [], []
    for x in split_clips:
        pos.append(x)
        pos.append(segment_like(x, rng))
        for _ in range(n_aug):
            a = augment(x, rng)
            pos.append(segment_like(a, rng) if rng.random() < 0.5 else a)
        neg += rest_fragments(x, rng)
        seen = np.flatnonzero(hands_seen(x))
        for _ in range(2 + n_aug):
            if len(seen):
                neg.append(fidget(x[rng.choice(seen)], rng))
    for x in kag_pos:
        pos.append(x)
    for x in kag_neg:
        neg.append(x)
        seen = np.flatnonzero(hands_seen(x))
        if len(seen) and rng.random() < 0.5:
            neg.append(fidget(x[rng.choice(seen)], rng))
    kinds = {"rest+fidget+other": len(neg)}
    return pos, neg, kinds


def load(root, ids):
    out = []
    for v in ids:
        p = Path(root) / f"{v}.npz"
        if p.exists():
            x = np.load(p)["xyz"].astype(np.float32)
            if len(x) >= 4:
                out.append(x)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/gate")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--kaggle", type=int, default=5000, help="Kaggle clips per side (train)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    m = pd.read_csv("data/wlasl/manifest.csv", dtype={"video_id": str})
    k = pd.read_csv("data/kaggle_asl/manifest_split.csv", dtype={"video_id": str})
    kdir = "data/kaggle_asl/landmarks_fixed"

    def kag(split, flag, n):
        d = k[(k.split == split) & (k.in_wlasl == flag)]
        return load(kdir, d.sample(min(n, len(d)), random_state=args.seed).video_id)

    tr_pos, tr_neg, _ = build(load("data/wlasl/landmarks", m[m.split == "train"].video_id),
                              kag("train", True, args.kaggle), kag("train", False, args.kaggle), rng, n_aug=4)
    # held out: WLASL val + test clips (unseen signers in val), Kaggle val signers
    ho = load("data/wlasl/landmarks", m[m.split.isin(["val", "test"])].video_id)
    te_pos = [c for x in ho for c in (x, segment_like(x, rng))]
    te_rest = [f for x in ho for f in rest_fragments(x, rng)]
    te_fid = [fidget(x[rng.choice(np.flatnonzero(hands_seen(x)))], rng) for x in ho for _ in range(3) if hands_seen(x).any()]
    te_other = kag("val", False, 1500)
    te_kpos = kag("val", True, 800)
    print(f"train: {len(tr_pos)} signs, {len(tr_neg)} non-signs; held out: {len(te_pos)} signs, "
          f"{len(te_rest)} rest, {len(te_fid)} fidget, {len(te_other)} other signs", flush=True)

    Xp, Xn = feats(tr_pos), feats(tr_neg)
    X = torch.from_numpy(np.concatenate([Xp, Xn]))
    y = torch.cat([torch.ones(len(Xp)), torch.zeros(len(Xn))]).long()
    cfg = dict(slices=group_slices("canonical"), n_classes=2, dim=64, fusion="gated", n_conv=2, n_attn=1,
               length=LEN, dropout=0.2, mode="canonical")
    model = EchoNet(cfg["slices"], 2, d=64, fusion="gated", n_conv=2, n_attn=1, length=LEN, dropout=0.2)
    flat = X.reshape(-1, X.shape[-1])
    model.mu.copy_(flat.mean(0))
    model.sigma.copy_(flat.std(0).clamp_min(1e-3))
    print("gate parameters:", sum(p.numel() for p in model.parameters()), flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=args.epochs * (len(X) // 128 + 1))
    w = torch.tensor([1.0, len(Xn) / len(Xp)])
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(X))
        tot = 0.0
        for i in range(0, len(X), 128):
            b = perm[i:i + 128]
            loss = nn.functional.cross_entropy(model(X[b]), y[b], weight=w, label_smoothing=0.05)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(b)
        if ep % 5 == 4 or ep == args.epochs - 1:
            print(f"ep {ep + 1} loss {tot / len(X):.3f}", flush=True)
    model.eval()

    def p_sign(clips):
        with torch.no_grad():
            return torch.softmax(model(torch.from_numpy(feats(clips))), 1)[:, 1].numpy()

    sets = {"signs (WLASL val+test)": p_sign(te_pos), "signs (Kaggle, same words)": p_sign(te_kpos),
            "rest": p_sign(te_rest), "fidget": p_sign(te_fid), "other signs (Kaggle)": p_sign(te_other)}
    # threshold: keep 97% of held-out WLASL signs
    thr = float(np.quantile(sets["signs (WLASL val+test)"], 0.03))
    report = {"threshold": round(thr, 3), "params": int(sum(p.numel() for p in model.parameters()))}
    for name, p in sets.items():
        acc = float((p >= thr).mean())
        report[name] = {"n": int(len(p)), "passed": round(acc, 4)}
        print(f"{name:28s} n={len(p):5d}  let through {100 * acc:.1f}%", flush=True)

    torch.save({"model": model.state_dict(), "config": cfg}, out / "best.pt")
    dummy = torch.zeros(1, LEN, X.shape[-1])
    torch.onnx.export(model, dummy, str(out / "gate_fp32.onnx"), input_names=["x"], output_names=["logits"],
                      opset_version=17, dynamo=False)
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    class Reader(CalibrationDataReader):
        def __init__(self):
            idx = np.random.default_rng(0).choice(len(X), 300, replace=False)
            self.it = iter([{"x": X[i:i + 1].numpy()} for i in idx])

        def get_next(self):
            return next(self.it, None)

    quant_pre_process(str(out / "gate_fp32.onnx"), str(out / "gate_pre.onnx"))
    quantize_static(str(out / "gate_pre.onnx"), str(out / "gate_int8.onnx"), Reader(), quant_format=QuantFormat.QDQ,
                    per_channel=True, activation_type=QuantType.QInt8, weight_type=QuantType.QInt8)
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out / "gate_int8.onnx"), providers=["CPUExecutionProvider"])

    def p_int8(clips):
        z = np.concatenate([sess.run(None, {"x": f[None]})[0] for f in feats(clips)])
        e = np.exp(z - z.max(1, keepdims=True))
        return e[:, 1] / e.sum(1)

    report["int8"] = {name: round(float((p_int8(c) >= thr).mean()), 4) for name, c in
                      [("signs (WLASL val+test)", te_pos), ("rest", te_rest), ("fidget", te_fid), ("other signs (Kaggle)", te_other)]}
    report["size_mb"] = round((out / "gate_int8.onnx").stat().st_size / 1e6, 3)
    print("INT8:", report["int8"], "size", report["size_mb"], "MB", flush=True)
    (out / "gate_meta.json").write_text(json.dumps({"threshold": round(thr, 3), "length": LEN, "mode": "canonical"}, indent=1))
    (out / "report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
