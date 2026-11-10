"""Accuracy when the phone is close (shoulders off-screen) or far (small, noisy signer).

Takes the clips of one split, makes fixed close-up and far-away versions of
their landmarks (same random seed for every model), and reports top-1 per
condition. Also reports how often a shown caption would be right for the
caption rule used by the app (top prob >= P and lead over the 2nd >= M).

    python scripts/eval_framing.py --manifest data/wlasl/manifest_holdout_100.csv \\
        --split holdout --runs runs/h100_base runs/h100_near_far
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from echogesture.augment import close_up, far_away  # noqa: E402
from echogesture.data import load_split  # noqa: E402
from echogesture.inputs import make_input  # noqa: E402
from scripts.robustness import load_model  # noqa: E402

CONDITIONS = {
    "clean": lambda x, rng: x,
    "near_arm": lambda x, rng: close_up(x, rng, aspect=9 / 16, face_frac=0.30),
    "near_close": lambda x, rng: close_up(x, rng, aspect=9 / 16, face_frac=0.50),
    "far_half": lambda x, rng: far_away(x, rng, scale=0.5),
    "far_third": lambda x, rng: far_away(x, rng, scale=0.33),
}


class OnnxModel:
    """An exported model_int8.onnx (+ model_meta.json) folder, used like the PyTorch one."""
    def __init__(self, folder):
        import onnxruntime as ort
        meta = json.loads((Path(folder) / "model_meta.json").read_text())
        self.cfg, self.vocab = {"mode": meta["mode"], "length": meta["length"]}, meta["vocab"]
        self.sess = ort.InferenceSession(str(Path(folder) / "model_int8.onnx"), providers=["CPUExecutionProvider"])

    def __call__(self, X):
        return torch.from_numpy(np.concatenate([self.sess.run(None, {"x": x[None]})[0] for x in X.numpy()]))


def probs(model, cfg, clips):
    X = torch.from_numpy(np.stack([make_input(x, cfg["mode"], cfg["length"]) for x in clips]))
    with torch.no_grad():
        return model(X).softmax(-1).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run dirs; a+b = ensemble")
    ap.add_argument("--manifest", default="data/wlasl/manifest.csv")
    ap.add_argument("--landmarks", default="data/wlasl/landmarks")
    ap.add_argument("--split", default="holdout")
    ap.add_argument("--show-prob", type=float, default=0.45)
    ap.add_argument("--show-margin", type=float, default=0.20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    results = {}
    for run in args.runs:
        if (Path(run) / "model_int8.onnx").exists() and not (Path(run) / "best.pt").exists():
            model = OnnxModel(run)
            cfg, vocab = model.cfg, model.vocab
        else:
            model, cfg, vocab = load_model(run)
        paths, y, _, _ = load_split(args.manifest, args.landmarks, args.split, vocab)
        clean = [np.load(p)["xyz"].astype(np.float32) for p in paths]
        res = {"clips": len(y)}
        for name, fn in CONDITIONS.items():
            rng = np.random.default_rng(0)
            P = probs(model, cfg, [fn(x, rng) for x in clean])
            top = np.sort(P, 1)[:, ::-1]
            pred = P.argmax(1)
            shown = (top[:, 0] >= args.show_prob) & (top[:, 0] - top[:, 1] >= args.show_margin)
            res[name] = {"top1": round(float((pred == y).mean()), 4),
                         "shown": round(float(shown.mean()), 4),
                         "shown_correct": round(float((pred[shown] == y[shown]).mean()) if shown.any() else 0.0, 4)}
        results[run] = res

    names = list(CONDITIONS)
    print(f"split {args.split}, {results[args.runs[0]]['clips']} clips; top-1 (shown share / shown correct)")
    print("| run | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    for run, r in results.items():
        cells = [f"{100 * r[n]['top1']:.1f}% ({100 * r[n]['shown']:.0f}% / {100 * r[n]['shown_correct']:.0f}%)" for n in names]
        print(f"| {run} | " + " | ".join(cells) + " |")
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
