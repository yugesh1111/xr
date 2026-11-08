"""Evaluate a trained run.

    python scripts/evaluate.py --run runs/e5 --split test
    python scripts/evaluate.py --run runs/e5 --split valtest --tta 3
    python scripts/evaluate.py --run runs/ens_e4_e12 --onnx --split valtest   # the exported INT8 file

Reports top-1/top-5, macro accuracy (mean per-class recall), the weakest
classes, per-signer accuracy, and for the test split seen vs unseen signers.
`valtest` pools validation (unseen signers) and test for a larger sample.
--tta N averages N predictions made with different frame sampling.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from echogesture.data import load_split  # noqa: E402
from echogesture.inputs import make_input  # noqa: E402
from echogesture.model import build_model  # noqa: E402

TTA_WEIGHTS = [0.5, 0.3, 0.8, 0.15, 1.0]  # uniform_weight of the frame sampler


def wilson(k, n, z=1.96):
    """95% confidence interval for a proportion."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="test", help="train, val, test or valtest")
    ap.add_argument("--tta", type=int, default=1)
    ap.add_argument("--onnx", action="store_true", help="evaluate <run>/model_int8.onnx instead of best.pt")
    ap.add_argument("--manifest", default="data/wlasl/manifest.csv")
    ap.add_argument("--landmarks", default="data/wlasl/landmarks")
    args = ap.parse_args()

    if args.onnx:
        import onnxruntime as ort
        meta = json.loads((Path(args.run) / "model_meta.json").read_text())
        cfg, vocab = {"mode": meta["mode"], "length": meta["length"]}, meta["vocab"]
        sess = ort.InferenceSession(str(Path(args.run) / "model_int8.onnx"), providers=["CPUExecutionProvider"])

        def model(X):
            out = np.concatenate([sess.run(None, {"x": x[None]})[0] for x in X.numpy()])
            return torch.from_numpy(out)
    else:
        ck = torch.load(Path(args.run) / "best.pt", map_location="cpu")
        cfg, vocab = ck["config"], ck["vocab"]
        model = build_model(cfg)
        model.load_state_dict(ck["model"])
        model.eval()

    splits = ["val", "test"] if args.split == "valtest" else [args.split]
    parts = [load_split(args.manifest, args.landmarks, s, vocab) for s in splits]
    paths = [p for part in parts for p in part[0]]
    y = np.concatenate([part[1] for part in parts])
    df = pd.concat([part[3] for part in parts])
    clips = [np.load(p)["xyz"].astype(np.float32) for p in paths]

    prob = 0
    with torch.no_grad():
        for w in TTA_WEIGHTS[:args.tta]:
            X = torch.from_numpy(np.stack([make_input(x, cfg["mode"], cfg["length"], uniform_weight=w) for x in clips]))
            prob = prob + model(X).softmax(-1).numpy()
    top5 = np.argsort(-prob, 1)[:, :5]
    hit1, hit5 = top5[:, 0] == y, (top5 == y[:, None]).any(1)

    per_class = pd.Series(hit1).groupby(y).mean()
    lo, hi = wilson(int(hit1.sum()), len(y))
    res = {"run": Path(args.run).name, "split": args.split, "tta": args.tta, "clips": len(y),
           "top1": round(float(hit1.mean()), 4), "top1_95ci": [round(lo, 4), round(hi, 4)],
           "top5": round(float(hit5.mean()), 4), "macro_top1": round(float(per_class.mean()), 4),
           "classes": int(per_class.size)}
    res["weakest_classes"] = {vocab[c]: round(float(a), 2) for c, a in per_class.sort_values().head(10).items()}
    sig = pd.Series(hit1).groupby(df.signer_id.to_numpy())
    res["per_signer_top1"] = {"signers": int(sig.ngroups), "min": round(float(sig.mean().min()), 3),
                              "median": round(float(sig.mean().median()), 3)}
    if "test_signer_seen" in df and args.split == "test":
        seen = df.test_signer_seen.to_numpy().astype(bool)
        for name, m in (("seen_signers", seen), ("unseen_signers", ~seen)):
            if m.any():
                res[name] = {"clips": int(m.sum()), "top1": round(float(hit1[m].mean()), 4)}
    tag = f"{args.split}" + (f"_tta{args.tta}" if args.tta > 1 else "") + ("_int8" if args.onnx else "")
    res["model"] = "int8 onnx" if args.onnx else "pytorch fp32"
    (Path(args.run) / f"eval_{tag}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
