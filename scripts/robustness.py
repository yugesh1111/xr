"""Robustness benchmark: accuracy under corruptions vs clean, for one or more runs.

    # landmark-level (fast)
    python scripts/robustness.py --runs runs/e1 runs/e5 --level landmark
    # video-level: re-runs MediaPipe on corrupted pixels (slow, cached)
    python scripts/robustness.py --runs runs/e1 runs/e5 --level video
    # "a+b" = ensemble of runs a and b (probabilities averaged)
    python scripts/robustness.py --runs runs/e4 runs/e4+runs/e12 --level video

Evaluates on val + test together (unseen-signer val and official test).
Writes results/robustness_<level>.json and prints a markdown table.
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from echogesture import corruptions as C  # noqa: E402
from echogesture.inputs import make_input  # noqa: E402
from echogesture.model import build_model  # noqa: E402


def load_model(run):
    """`run` may be several runs joined by '+', evaluated as an ensemble."""
    members = []
    for r in run.split("+"):
        ck = torch.load(Path(r) / "best.pt", map_location="cpu")
        m = build_model(ck["config"])
        m.load_state_dict(ck["model"])
        members.append(m.eval())
    if len(members) == 1:
        return members[0], ck["config"], ck["vocab"]
    from echogesture.model import Ensemble
    return Ensemble(members).eval(), ck["config"], ck["vocab"]


def accuracy(model, cfg, clips, y):
    X = torch.from_numpy(np.stack([make_input(x, cfg["mode"], cfg["length"]) for x in clips]))
    with torch.no_grad():
        return float((model(X).argmax(1).numpy() == y).mean())


def corrupt_video_job(row, root, name, enhance=False):
    from echogesture.extract import extract_video
    out = root / f"landmarks_{name}{'_enh' if enhance else ''}" / f"{row['video_id']}.npz"
    if not out.exists():
        src = next((root / "raw").glob(f"{row['video_id']}.*"))
        xyz = extract_video(src, int(row["frame_start"]), int(row["frame_end"]), transform=C.VIDEO[name] if name != "clean" else None, enhance=enhance)
        np.savez_compressed(out, xyz=xyz.astype(np.float16))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--level", choices=["landmark", "video"], default="landmark")
    ap.add_argument("--root", default="data/wlasl")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--enhance", action="store_true",
                    help="video level: run the deployed pipeline (adaptive sharpening before MediaPipe), "
                         "clean included; writes robustness_video_enhanced.json")
    args = ap.parse_args()
    root = Path(args.root)

    df = pd.read_csv(root / "manifest.csv", dtype={"video_id": str})
    df = df[df.split.isin(["val", "test"])]
    df = df[[(root / "landmarks" / f"{v}.npz").exists() for v in df.video_id]]
    clean = {v: np.load(root / "landmarks" / f"{v}.npz")["xyz"].astype(np.float32) for v in df.video_id}

    conditions = {"clean": clean}
    if args.level == "landmark":
        for name, fn in C.LANDMARK.items():
            rng = np.random.default_rng(0)
            conditions[name] = {v: fn(x, rng) for v, x in clean.items()}
    else:
        rows = df[["video_id", "frame_start", "frame_end"]].to_dict("records")
        names = (["clean"] if args.enhance else []) + list(C.VIDEO)
        suffix = "_enh" if args.enhance else ""
        for name in names:
            (root / f"landmarks_{name}{suffix}").mkdir(exist_ok=True)
            with ProcessPoolExecutor(args.workers) as pool:
                list(tqdm(pool.map(partial(corrupt_video_job, root=root, name=name, enhance=args.enhance), rows),
                          total=len(rows), desc=name))
            conditions[name] = {v: np.load(root / f"landmarks_{name}{suffix}" / f"{v}.npz")["xyz"].astype(np.float32)
                                for v in df.video_id}

    results = {}
    for run in args.runs:
        model, cfg, vocab = load_model(run)
        idx = {g: i for i, g in enumerate(vocab)}
        keep = df[df.gloss.isin(idx)]
        y = np.array([idx[g] for g in keep.gloss])
        res = {}
        for name, data in conditions.items():
            clips = [data[v] if len(data[v]) else clean[v] for v in keep.video_id]
            res[name] = round(accuracy(model, cfg, clips, y), 4)
        results["+".join(Path(r).name for r in run.split("+"))] = res

    out = Path("results") / f"robustness_{args.level}{'_enhanced' if args.enhance else ''}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"clips": len(df), "results": results}, indent=1))
    names = list(conditions)
    print(f"{len(df)} clips (val + test), top-1 accuracy, drop vs clean in points\n")
    print("| run | " + " | ".join(names) + " | mean drop |")
    print("|---|" + "---|" * (len(names) + 1))
    for run, r in results.items():
        drops = [100 * (r["clean"] - r[n]) for n in names[1:]]
        cells = [f"{100 * r['clean']:.1f}%"] + [f"{100 * r[n]:.1f}% ({d:+.1f})" for n, d in zip(names[1:], [-x for x in drops])]
        print(f"| {run} | " + " | ".join(cells) + f" | {np.mean(drops):.1f} |")


if __name__ == "__main__":
    main()
