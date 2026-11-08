"""Fill the "Data report" section of README.md: what data we have and how good it is.

Run after download + landmark extraction:
    python scripts/data_report.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="data/wlasl")
ap.add_argument("--readme", default="README.md")
args = ap.parse_args()
root = Path(args.root)

full = json.loads((root / "WLASL_v0.3.json").read_text())
df = pd.read_csv(root / "manifest.csv", dtype={"video_id": str})
n_vocab = df.gloss.nunique()
listed = sum(len(e["instances"]) for e in full[:n_vocab]) if n_vocab in (100, 300, 1000, 2000) else None
df["host"] = df.url.str.split("/").str[2]

rows = []
for r in df.itertuples():
    p = root / "landmarks" / f"{r.video_id}.npz"
    if not p.exists():
        rows.append({"video_id": r.video_id, "extracted": False})
        continue
    x = np.load(p)["xyz"].astype(np.float32)
    seen = lambda a, b: float((~np.isnan(x[:, a:b, 0]).all(1)).mean()) if len(x) else 0.0
    rows.append({"video_id": r.video_id, "extracted": len(x) > 0, "frames": len(x),
                 "pose": seen(0, 33), "left_hand": seen(33, 54), "right_hand": seen(54, 75),
                 "any_hand": float((~np.isnan(x[:, 33:75, 0])).any(1).mean()) if len(x) else 0.0})
q = df.merge(pd.DataFrame(rows), on="video_id")
ok = q[q.extracted == True]  # noqa: E712

L = [f"**WLASL{n_vocab}**", "", "Coverage:", ""]
if listed:
    L.append(f"- Clips listed in WLASL for these {n_vocab} glosses: **{listed}**")
L += [f"- Clips we could obtain (Kaggle mirror + direct links): **{len(df)}**",
      f"- Clips with landmarks extracted: **{len(ok)}**", "",
      "| split | clips | signers | glosses |", "|---|---|---|---|"]
for s in ("train", "val", "test"):
    p = ok[ok.split == s]
    L.append(f"| {s} | {len(p)} | {p.signer_id.nunique()} | {p.gloss.nunique()} |")
L += ["", "Clips per gloss in train: "
      f"min {ok[ok.split == 'train'].gloss.value_counts().min()}, "
      f"median {int(ok[ok.split == 'train'].gloss.value_counts().median())}, "
      f"max {ok[ok.split == 'train'].gloss.value_counts().max()}", "",
      "Clips by source site:", "", "| host | clips |", "|---|---|"]
for h, n in ok.host.value_counts().items():
    L.append(f"| {h} | {n} |")
L += ["", "Landmark quality (MediaPipe Holistic):", "",
      f"- Frames per clip: median {int(ok.frames.median())}, min {int(ok.frames.min())}, max {int(ok.frames.max())}",
      f"- Pose found: {ok.pose.mean():.0%} of frames on average",
      f"- At least one hand found: {ok.any_hand.mean():.0%} of frames on average "
      "(low values are mostly rest pose at the start/end of a clip)",
      f"- Clips where no hand was ever found: {(ok.any_hand == 0).sum()}", ""]
readme = Path(args.readme)
text = readme.read_text()
start, end = "<!-- data-report:start -->", "<!-- data-report:end -->"
a, b = text.index(start) + len(start), text.index(end)
readme.write_text(text[:a] + "\n" + "\n".join(L) + "\n" + text[b:])
print("\n".join(L))
