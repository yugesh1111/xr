"""INCLUDE-50 (Indian Sign Language) -> data/include50/manifest.csv + raw/ links.

    python scripts/prepare_include.py --videos <folder with Category/Word/MVI_xxxx.MOV>

INCLUDE (AI4Bharat, CC BY 4.0, zenodo.org/records/4010759) was recorded by deaf
students of St. Louis School for the Deaf, Adyar, Chennai. INCLUDE-50 is its
50-word subset. Splits are the official ones from the Hugging Face card
(ai4bharat/INCLUDE: train / val / test parquet files, passed with --splits).
Videos are linked, not copied. Then:

    python scripts/extract_landmarks.py --root data/include50
"""
import argparse
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def gloss(label):
    """'51. Good Morning' -> 'GOOD MORNING'; 'big large' -> 'BIG'."""
    g = re.sub(r"^\d+\.\s*", "", label).strip()
    g = {"big large": "big", "small little": "small", "Store or Shop": "shop", "you (plural)": "you all",
         "train ticket": "ticket", "Cell phone": "phone"}.get(g, g)
    return g.upper()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--splits", required=True, help="folder with train/val/test .parquet from ai4bharat/INCLUDE")
    ap.add_argument("--out", default=str(ROOT / "data" / "include50"))
    a = ap.parse_args()
    out, videos = Path(a.out), Path(a.videos)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    rows = []
    for split in ["train", "val", "test"]:
        d = pd.read_parquet(Path(a.splits) / f"{split}.parquet")
        for r in d[d.include_50].itertuples():
            src = videos / r.video_path
            if not src.exists():
                continue
            vid = re.sub(r"[^A-Za-z0-9]+", "_", r.video_path.rsplit(".", 1)[0]).strip("_")
            link = out / "raw" / f"{vid}{src.suffix.lower()}"
            if not link.exists():
                link.symlink_to(src.resolve())
            rows.append({"video_id": vid, "gloss": gloss(r.label), "split": split, "frame_start": 1, "frame_end": -1,
                         "source": "include50", "signer_id": "include", "category": r.parent_label})
    df = pd.DataFrame(rows)
    vocab = sorted(df.gloss.unique())
    df["label"] = df.gloss.map({g: i for i, g in enumerate(vocab)})
    df.to_csv(out / "manifest.csv", index=False)
    print(df.split.value_counts().to_string(), f"\n{len(vocab)} words:", ", ".join(vocab))


if __name__ == "__main__":
    main()
