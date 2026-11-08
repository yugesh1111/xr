"""Build data/wlasl/manifest.csv with a signer-independent split.

    python scripts/prepare_wlasl.py --vocab-size 100
    python scripts/prepare_wlasl.py --vocab-file organiser_vocab.txt
"""
import argparse
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from echogesture import wlasl  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="data/wlasl")
ap.add_argument("--vocab-size", type=int, default=None, help="100, 300, 1000 or 2000")
ap.add_argument("--vocab-file", default=None, help="one gloss per line; overrides --vocab-size")
ap.add_argument("--val-fraction", type=float, default=0.15)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--holdout-fraction", type=float, default=0.0,
                help="share of non-test clips (whole signers) kept out of everything, for an unbiased new-signer score")
ap.add_argument("--holdout-seed", type=int, default=100)
ap.add_argument("--out", default="manifest.csv", help="file name inside --root")
ap.add_argument("--only-available", action="store_true",
                help="keep only clips already in <root>/raw, then split (many WLASL links are dead)")
args = ap.parse_args()

root = Path(args.root)
root.mkdir(parents=True, exist_ok=True)
js = root / "WLASL_v0.3.json"
if not js.exists():
    urllib.request.urlretrieve(wlasl.WLASL_JSON_URL, js)

vocab = Path(args.vocab_file).read_text().split() if args.vocab_file else None
df = wlasl.build_manifest(wlasl.load_entries(js), vocab_size=args.vocab_size, vocab=vocab)
if args.only_available:
    have = {p.stem for p in (root / "raw").glob("*.*")}
    print(f"{df.video_id.isin(have).sum()} of {len(df)} clips available")
    df = df[df.video_id.isin(have)]
df = wlasl.signer_split(df, args.val_fraction, args.seed, args.holdout_fraction, args.holdout_seed)
df.to_csv(root / args.out, index=False)
print(wlasl.summarize(df))
print(f"wrote {root / args.out}")
