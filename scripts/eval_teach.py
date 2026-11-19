"""How well does "teach the phone a new sign" work? (core/CustomSigns.kt)

A user teaches K new words, signing each one SHOTS times. Later, other people
sign those words (should be recognised as the right taught word) and also sign
everything else (must not be taken for a taught word). Nothing is retrained:
signs are compared by the cosine similarity of the word model's embedding.

Clips: Kaggle ASL Signs (250 signs, 21 signers) restricted to signs that are
NOT in the model's vocabulary, so every taught word is new to the model.
Teaching clips come from one signer, test clips from the other signers.

    python scripts/eval_teach.py --ckpt runs/f_base/best.pt
Writes results/teach_eval.json and prints the threshold table.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from echogesture.data import ClipDataset  # noqa: E402
from echogesture.model import build_model  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="runs/f_base/best.pt")
ap.add_argument("--manifest", default="data/kaggle_asl/manifest.csv")
ap.add_argument("--landmarks", default="data/kaggle_asl/landmarks")
ap.add_argument("--words", type=int, default=10, help="new words the user teaches")
ap.add_argument("--shots", type=int, default=3)
ap.add_argument("--trials", type=int, default=40)
ap.add_argument("--per-word", type=int, default=12, help="test clips per word and trial")
ap.add_argument("--max-far", type=float, default=0.05, help="allowed share of other signs taken for a taught word")
ap.add_argument("--same-signer", action="store_true",
                help="test on the teacher's own other clips (a user teaching the phone their own signs)")
ap.add_argument("--bg", type=int, default=0,
                help="background bank: embeddings of this many clips of the model's own words, shipped with the app; "
                     "a taught word must beat the nearest of them by the margin")
ap.add_argument("--center", type=int, default=0, help="subtract the mean embedding of this many clips (0 = off)")
ap.add_argument("--out", default="results/teach_eval.json")
a = ap.parse_args()

ck = torch.load(a.ckpt, map_location="cpu")
model = build_model(ck["config"]).eval()
model.load_state_dict(ck["model"])
vocab = set(ck["vocab"])
cfg = ck["config"]

df = pd.read_csv(a.manifest, dtype={"video_id": str})
lm = Path(a.landmarks)
df = df[[(lm / f"{v}.npz").exists() for v in df.video_id]]
rng = np.random.default_rng(0)
new_words = sorted(set(df.gloss) - vocab)
known_words = sorted(set(df.gloss) & vocab)
print(f"{len(new_words)} signs not in the vocabulary, {len(known_words)} in it, {df.signer_id.nunique()} signers")

cache = {}


MEAN = 0.0


@torch.no_grad()
def raw(vid):
    ds = ClipDataset([lm / f"{vid}.npz"], [0], cfg["mode"], cfg["length"])
    return model.embed(ds[0][0][None]).numpy()[0]


def emb(vid):
    if vid not in cache:
        e = raw(vid) - MEAN
        cache[vid] = e / max(np.linalg.norm(e), 1e-6)
    return cache[vid]


if a.center:
    MEAN = np.mean([raw(v) for v in df.sample(a.center, random_state=1).video_id], 0)


BG = np.zeros((0, 1))
BG_IDS = set()
if a.bg:
    bgd = df[df.gloss.isin(known_words)].sample(a.bg, random_state=2)
    BG_IDS = set(bgd.video_id)
    BG = np.stack([emb(v) for v in bgd.video_id])


def match(e, bank):
    """bank: [(word, emb)] -> (word, similarity, margin)"""
    best = {}
    for w, s in bank:
        best[w] = max(best.get(w, -2.0), float(e @ s))
    r = sorted(best.items(), key=lambda kv: -kv[1])
    rival = r[1][1] if len(r) > 1 else -1.0
    if len(BG):
        rival = max(rival, float((BG @ e).max()))
    return r[0][0], r[0][1], r[0][1] - rival


rows = []  # (is_taught_word, right_word, similarity, margin)
for t in range(a.trials):
    words = list(rng.choice(new_words, a.words, replace=False))
    bank, tests = [], []
    for w in words:
        d = df[df.gloss == w]
        # the teacher: a signer with enough clips of this word
        counts = d.signer_id.value_counts()
        teacher = rng.choice(counts[counts >= a.shots + (2 if a.same_signer else 0)].index)
        teach = d[d.signer_id == teacher].sample(a.shots, random_state=int(rng.integers(1 << 30)))
        bank += [(w, emb(v)) for v in teach.video_id]
        others = (d[(d.signer_id == teacher) & ~d.video_id.isin(teach.video_id)] if a.same_signer
                  else d[d.signer_id != teacher])
        tests += [(w, v) for v in others.sample(min(a.per_word, len(others)), random_state=int(rng.integers(1 << 30))).video_id]
    # signs that were not taught: other new signs and the model's own words
    rest = df[~df.gloss.isin(words) & ~df.video_id.isin(BG_IDS)]
    tests += [(None, v) for v in rest.sample(a.words * a.per_word, random_state=int(rng.integers(1 << 30))).video_id]
    for truth, v in tests:
        w, sim, margin = match(emb(v), bank)
        rows.append((truth is not None, truth == w, sim, margin))
    print(f"trial {t + 1}/{a.trials}", end="\r", flush=True)
print()

r = np.array(rows, dtype=float)
taught, right, sim, margin = r[:, 0] > 0, r[:, 1] > 0, r[:, 2], r[:, 3]
table = []
for th in np.arange(0.50, 0.96, 0.025):
    for m in ([-0.05, 0.0, 0.02, 0.05, 0.08, 0.12] if a.bg else [0.0, 0.02, 0.03, 0.05]):
        acc = (sim >= th) & (margin >= m)
        table.append({"similarity": round(float(th), 3), "margin": m,
                      "recall": float((acc & right)[taught].mean()),      # taught word shown, and the right one
                      "wrong_word": float((acc & ~right)[taught].mean()),  # taught word signed, another taught word shown
                      "false_accept": float(acc[~taught].mean())})        # an untaught sign shown as a taught word
ok = [x for x in table if x["false_accept"] <= a.max_far]
best = max(ok, key=lambda x: (x["recall"] - x["wrong_word"], -x["similarity"]))
nearest = float(right[taught].mean())
report = {"checkpoint": a.ckpt, "words_taught": a.words, "shots": a.shots, "trials": a.trials,
          "test_clips_taught": int(taught.sum()), "test_clips_other": int((~taught).sum()),
          "same_signer": a.same_signer, "center": a.center, "background": a.bg, "nearest_neighbour_top1": round(nearest, 4), "chosen": best, "max_false_accept": a.max_far,
          "table": table,
          "note": "Kaggle ASL Signs, signs outside the model vocabulary; taught by one signer, tested on other signers"}
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(json.dumps(report, indent=1))
print(f"nearest taught word right (no threshold): {nearest:.1%}")
print(f"chosen: similarity >= {best['similarity']}, margin >= {best['margin']}: "
      f"{best['recall']:.1%} shown right, {best['wrong_word']:.1%} wrong word, {best['false_accept']:.1%} false accept")
