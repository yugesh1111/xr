"""WLASL metadata -> manifest with a signer-independent validation split.

The official WLASL train/val/test split shares signers between train and test,
so its validation accuracy flatters a model. We hold out whole signers for
validation instead; the organisers' hidden test set is most likely made of
people the model has never seen.
"""
import json
import random
from pathlib import Path

import pandas as pd

WLASL_JSON_URL = "https://raw.githubusercontent.com/dxli94/WLASL/master/start_kit/WLASL_v0.3.json"


def load_entries(json_path):
    return json.loads(Path(json_path).read_text())


def build_manifest(entries, vocab_size=None, vocab=None):
    """One row per clip. WLASL orders glosses by frequency, so the first
    100 entries are the WLASL100 subset, the first 300 are WLASL300, etc.
    `vocab` (an explicit list of glosses, e.g. the organisers' list) wins
    over `vocab_size`."""
    if vocab is not None:
        wanted = {g.strip().upper() for g in vocab}
        entries = [e for e in entries if e["gloss"].upper() in wanted]
        missing = wanted - {e["gloss"].upper() for e in entries}
        if missing:
            print(f"warning: {len(missing)} glosses not in WLASL: {sorted(missing)[:20]}")
    elif vocab_size:
        entries = entries[:vocab_size]

    rows = []
    for label, e in enumerate(entries):
        for i in e["instances"]:
            rows.append({
                "video_id": i["video_id"],
                "gloss": e["gloss"].upper(),
                "label": label,
                "signer_id": i["signer_id"],
                "official_split": i["split"],
                "url": i["url"],
                "frame_start": i["frame_start"],
                "frame_end": i["frame_end"],
                "fps": i["fps"],
                "source": i["source"],
                "variation_id": i["variation_id"],
            })
    return pd.DataFrame(rows)


def signer_split(df, val_fraction=0.15, seed=0):
    """Adds column `split`: official test stays test; the remaining clips are
    divided into train/val with no signer appearing in both.
    Also adds `test_signer_seen` so test accuracy can be reported separately
    for seen and unseen signers."""
    df = df.copy()
    pool = df[df.official_split != "test"]
    counts = pool.signer_id.value_counts()
    signers = list(counts.index)
    random.Random(seed).shuffle(signers)
    target, n, val_signers = val_fraction * len(pool), 0, set()
    for s in signers:
        if n >= target:
            break
        # skip the few signers that dominate the data; holding one out would
        # remove a large slice of the vocabulary from training
        if counts[s] > 0.2 * len(pool):
            continue
        val_signers.add(s)
        n += counts[s]
    df["split"] = "train"
    df.loc[df.signer_id.isin(val_signers) & (df.official_split != "test"), "split"] = "val"
    df.loc[df.official_split == "test", "split"] = "test"
    train_signers = set(df.loc[df.split == "train", "signer_id"])
    df["test_signer_seen"] = df.signer_id.isin(train_signers) & (df.split == "test")
    return df


def summarize(df):
    lines = [f"clips: {len(df)}, glosses: {df.gloss.nunique()}, signers: {df.signer_id.nunique()}"]
    for s in ("train", "val", "test"):
        part = df[df.split == s]
        lines.append(f"  {s:5s}: {len(part):6d} clips, {part.signer_id.nunique():3d} signers, "
                     f"{part.gloss.nunique():4d} glosses")
    val_cov = df[df.split == "val"].gloss.nunique() / max(df.gloss.nunique(), 1)
    lines.append(f"  val covers {val_cov:.0%} of the vocabulary")
    test = df[df.split == "test"]
    if len(test):
        lines.append(f"  test clips from signers unseen in train: {(~test.test_signer_seen).mean():.0%}")
    return "\n".join(lines)
