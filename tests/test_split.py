import pandas as pd

from echogesture.wlasl import signer_split


def _df():
    rows = []
    for s in range(30):
        for c in range(10):
            rows.append({"video_id": f"{s:02d}{c}", "gloss": f"G{c}", "signer_id": s,
                         "official_split": "test" if (s % 7 == 0 and c < 3) else "train"})
    return pd.DataFrame(rows)


def test_no_holdout_is_unchanged():
    a = signer_split(_df(), 0.15, 0)
    b = signer_split(_df(), 0.15, 0, holdout_fraction=0.0)
    assert (a.split == b.split).all()
    assert "holdout" not in set(a.split)


def test_holdout_signers_are_disjoint():
    df = signer_split(_df(), 0.15, 0, holdout_fraction=0.15)
    signers = {s: set(df[df.split == s].signer_id) for s in ("train", "val", "holdout")}
    assert signers["holdout"]
    assert not signers["holdout"] & signers["train"]
    assert not signers["holdout"] & signers["val"]
    assert not signers["val"] & signers["train"]
    assert (df[df.official_split == "test"].split == "test").all()
    pool = df[df.official_split != "test"]
    assert 0.1 < (pool.split == "holdout").mean() < 0.25
