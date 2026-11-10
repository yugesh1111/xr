"""Dataset over landmark .npz files listed in a manifest."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .augment import augment
from .inputs import make_input


def load_split(manifest, landmark_dir, split, vocab=None):
    """Rows of `split` whose landmark file exists -> (paths, labels, vocab)."""
    df = pd.read_csv(manifest, dtype={"video_id": str})
    df = df[df.split == split]
    landmark_dir = Path(landmark_dir)
    df = df[[(landmark_dir / f"{v}.npz").exists() for v in df.video_id]]
    if vocab is None:
        vocab = sorted(df.gloss.unique())
    index = {g: i for i, g in enumerate(vocab)}
    df = df[df.gloss.isin(index)]
    paths = [landmark_dir / f"{v}.npz" for v in df.video_id]
    return paths, np.array([index[g] for g in df.gloss]), vocab, df


class ClipDataset(Dataset):
    def __init__(self, paths, labels, mode="canonical", length=32, train=False, seed=0, aug="default",
                 alt_dirs=(), p_alt=0.5, p_close=0.0, p_far=0.0):
        self.paths, self.labels = list(paths), np.asarray(labels)
        # landmarks of the same clip extracted from degraded video (training only)
        self.alts = [[Path(d) / Path(p).name for d in alt_dirs if (Path(d) / Path(p).name).exists()]
                     for p in self.paths]
        self.p_alt = p_alt
        self.mode, self.length, self.train = mode, length, train
        self.seed, self.aug = seed, aug
        self.p_close = p_close  # share of training samples turned into phone close-ups
        self.p_far = p_far      # ... and into a signer far from the phone
        self._cache = {}

    def __len__(self):
        return len(self.paths)

    def xyz(self, i, path=None):
        path = path or self.paths[i]
        if path not in self._cache:  # raw landmarks are small; keep them in RAM
            self._cache[path] = np.load(path)["xyz"]
        return self._cache[path].astype(np.float32)

    def __getitem__(self, i):
        rng = np.random.default_rng()
        path = None
        if self.train and self.alts[i] and rng.random() < self.p_alt:
            path = self.alts[i][rng.integers(len(self.alts[i]))]
        x = self.xyz(i, path)
        if len(x) == 0:
            x = self.xyz(i)
        if self.train:
            x = augment(x, rng, self.aug, p_close=self.p_close, p_far=self.p_far)
            x = make_input(x, self.mode, self.length, uniform_weight=rng.uniform(0.3, 0.8))
        else:
            x = make_input(x, self.mode, self.length)
        return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)), int(self.labels[i])


class ClassBalancedSampler(torch.utils.data.Sampler):
    """Batches of P classes x K clips each, so every batch has positives
    for the contrastive loss and frequent signs don't dominate."""

    def __init__(self, labels, p=16, k=4, seed=0):
        self.labels, self.p, self.k = np.asarray(labels), p, k
        self.by_class = {c: np.flatnonzero(self.labels == c) for c in np.unique(self.labels)}
        self.rng = np.random.default_rng(seed)
        self.n_batches = len(self.labels) // (p * k)

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        classes = list(self.by_class)
        for _ in range(self.n_batches):
            batch = []
            for c in self.rng.choice(classes, self.p, replace=False):
                idx = self.by_class[c]
                batch += list(self.rng.choice(idx, self.k, replace=len(idx) < self.k))
            yield batch
