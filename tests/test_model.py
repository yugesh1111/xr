import numpy as np
import torch

from echogesture.augment import augment
from echogesture.inputs import group_slices, make_input
from echogesture.losses import ConfusionMarginLoss
from echogesture.model import EchoNet
from tests.test_features import fake_clip


def test_all_ablation_configs_run():
    for mode, fusion in [("raw", "flat"), ("canonical", "flat"), ("canonical", "sum"), ("canonical", "gated")]:
        s = group_slices(mode)
        x = torch.from_numpy(np.stack([make_input(fake_clip(seed=i), mode) for i in range(3)]))
        assert EchoNet(s, 10, d=64, fusion=fusion)(x).shape == (3, 10)


def test_augmented_clip_still_gives_finite_input():
    rng = np.random.default_rng(1)
    for _ in range(20):
        assert np.isfinite(make_input(augment(fake_clip(), rng))).all()


def test_confusion_margin_targets_confused_class():
    loss = ConfusionMarginLoss(3, margin=1.0)
    y = torch.zeros(8, dtype=torch.long)
    logits = torch.tensor([[2.0, 1.5, -3.0]]).repeat(8, 1)  # class 0 mixed up with class 1
    loss(logits, y)
    c = loss.confusion()
    assert c[0, 1] > c[0, 2]
    assert loss(logits, y) > ConfusionMarginLoss(3, margin=0.0)(logits, y)
