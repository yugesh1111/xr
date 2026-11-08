"""Confusion-aware loss.

Keeps a running estimate of which classes the model mixes up (from its own
softmax outputs during training) and adds a margin to those competitors'
logits for each sample. The model then has to beat its usual confusions by
a clear gap, not just barely. Training-only: inference cost is unchanged.
"""
import torch
from torch import nn
import torch.nn.functional as F


class ConfusionMarginLoss(nn.Module):
    def __init__(self, n_classes, margin=0.0, smoothing=0.1, decay=0.5):
        super().__init__()
        self.margin, self.smoothing, self.decay = margin, smoothing, decay
        self.register_buffer("sum", torch.zeros(n_classes, n_classes))
        self.register_buffer("count", torch.zeros(n_classes))

    def confusion(self):
        c = self.sum / self.count.clamp_min(1)[:, None]
        c = c * (1 - torch.eye(len(c), device=c.device))
        return c / c.max(1, keepdim=True).values.clamp_min(1e-6)

    def new_epoch(self):
        # forget slowly so the estimate follows the model as it improves
        self.sum.mul_(self.decay)
        self.count.mul_(self.decay)

    def forward(self, logits, y):
        if self.margin > 0:
            with torch.no_grad():
                self.sum.index_add_(0, y, logits.softmax(-1).float())
                self.count.index_add_(0, y, torch.ones_like(y, dtype=torch.float))
            logits = logits + self.margin * self.confusion()[y]
        return F.cross_entropy(logits, y, label_smoothing=self.smoothing)


class SupConLoss(nn.Module):
    """Supervised contrastive loss (Khosla et al., 2020): clips of the same
    sign are pulled together in embedding space, all other clips in the batch
    - including the signs it is usually confused with - are pushed away."""

    def __init__(self, temperature=0.1):
        super().__init__()
        self.t = temperature

    def forward(self, emb, y):
        z = F.normalize(emb, dim=-1)
        sim = z @ z.T / self.t
        self_mask = torch.eye(len(y), dtype=torch.bool, device=y.device)
        sim = sim.masked_fill(self_mask, -1e9)
        pos = (y[:, None] == y[None]) & ~self_mask
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        n_pos = pos.sum(1)
        has = n_pos > 0
        if not has.any():
            return emb.new_zeros(())
        return -((log_prob * pos).sum(1)[has] / n_pos[has]).mean()
