"""EchoNet: factorized, quality-gated sign classifier.

  input (B, T, D)  -- concatenated feature groups, see inputs.py
    -> per-group projection to d  (handshape, orientation, location, ...)
    -> quality gate: per frame, per group weight from the quality features
       (e.g. dominant hand not seen -> trust its handshape less)
    -> conv blocks (local motion) + attention blocks (whole-sign context)
    -> mean pool -> embedding -> classifier

fusion="flat" collapses the first two steps into one linear layer
(used for the E1/E2 baselines). Everything is plain Conv1d/Linear/attention
so it exports to ONNX/TFLite and quantizes to INT8.
"""
import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, d, kernel=7, expand=2, dropout=0.1):
        super().__init__()
        h = d * expand
        self.net = nn.Sequential(
            nn.Conv1d(d, h, 1), nn.SiLU(),
            nn.Conv1d(h, h, kernel, padding=kernel // 2, groups=h),
            nn.BatchNorm1d(h), nn.SiLU(),
            nn.Conv1d(h, d, 1), nn.Dropout(dropout),
        )

    def forward(self, x):  # (B, T, d)
        return x + self.net(x.transpose(1, 2)).transpose(1, 2)


class AttnBlock(nn.Module):
    def __init__(self, d, heads=4, dropout=0.1):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.n1(x)
        x = x + self.drop(self.attn(h, h, h, need_weights=False)[0])
        return x + self.drop(self.ff(self.n2(x)))


class EchoNet(nn.Module):
    def __init__(self, slices, n_classes, d=192, fusion="gated", n_conv=4, n_attn=2,
                 length=32, dropout=0.2):
        super().__init__()
        self.slices = {k: tuple(v) for k, v in slices.items()}
        self.fusion = fusion
        in_dim = max(stop for _, stop in self.slices.values())
        self.register_buffer("mu", torch.zeros(in_dim))
        self.register_buffer("sigma", torch.ones(in_dim))

        if fusion == "flat":
            self.proj = nn.Linear(in_dim, d)
        else:
            self.groups = [g for g in self.slices if g != "quality"]
            self.proj = nn.ModuleDict({g: nn.Sequential(nn.Linear(b - a, d), nn.LayerNorm(d))
                                       for g, (a, b) in self.slices.items()})
            if fusion == "gated":
                qa, qb = self.slices["quality"]
                self.gate = nn.Linear(qb - qa, len(self.groups))
                nn.init.zeros_(self.gate.weight)
                nn.init.constant_(self.gate.bias, 2.0)  # start (almost) fully open
        self.pos = nn.Parameter(torch.zeros(1, length, d))
        blocks = []
        for i in range(n_conv + n_attn):
            # interleave: conv, conv, attn, conv, conv, attn ...
            blocks.append(AttnBlock(d, dropout=dropout / 2) if (i + 1) % 3 == 0 and n_attn
                          else ConvBlock(d, dropout=dropout / 2))
        self.blocks = nn.Sequential(*blocks)
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d, n_classes)

    def set_normalization(self, mu, sigma):
        self.mu.copy_(torch.as_tensor(mu, dtype=torch.float32))
        self.sigma.copy_(torch.as_tensor(sigma, dtype=torch.float32).clamp_min(1e-3))

    def embed(self, x):
        raw = x
        x = (x - self.mu) / self.sigma
        if self.fusion == "flat":
            h = self.proj(x)
        else:
            parts = [self.proj[g](x[..., a:b]) for g, (a, b) in self.slices.items() if g != "quality"]
            if self.fusion == "gated":
                qa, qb = self.slices["quality"]
                w = torch.sigmoid(self.gate(raw[..., qa:qb]))  # (B, T, n_groups)
                parts = [p * w[..., i:i + 1] for i, p in enumerate(parts)]
            h = torch.stack(parts, 0).sum(0)
            if "quality" in self.slices:
                qa, qb = self.slices["quality"]
                h = h + self.proj["quality"](x[..., qa:qb])
        h = self.blocks(h + self.pos[:, :h.shape[1]])
        return self.norm(h.mean(1))

    def forward(self, x):
        return self.head(self.drop(self.embed(x)))


def build_model(cfg):
    """Rebuild a model from the config saved in a checkpoint."""
    return EchoNet(cfg["slices"], cfg["n_classes"], d=cfg["dim"], fusion=cfg["fusion"],
                   n_conv=cfg.get("n_conv", 4), n_attn=cfg.get("n_attn", 2),
                   length=cfg["length"], dropout=cfg.get("dropout", 0.2))


class Ensemble(nn.Module):
    """Averages the class probabilities of several models that share the same
    vocabulary and input. Exported as one ONNX graph, so the app and main.py
    run it like a single model. Output is log of the mean probability, so
    argmax/softmax downstream behave as usual."""

    def __init__(self, models):
        super().__init__()
        self.members = nn.ModuleList(models)

    def forward(self, x):
        p = torch.stack([m(x).softmax(-1) for m in self.members]).mean(0)
        return torch.log(p + 1e-9)
