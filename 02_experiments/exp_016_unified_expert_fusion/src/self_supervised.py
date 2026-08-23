from __future__ import annotations

import torch
from torch import nn

from .dual_axis import MultiScaleTimeEncoder


class SelfSupervisedEncoder(nn.Module):
    """本地无标签预训练编码器；共享双轴时间编码逻辑。"""
    def __init__(self, features: int = 40, hidden: int = 64):
        super().__init__()
        self.encoder = MultiScaleTimeEncoder(features=features, hidden=hidden)
        self.reconstruct = nn.Linear(hidden, features)
        self.order = nn.Linear(hidden, 2)
        self.quantile = nn.Linear(hidden, 5)

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        embedding = self.encoder(sequence, mask)
        return {"embedding": embedding, "reconstruction": self.reconstruct(embedding), "order": self.order(embedding), "quantile": self.quantile(embedding)}


def self_supervised_loss(outputs: dict[str, torch.Tensor], target: torch.Tensor, order_target: torch.Tensor, quantile_target: torch.Tensor) -> torch.Tensor:
    reconstruction = nn.functional.smooth_l1_loss(outputs["reconstruction"], target)
    order = nn.functional.cross_entropy(outputs["order"], order_target)
    quantile = nn.functional.cross_entropy(outputs["quantile"], quantile_target)
    return reconstruction + 0.20 * order + 0.20 * quantile


class FoundationRepresentationHead(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(hidden + 1, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, embedding: torch.Tensor, coverage: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([embedding, coverage[:, None]], dim=1)).squeeze(-1)
