from __future__ import annotations

import torch
from torch import nn


class MultiObjectiveRankHead(nn.Module):
    def __init__(self, expert_count: int = 6, state_dim: int = 8, hidden: int = 48):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(expert_count + state_dim + 2, hidden), nn.GELU(), nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU())
        self.score = nn.Linear(hidden, 1)
        self.top = nn.Linear(hidden, 1)
        self.bottom = nn.Linear(hidden, 1)
        self.confidence = nn.Sequential(nn.Linear(hidden, 1), nn.Sigmoid())

    def forward(self, experts: torch.Tensor, state: torch.Tensor) -> dict[str, torch.Tensor]:
        spread = experts.std(dim=1, keepdim=True)
        mean = experts.mean(dim=1, keepdim=True)
        hidden = self.backbone(torch.cat([experts, state, mean, spread], dim=1))
        return {"score": self.score(hidden).squeeze(-1), "top": self.top(hidden).squeeze(-1), "bottom": self.bottom(hidden).squeeze(-1), "confidence": self.confidence(hidden).squeeze(-1)}
