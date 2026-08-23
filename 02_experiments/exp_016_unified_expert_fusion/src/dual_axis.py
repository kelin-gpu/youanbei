from __future__ import annotations

import torch
from torch import nn


class MultiScaleTimeEncoder(nn.Module):
    def __init__(self, features: int = 40, hidden: int = 64, scales: tuple[int, ...] = (20, 60, 240)):
        super().__init__()
        self.scales = scales
        self.input = nn.Linear(features, hidden)
        self.convs = nn.ModuleList([nn.Conv1d(hidden, hidden, 3, padding=1, groups=1) for _ in scales])
        self.gate = nn.Sequential(nn.Linear(hidden + 1, hidden), nn.GELU(), nn.Linear(hidden, len(scales)))

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # sequence: [batch, window, features], mask: [batch, window]
        encoded = self.input(sequence)
        parts = []
        for scale, conv in zip(self.scales, self.convs):
            visible = mask[:, -scale:].unsqueeze(-1)
            value = (encoded[:, -scale:] * visible).transpose(1, 2)
            pooled = (conv(value).transpose(1, 2) * visible).sum(1) / visible.sum(1).clamp_min(1.0)
            parts.append(pooled)
        stack = torch.stack(parts, dim=1)
        coverage = mask.mean(1, keepdim=True)
        logits = self.gate(torch.cat([stack.mean(1), coverage], dim=1))
        return (stack * torch.softmax(logits, dim=1).unsqueeze(-1)).sum(1)


class PrototypeInteraction(nn.Module):
    def __init__(self, hidden: int = 64, prototype_count: int = 32):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(prototype_count, hidden) * 0.02)
        self.proj = nn.Linear(hidden, hidden)

    def forward(self, state: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
        # state [stocks, hidden]; preserves permutation equivariance.
        keys = self.proj(state)
        score = self.queries @ keys.T / (keys.shape[-1] ** 0.5)
        if valid is not None:
            score = score.masked_fill(~valid.bool().unsqueeze(0), -1e4)
        prototypes = torch.softmax(score, dim=1) @ state
        stock_to_proto = torch.softmax(keys @ prototypes.T / (keys.shape[-1] ** 0.5), dim=1)
        return state + stock_to_proto @ prototypes


class DualAxisExpert(nn.Module):
    def __init__(self, current_dim: int, features: int = 40, hidden: int = 64):
        super().__init__()
        self.time = MultiScaleTimeEncoder(features=features, hidden=hidden)
        self.current = nn.Sequential(nn.Linear(current_dim, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.interaction = PrototypeInteraction(hidden=hidden)
        self.direct = nn.Linear(hidden, 1)
        self.residual = nn.Sequential(nn.Linear(hidden + 1, hidden), nn.GELU(), nn.Linear(hidden, 1), nn.Tanh())

    def forward(self, current: torch.Tensor, sequence: torch.Tensor, mask: torch.Tensor, anchor: torch.Tensor, valid: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        state = self.time(sequence, mask) + self.current(current)
        state = self.interaction(state, valid)
        direct = self.direct(state).squeeze(-1)
        correction = 0.20 * self.residual(torch.cat([state, anchor[:, None]], dim=1)).squeeze(-1)
        return {"direct": direct, "residual": anchor + correction, "state": state}
