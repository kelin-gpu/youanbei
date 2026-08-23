from __future__ import annotations

import torch
from torch import nn


class StateRouter(nn.Module):
    def __init__(self, state_dim: int, family_count: int, base_weights: torch.Tensor, min_weights: torch.Tensor, hidden: int = 32):
        super().__init__()
        if family_count != base_weights.numel() or family_count != min_weights.numel():
            raise ValueError("Router family dimensions do not match its priors.")
        if float(min_weights.sum()) >= 1.0 or bool(torch.any(min_weights <= 0)):
            raise ValueError("Router minima must be positive and leave free probability mass.")
        self.network = nn.Sequential(nn.Linear(state_dim, hidden), nn.GELU(), nn.Linear(hidden, family_count))
        self.register_buffer("base_weights", base_weights / base_weights.sum())
        self.register_buffer("min_weights", min_weights)

    def forward(self, state: torch.Tensor, confidence: torch.Tensor | None = None) -> torch.Tensor:
        free = 1.0 - self.min_weights.sum()
        dynamic = self.min_weights + free * torch.softmax(self.network(state), dim=-1)
        if confidence is None:
            confidence = torch.ones(state.shape[0], device=state.device)
        confidence = confidence.clamp(0.0, 1.0)
        output = confidence[:, None] * dynamic + (1.0 - confidence[:, None]) * self.base_weights
        return output / output.sum(dim=-1, keepdim=True)


def cross_section_state_features(
    current: torch.Tensor, mask: torch.Tensor, expert_predictions: torch.Tensor, groups: torch.Tensor,
    periodic_confidence: torch.Tensor | None = None, graph_stability: torch.Tensor | None = None,
) -> torch.Tensor:
    """Aggregate coverage/drift/structure/disagreement to one row per time group."""
    rows, offset = [], 0
    for group_index, size_value in enumerate(groups.detach().cpu().tolist()):
        size = int(size_value)
        x = current[offset:offset + size]
        visible = mask[offset:offset + size]
        experts = expert_predictions[offset:offset + size]
        periodic = x.new_tensor(0.0) if periodic_confidence is None else periodic_confidence[offset:offset + size].mean()
        stability = x.new_tensor(1.0) if graph_stability is None else graph_stability[group_index]
        row = torch.stack([
            visible.mean(),
            x.mean(),
            x.std(unbiased=False),
            x.abs().mean(),
            x.std(dim=0, unbiased=False).mean(),
            periodic,
            stability,
            experts.std(dim=1, unbiased=False).mean(),
        ])
        rows.append(row)
        offset += size
    if offset != current.shape[0]:
        raise ValueError("Router groups do not cover the current panel.")
    return torch.stack(rows)


def expand_group_values(values: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
    return torch.repeat_interleave(values, groups.to(values.device), dim=0)


# Backward-compatible name used by callers, now explicitly cross-sectional.
state_features = cross_section_state_features
