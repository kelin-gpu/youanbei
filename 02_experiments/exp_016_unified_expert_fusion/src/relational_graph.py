from __future__ import annotations

import numpy as np
import torch
from torch import nn


def knn_graph(state: np.ndarray, neighbor_count: int = 16, prototype_count: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Prototype-constrained sparse KNN; avoids a full stock-by-stock matrix."""
    state = np.asarray(state, np.float32)
    if state.ndim != 2 or state.shape[0] < 2:
        raise ValueError("Sparse KNN requires at least two stock rows.")
    norm = state / np.maximum(np.linalg.norm(state, axis=1, keepdims=True), 1e-6)
    count = min(neighbor_count, max(1, state.shape[0] - 1))
    prototypes = min(prototype_count, max(1, state.shape[0] // (count + 1)))
    centers = norm[np.linspace(0, state.shape[0] - 1, prototypes, dtype=np.int64)].copy()
    for _ in range(3):
        assignment = np.argmax(norm @ centers.T, axis=1)
        for group in range(prototypes):
            selected = assignment == group
            if selected.any():
                center = norm[selected].mean(0); centers[group] = center / max(np.linalg.norm(center), 1e-6)
    prototype_order = np.argsort(-(centers @ centers.T), axis=1)
    index = np.empty((state.shape[0], count), dtype=np.int32)
    weight = np.empty((state.shape[0], count), dtype=np.float32)
    members = [np.where(assignment == group)[0] for group in range(prototypes)]
    for row in range(state.shape[0]):
        candidates = []
        for group in prototype_order[assignment[row]]:
            candidates.extend(members[int(group)].tolist())
            if len(candidates) >= count + 1:
                break
        candidate = np.asarray([value for value in candidates if value != row], dtype=np.int64)
        similarity = norm[candidate] @ norm[row]
        chosen = np.argpartition(-similarity, count - 1)[:count]
        index[row] = candidate[chosen]
        weight[row] = similarity[chosen]
    weight = np.exp(weight - weight.max(axis=1, keepdims=True)); weight /= weight.sum(axis=1, keepdims=True)
    return index.astype(np.int32), weight.astype(np.float32)


def category_context(state: np.ndarray, categories: np.ndarray) -> np.ndarray:
    output = np.zeros_like(state, dtype=np.float32)
    categories = np.asarray(categories, np.int64)
    for col in range(categories.shape[1]):
        for value in np.unique(categories[:, col]):
            group = categories[:, col] == value
            output[group] += state[group].mean(axis=0, keepdims=True)
    return output / max(1, categories.shape[1])


def apply_sparse_lead_lag(history: np.ndarray, mask: np.ndarray, neighbors: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Add directed lag evidence only on already sparse candidate edges."""
    values = np.asarray(history, np.float32)[:, :, 0]
    observed = np.asarray(mask, np.float32)
    result = np.asarray(weights, np.float32).copy()
    for row in range(values.shape[0]):
        candidate = neighbors[row]
        target = values[row, 1:]
        target_mask = observed[row, 1:]
        source = values[candidate, :-1]
        usable = observed[candidate, :-1] * target_mask[None, :]
        source_mean = (source * usable).sum(1) / np.maximum(usable.sum(1), 1.0)
        target_mean = (target[None, :] * usable).sum(1) / np.maximum(usable.sum(1), 1.0)
        numerator = ((source - source_mean[:, None]) * (target[None, :] - target_mean[:, None]) * usable).sum(1)
        denominator = np.sqrt((((source - source_mean[:, None]) ** 2) * usable).sum(1) *
                              (((target[None, :] - target_mean[:, None]) ** 2) * usable).sum(1)) + 1e-6
        lag = np.maximum(numerator / denominator, 0.0)
        if lag.sum() > 0:
            lag /= lag.sum()
            result[row] = 0.70 * result[row] + 0.30 * lag
    return result / np.maximum(result.sum(1, keepdims=True), 1e-6)


class RelationalGraphExpert(nn.Module):
    def __init__(self, dim: int, hidden: int = 64):
        super().__init__()
        self.self_proj = nn.Linear(dim, hidden)
        self.neighbor_proj = nn.Linear(dim, hidden)
        self.category_proj = nn.Linear(dim, hidden)
        self.second_self = nn.Linear(hidden, hidden)
        self.second_neighbor = nn.Linear(hidden, hidden)
        self.head = nn.Sequential(nn.GELU(), nn.Linear(hidden, 1), nn.Tanh())

    def forward(self, state: torch.Tensor, neighbors: torch.Tensor, weights: torch.Tensor, category: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        messages = state[neighbors]
        neighbor = (messages * weights.unsqueeze(-1)).sum(1)
        hidden = self.self_proj(state) + self.neighbor_proj(neighbor) + self.category_proj(category)
        second_message = hidden[neighbors]
        second_neighbor = (second_message * weights.unsqueeze(-1)).sum(1)
        hidden = self.second_self(hidden) + self.second_neighbor(second_neighbor)
        return anchor + 0.20 * self.head(hidden).squeeze(-1)
