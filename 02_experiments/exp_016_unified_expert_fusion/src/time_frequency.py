from __future__ import annotations

import numpy as np
import torch
from torch import nn


def fit_period_library(history: np.ndarray, mask: np.ndarray, period_count: int = 8) -> np.ndarray:
    """Fit a global period library from the supplied Train-only histories."""
    x, observed = np.asarray(history, np.float32), np.asarray(mask, np.float32)
    centered = (x - (x * observed[..., None]).sum(1, keepdims=True) /
                np.maximum(observed.sum(1, keepdims=True)[..., None], 1.0)) * observed[..., None]
    energy = np.abs(np.fft.rfft(centered, axis=1)).mean(axis=(0, 2))
    mask_energy = np.abs(np.fft.rfft(observed, axis=1)).mean(axis=0)
    corrected = energy / np.maximum(mask_energy, np.median(mask_energy[1:]) * 0.25 + 1e-6)
    frequencies = np.arange(corrected.size)
    periods = np.divide(history.shape[1], frequencies, out=np.full_like(corrected, np.inf), where=frequencies > 0)
    allowed = (periods >= 4) & (periods <= 120)
    candidates = np.where(allowed)[0]
    selected = candidates[np.argsort(corrected[candidates])[-min(period_count, candidates.size):]]
    result = np.unique(np.rint(periods[selected]).astype(np.int64))
    if result.size == 0:
        raise RuntimeError("No usable periods were found in the Train-only period library.")
    return result


def causal_decompose(history: np.ndarray, mask: np.ndarray, period_count: int = 8, periods: np.ndarray | None = None) -> dict[str, np.ndarray]:
    x, observed = np.asarray(history, np.float32), np.asarray(mask, np.float32)
    valid = observed[..., None]
    trend = np.cumsum(x * valid, axis=1) / np.maximum(np.cumsum(valid, axis=1), 1.0)
    periodic = (x - trend) * valid
    shock = periodic[:, -1] / (np.std(periodic, axis=1) + 1e-5)
    library = fit_period_library(x, observed, period_count) if periods is None else np.asarray(periods, np.int64)
    spectrum = np.abs(np.fft.rfft(periodic, axis=1)).mean(axis=2)
    bins = np.clip(np.rint(x.shape[1] / library).astype(np.int64), 1, spectrum.shape[1] - 1)
    selected_energy = spectrum[:, bins].mean(axis=1)
    total_energy = spectrum[:, 1:].mean(axis=1) + 1e-5
    coverage = observed.mean(axis=1)
    confidence = np.clip(selected_energy / total_energy, 0.0, 4.0) / 4.0 * coverage
    return {"trend": trend[:, -1], "periodic": periodic, "shock": shock, "periods": library,
            "confidence": confidence.astype(np.float32)}


class TimeFrequencyExpert(nn.Module):
    def __init__(self, features: int = 40, hidden: int = 48):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(2, hidden, 3, padding=1), nn.GELU(), nn.AdaptiveAvgPool2d((1, 1)))
        self.head = nn.Sequential(nn.Linear(hidden + 1, hidden), nn.GELU(), nn.Linear(hidden, 1), nn.Tanh())

    def _fold(self, periodic: torch.Tensor, mask: torch.Tensor, period: int) -> torch.Tensor:
        length = max(period, (periodic.shape[1] // period) * period)
        value = periodic[:, -length:, :min(periodic.shape[2], 16)].mean(-1)
        visible = mask[:, -length:]
        rows = length // period
        return torch.stack([value.reshape(value.shape[0], rows, period), visible.reshape(visible.shape[0], rows, period)], dim=1)

    def forward(self, periodic: torch.Tensor, mask: torch.Tensor, anchor: torch.Tensor, confidence: torch.Tensor,
                periods: torch.Tensor | np.ndarray | None = None) -> torch.Tensor:
        period_values = [15] if periods is None else [int(value) for value in torch.as_tensor(periods).detach().cpu().tolist()]
        hidden = torch.stack([self.conv(self._fold(periodic, mask, period)).flatten(1) for period in period_values]).mean(0)
        return anchor + 0.20 * self.head(torch.cat([hidden, confidence[:, None]], dim=1)).squeeze(-1)
