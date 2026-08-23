from __future__ import annotations

import torch
from torch.nn import functional as F


def _blocks(groups: torch.Tensor):
    offset = 0
    for size_value in groups.detach().cpu().tolist():
        size = int(size_value)
        if size <= 0:
            raise ValueError("Ranking groups must be positive.")
        yield slice(offset, offset + size)
        offset += size


def correlation_loss(scores: torch.Tensor, target: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
    losses = []
    for block in _blocks(groups):
        block_s, block_y = scores[block], target[block]
        centered_s, centered_y = block_s - block_s.mean(), block_y - block_y.mean()
        corr = (centered_s * centered_y).mean() / (
            centered_s.std(unbiased=False).clamp_min(1e-6) * centered_y.std(unbiased=False).clamp_min(1e-6)
        )
        losses.append(1.0 - corr)
    return torch.stack(losses).mean()


def pairwise_loss(scores: torch.Tensor, target: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
    """Deterministic stratified pairs, constructed strictly within each group."""
    losses = []
    for block in _blocks(groups):
        score, truth = scores[block], target[block]
        order = torch.argsort(truth)
        count = order.numel()
        if count < 2:
            continue
        quarter = max(1, count // 4)
        pair_sets = [
            (order[:quarter], order[-quarter:]),
            (order[quarter:2 * quarter], order[-2 * quarter:-quarter]),
            (order[:-1:2], order[1::2]),
        ]
        for left, right in pair_sets:
            usable = min(left.numel(), right.numel())
            if usable:
                direction = (truth[right[:usable]] - truth[left[:usable]]).sign()
                non_tie = direction != 0
                if bool(non_tie.any()):
                    margin = score[right[:usable]] - score[left[:usable]]
                    losses.append(F.softplus(-direction[non_tie] * margin[non_tie]).mean())
    return torch.stack(losses).mean() if losses else scores.sum() * 0.0


def tail_targets(target: torch.Tensor, groups: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    top, bottom = torch.zeros_like(target), torch.zeros_like(target)
    for block in _blocks(groups):
        truth = target[block]
        top[block] = (truth >= torch.quantile(truth, 0.90)).to(target.dtype)
        bottom[block] = (truth <= torch.quantile(truth, 0.10)).to(target.dtype)
    return top, bottom


def multi_objective_loss(
    outputs: dict[str, torch.Tensor], target: torch.Tensor, groups: torch.Tensor,
    anchor: torch.Tensor | None = None,
) -> torch.Tensor:
    scores = outputs["score"]
    top_target, bottom_target = tail_targets(target, groups)
    corr = correlation_loss(scores, target, groups)
    pair = pairwise_loss(scores, target, groups)
    tails = F.binary_cross_entropy_with_logits(outputs["top"], top_target)
    tails = tails + F.binary_cross_entropy_with_logits(outputs["bottom"], bottom_target)
    confidence_target = 1.0 - torch.clamp((scores.detach() - target).abs(), 0.0, 1.0)
    confidence = F.smooth_l1_loss(outputs["confidence"], confidence_target)
    residual = scores.sum() * 0.0
    if anchor is not None:
        residual = F.smooth_l1_loss(scores - anchor, target - anchor)
    return corr + 0.30 * pair + 0.10 * tails + 0.05 * confidence + 0.10 * residual
