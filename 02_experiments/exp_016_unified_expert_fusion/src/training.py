from __future__ import annotations

"""Protected full-mode training primitives for exp016.

All functions in this module require the explicit full-mode authorization.  The
non-training modes use the matching model/feature constructors directly and
never import or invoke these routines.
"""

from dataclasses import dataclass
from typing import Iterable
import io

import numpy as np

from ..config import RunConfig, require_training
from .artifacts import atomic_bytes, atomic_npy
from .ranking import group_rank


@dataclass
class TrainHistory:
    loss: list[float]
    steps: int


def restore_torch_checkpoint(model, checkpoint_path) -> dict[str, object]:
    """Load a completed local checkpoint after strict architecture/finite checks."""
    import torch
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
        raise RuntimeError(f"Invalid checkpoint payload: {checkpoint_path}")
    expected, saved = model.state_dict(), checkpoint["model"]
    if set(expected) != set(saved):
        raise RuntimeError(f"Checkpoint architecture keys do not match: {checkpoint_path}")
    for name, tensor in saved.items():
        if tensor.shape != expected[name].shape or not bool(torch.isfinite(tensor).all()):
            raise RuntimeError(f"Checkpoint tensor contract failed for {name}: {checkpoint_path}")
    history = checkpoint.get("history")
    if not isinstance(history, list) or not history or not np.isfinite(np.asarray(history, np.float64)).all():
        raise RuntimeError(f"Checkpoint training history is missing or invalid: {checkpoint_path}")
    model.load_state_dict(saved, strict=True)
    return {"path": str(checkpoint_path), "steps": len(history), "reused": True}


def train_torch_model(config: RunConfig, model, batches: Iterable[tuple], loss_fn, epochs: int, learning_rate: float, checkpoint_path=None) -> TrainHistory:
    """Common explicit-training loop; called only by full pipeline stages."""
    require_training(config, f"{model.__class__.__name__} 训练")
    import torch
    if checkpoint_path is not None and checkpoint_path.is_file():
        restored = restore_torch_checkpoint(model, checkpoint_path)
        return TrainHistory(loss=[], steps=int(restored["steps"]))
    device = torch.device(config.device)
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    losses: list[float] = []
    batch_source = batches if callable(batches) else lambda: iter(batches)
    for _ in range(int(epochs)):
        epoch_steps = 0
        for batch in batch_source():
            epoch_steps += 1
            optimizer.zero_grad(set_to_none=True)
            batch = tuple(x.to(device) if isinstance(x, torch.Tensor) else x for x in batch)
            loss = loss_fn(model, *batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{model.__class__.__name__} 训练损失非有限。")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if epoch_steps == 0:
            raise RuntimeError("Training batch source yielded no batches.")
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        buffer = io.BytesIO()
        torch.save({"model": model.state_dict(), "history": losses}, buffer)
        atomic_bytes(checkpoint_path, buffer.getvalue())
    return TrainHistory(loss=losses, steps=len(losses))


def train_dual_axis(config: RunConfig, model, batches, checkpoint_path):
    import torch
    from .ranking_objectives import correlation_loss

    def objective(net, current, sequence, mask, anchor, target, groups):
        output = net(current, sequence, mask, anchor)
        return correlation_loss(output["direct"], target, groups) + correlation_loss(output["residual"], target, groups)

    return train_torch_model(config, model, batches, objective, epochs=8, learning_rate=2e-4, checkpoint_path=checkpoint_path)


def train_time_frequency(config: RunConfig, model, batches, checkpoint_path):
    import torch
    from .ranking_objectives import correlation_loss

    def objective(net, periodic, mask, anchor, confidence, periods, target, groups):
        scores = net(periodic, mask, anchor, confidence, periods)
        return correlation_loss(scores, target, groups)

    return train_torch_model(config, model, batches, objective, epochs=8, learning_rate=2e-4, checkpoint_path=checkpoint_path)


def train_graph(config: RunConfig, model, batches, checkpoint_path):
    from .ranking_objectives import correlation_loss

    def objective(net, state, neighbors, weights, category, anchor, target, groups):
        return correlation_loss(net(state, neighbors, weights, category, anchor), target, groups)

    return train_torch_model(config, model, batches, objective, epochs=8, learning_rate=2e-4, checkpoint_path=checkpoint_path)


def train_foundation(config: RunConfig, encoder, batches, checkpoint_path):
    from .self_supervised import self_supervised_loss

    def objective(net, sequence, mask, reconstruction_target, order_target, quantile_target, second_view, second_mask):
        first = net(sequence, mask)
        second = net(second_view, second_mask)
        consistency = (first["embedding"] - second["embedding"]).pow(2).mean()
        return self_supervised_loss(first, reconstruction_target, order_target, quantile_target) + 0.10 * consistency

    return train_torch_model(config, encoder, batches, objective, epochs=12, learning_rate=2e-4, checkpoint_path=checkpoint_path)


def train_foundation_head(config: RunConfig, encoder, head, batches, checkpoint_path):
    from .ranking_objectives import correlation_loss

    for parameter in encoder.parameters():
        parameter.requires_grad_(False)

    def objective(net, sequence, mask, target, groups):
        import torch
        with torch.no_grad():
            embedding = encoder(sequence, mask)["embedding"]
        return correlation_loss(net(embedding, mask.mean(1)), target, groups)

    return train_torch_model(config, head, batches, objective, epochs=8, learning_rate=2e-4, checkpoint_path=checkpoint_path)


def train_head(config: RunConfig, head, batches, checkpoint_path):
    from .ranking_objectives import multi_objective_loss

    def objective(net, experts, state, target, groups):
        output = net(experts, state)
        return multi_objective_loss(output, target, groups, anchor=experts[:, 0])

    return train_torch_model(config, head, batches, objective, epochs=12, learning_rate=3e-4, checkpoint_path=checkpoint_path)


def train_router(config: RunConfig, router, states, target_weights, checkpoint_path):
    import torch

    def objective(net, state, desired):
        output = net(state)
        return torch.nn.functional.smooth_l1_loss(output, desired) + 0.02 * output.var(dim=0).mean()

    return train_torch_model(config, router, [(states, target_weights)], objective, epochs=80, learning_rate=5e-4, checkpoint_path=checkpoint_path)


def save_family_prediction(path, values: np.ndarray, groups: np.ndarray) -> None:
    atomic_npy(path, group_rank(values, groups).astype(np.float32))


def predict_dual_axis(model, batches, device: str) -> np.ndarray:
    import torch
    outputs = []
    model.to(device).eval()
    with torch.no_grad():
        for current, sequence, mask, anchor, *_ in batches:
            result = model(current.to(device), sequence.to(device), mask.to(device), anchor.to(device))
            outputs.append(result["residual"].cpu().numpy())
    model.to("cpu"); torch.cuda.empty_cache()
    return np.concatenate(outputs).astype(np.float32)


def predict_time_frequency(model, batches, device: str) -> np.ndarray:
    import torch
    outputs = []
    model.to(device).eval()
    with torch.no_grad():
        for periodic, mask, anchor, confidence, periods, *_ in batches:
            outputs.append(model(periodic.to(device), mask.to(device), anchor.to(device), confidence.to(device), periods).cpu().numpy())
    model.to("cpu"); torch.cuda.empty_cache()
    return np.concatenate(outputs).astype(np.float32)


def predict_graph(model, batches, device: str) -> np.ndarray:
    import torch
    outputs = []
    model.to(device).eval()
    with torch.no_grad():
        for state, neighbors, weights, category, anchor, *_ in batches:
            outputs.append(model(state.to(device), neighbors.to(device), weights.to(device), category.to(device), anchor.to(device)).cpu().numpy())
    model.to("cpu"); torch.cuda.empty_cache()
    return np.concatenate(outputs).astype(np.float32)


def predict_foundation(encoder, head, batches, device: str) -> np.ndarray:
    import torch
    outputs = []
    encoder.to(device).eval(); head.to(device).eval()
    with torch.no_grad():
        for sequence, mask, *_ in batches:
            sequence, mask = sequence.to(device), mask.to(device)
            embedding = encoder(sequence, mask)["embedding"]
            outputs.append(head(embedding, mask.mean(1)).cpu().numpy())
    encoder.to("cpu"); head.to("cpu"); torch.cuda.empty_cache()
    return np.concatenate(outputs).astype(np.float32)
