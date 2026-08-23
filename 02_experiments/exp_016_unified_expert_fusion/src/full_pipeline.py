from __future__ import annotations

"""Authorized full stage graph for exp016.

Every supervised family follows fit -> predict on a disjoint interval.  The
final artifact gate refuses to write prediction.npy unless every family has a
finite, non-degenerate Test vector and cross-sectional routed weights.
"""

import time
from pathlib import Path

import numpy as np

from ..config import (BASE_WEIGHTS, CACHE_DIR, FAMILIES, MIN_WEIGHTS, OOF_FOLDS,
                      RESULT_DIR, RunConfig, STOCK_COUNT, TEST_START, TEST_STOP, VALID_START, VALID_STOP,
                      require_training)
from .artifacts import atomic_json, atomic_npy, sha256
from .data_context import DataContext
from .feature_views import robust_rank_features, state_features
from .oof_anchor import (load_exp015_anchor, load_exp015_test_anchor, predict_exp015_anchor,
                         save_exp015_anchor, train_exp015_anchor)
from .prediction_contract import validate_prediction, vector_to_grid
from .ranking import dynamic_blend_family_predictions, group_rank, rank_ic
from .tabular_experts import (load_tabular_family, predict_tabular_family,
                              save_tabular_family, train_tabular_family)


def _anchor_train_or_load(config, X, relevance, groups, path):
    path = Path(path)
    if path.is_file() and path.stat().st_size > 0:
        return load_exp015_anchor(path), True
    model = train_exp015_anchor(config, X, relevance, groups)
    save_exp015_anchor(model, path)
    return model, False


def _tabular_train_or_load(config, X, target, relevance, groups, directory):
    directory = Path(directory)
    try:
        return load_tabular_family(directory), True
    except (FileNotFoundError, RuntimeError, ValueError):
        models = train_tabular_family(config, X, target, relevance, groups)
        save_tabular_family(models, directory)
        return models, False


def _capped_positions(groups: np.ndarray, cap: int) -> list[np.ndarray]:
    offset, result = 0, []
    for size_value in groups:
        size, take = int(size_value), min(int(size_value), int(cap))
        result.append(offset + np.linspace(0, size - 1, take, dtype=np.int64))
        offset += size
    return result


def time_slice_batches(ctx: DataContext, split: str, start: int, stop: int, cap: int, window: int = 240):
    import torch
    rows, groups = ctx.row_slice(split, start, stop)
    positions = _capped_positions(groups, cap)
    times = np.asarray(ctx.common[split]["time"][rows], dtype=np.int32)
    stocks = np.asarray(ctx.common[split]["stock"][rows], dtype=np.int32)
    cursor = 0
    for size_value, pick in zip(groups, positions):
        size = int(size_value)
        absolute = rows.start + pick
        local_stocks = stocks[cursor:cursor + size][pick - cursor]
        now = int(times[cursor])
        history, mask = ctx.causal_history(now, local_stocks, window=window)
        tree = np.asarray(ctx.tree[split][absolute, :], dtype=np.float32)
        # Cross-sectional transforms must be fitted on the complete time slice;
        # computing ranks after cap sampling would shift Train versus Test.
        full_tree = np.asarray(ctx.tree[split][rows.start + cursor:rows.start + cursor + size, :64], dtype=np.float32)
        full_current = robust_rank_features(full_tree, numeric_count=40)
        current = full_current[pick - cursor]
        target = None if split == "test" else np.asarray(ctx.common[split]["y"][absolute], dtype=np.float32)
        yield {"time": now, "stocks": local_stocks, "absolute_rows": absolute, "tree": tree,
               "categories": np.rint(tree[:, 408:417]).astype(np.int64),
               "current": torch.from_numpy(current), "sequence": torch.from_numpy(history),
               "mask": torch.from_numpy(mask), "target": None if target is None else torch.from_numpy(target),
               "groups": torch.tensor([local_stocks.size], dtype=torch.long)}
        cursor += size


def _tabular_arrays(ctx: DataContext, split: str, start: int, stop: int, cap: int):
    rows, groups = ctx.row_slice(split, start, stop)
    selected = _capped_positions(groups, cap)
    indices = rows.start + np.concatenate(selected)
    capped = np.asarray([part.size for part in selected], dtype=np.int32)
    target = None if split == "test" else np.asarray(ctx.common[split]["y"][indices], dtype=np.float32)
    relevance = None if split == "test" else np.asarray(ctx.common[split]["relevance"][indices], dtype=np.int32)
    return np.asarray(ctx.tree[split][indices, :], dtype=np.float32), target, relevance, capped, indices


def _combined_supervised_arrays(ctx: DataContext, cap: int):
    train = _tabular_arrays(ctx, "train", 486, VALID_START, cap)
    valid = _tabular_arrays(ctx, "valid", VALID_START, VALID_STOP, cap)
    return (np.concatenate([train[0], valid[0]]), np.concatenate([train[1], valid[1]]),
            np.concatenate([train[2], valid[2]]), np.concatenate([train[3], valid[3]]))


def _combined_batch_factory(ctx, cap, train_anchor, valid_anchor, period_library=None):
    yield from _torch_batches(ctx, "train", 486, VALID_START, cap, train_anchor, period_library)
    yield from _torch_batches(ctx, "valid", VALID_START, VALID_STOP, cap, valid_anchor, period_library)


def _write_manifest(ctx: DataContext, config: RunConfig, output: Path) -> None:
    formal = Path(ctx.dataset_dir).parents[1] / "04_results" / "final_submission" / "prediction.npy"
    atomic_json(output / "run_manifest.json", {"experiment": "exp_016_unified_expert_fusion", "mode": config.mode,
        "manifest_sha256": ctx.manifest_sha256, "oof_folds": OOF_FOLDS, "families": FAMILIES,
        "training_authorization": True, "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_final_submission_sha256": sha256(formal)})


def _torch_batches(ctx, split, start, stop, cap, anchor_values=None, period_library=None):
    """Materialize capped cross-sections and derived inputs in one consistent order."""
    import torch
    from .relational_graph import apply_sparse_lead_lag, category_context, knn_graph
    from .time_frequency import causal_decompose

    anchor_offset = 0
    for probe in time_slice_batches(ctx, split, start, stop, cap):
        count = probe["stocks"].size
        anchor = torch.zeros(count) if anchor_values is None else torch.from_numpy(np.asarray(anchor_values[anchor_offset:anchor_offset + count], np.float32))
        decomposition = causal_decompose(probe["sequence"].numpy(), probe["mask"].numpy(), periods=period_library)
        state = state_features(probe["sequence"].numpy(), probe["mask"].numpy())
        neighbors, weights = knn_graph(state[:, :24], min(16, max(1, count - 1)))
        weights = apply_sparse_lead_lag(probe["sequence"].numpy(), probe["mask"].numpy(), neighbors, weights)
        category = category_context(state, probe["categories"])
        probe.update({"anchor": anchor, "decomposition": decomposition, "state": torch.from_numpy(state),
                      "periodic": torch.from_numpy(decomposition["periodic"]),
                      "neighbors": torch.from_numpy(neighbors), "neighbor_weights": torch.from_numpy(weights),
                      "category_context": torch.from_numpy(category)})
        anchor_offset += count
        yield probe
    if anchor_values is not None and anchor_offset != len(anchor_values):
        raise RuntimeError("Anchor rows do not match neural batch rows.")


def _masked_self_supervised_batches(batches, seed: int = 42):
    import torch
    generator = torch.Generator().manual_seed(seed)
    for probe in batches:
        sequence, mask = probe["sequence"].clone(), probe["mask"].clone()
        reconstruction = sequence[:, -1].clone()
        artificial = (torch.rand(mask.shape, generator=generator) < 0.15) & mask.bool()
        sequence[artificial] = 0.0; masked = mask.clone(); masked[artificial] = 0.0
        reversed_view = torch.flip(sequence, dims=[1]); reversed_mask = torch.flip(masked, dims=[1])
        order = torch.arange(sequence.shape[0]) % 2
        second = torch.where(order[:, None, None].bool(), reversed_view, sequence)
        second_mask = torch.where(order[:, None].bool(), reversed_mask, masked)
        quantile = torch.clamp((reconstruction.mean(1).argsort().argsort().float() /
                                max(1, sequence.shape[0]) * 5).long(), max=4)
        yield sequence, masked, reconstruction, order.long(), quantile, second, second_mask


def _self_supervised_pretrain_factory(ctx, cap: int, seed: int = 42):
    """Yield label-free causal windows from the full [0,2918) sequence cache."""
    import torch
    rng = np.random.default_rng(seed)
    for time_index in range(VALID_START):
        available = np.flatnonzero(np.asarray(ctx.sequence_mask[time_index], dtype=bool))
        if available.size < 2:
            continue
        stocks = np.sort(rng.choice(available, size=min(cap, available.size), replace=False)).astype(np.int32)
        history, mask = ctx.causal_history(time_index, stocks, window=240)
        probe = [{"sequence": torch.from_numpy(history), "mask": torch.from_numpy(mask)}]
        yield from _masked_self_supervised_batches(probe, seed + time_index)


def _train_neural_families(config, batch_factory, checkpoint_dir, period_library, foundation_state=None):
    import torch
    from .dual_axis import DualAxisExpert
    from .relational_graph import RelationalGraphExpert
    from .self_supervised import FoundationRepresentationHead, SelfSupervisedEncoder
    from .time_frequency import TimeFrequencyExpert
    from .training import (train_dual_axis, train_foundation, train_foundation_head,
                           train_graph, train_time_frequency)

    sample = next(batch_factory())
    dual = DualAxisExpert(sample["current"].shape[1])
    dual_data = lambda: ((x["current"], x["sequence"], x["mask"], x["anchor"], x["target"], x["groups"]) for x in batch_factory())
    train_dual_axis(config, dual, dual_data, checkpoint_dir / "dual_axis.pt")
    dual.to("cpu"); torch.cuda.empty_cache()
    tf = TimeFrequencyExpert()
    tf_data = lambda: ((x["periodic"], x["mask"], x["anchor"],
                __import__("torch").from_numpy(x["decomposition"]["confidence"]), period_library,
                x["target"], x["groups"]) for x in batch_factory())
    train_time_frequency(config, tf, tf_data, checkpoint_dir / "time_frequency.pt")
    tf.to("cpu"); torch.cuda.empty_cache()
    graph = RelationalGraphExpert(sample["state"].shape[1])
    graph_data = lambda: ((x["state"], x["neighbors"], x["neighbor_weights"], x["category_context"], x["anchor"], x["target"], x["groups"]) for x in batch_factory())
    train_graph(config, graph, graph_data, checkpoint_dir / "relational_graph.pt")
    graph.to("cpu"); torch.cuda.empty_cache()
    foundation = SelfSupervisedEncoder(); foundation_head = FoundationRepresentationHead()
    if foundation_state is not None:
        foundation.load_state_dict(foundation_state)
    train_foundation(config, foundation, lambda: _masked_self_supervised_batches(batch_factory()), checkpoint_dir / "foundation.pt")
    supervised_foundation = lambda: ((x["sequence"], x["mask"], x["target"], x["groups"]) for x in batch_factory())
    train_foundation_head(config, foundation, foundation_head, supervised_foundation, checkpoint_dir / "foundation_head.pt")
    foundation.to("cpu"); foundation_head.to("cpu"); torch.cuda.empty_cache()
    return {"dual_axis": dual, "time_frequency": tf, "relational_graph": graph,
            "foundation_representation": (foundation, foundation_head)}


def _predict_neural(models, batch_factory, config, period_library):
    from .training import predict_dual_axis, predict_foundation, predict_graph, predict_time_frequency
    dual_data = ((x["current"], x["sequence"], x["mask"], x["anchor"]) for x in batch_factory())
    tf_data = ((x["periodic"], x["mask"], x["anchor"],
                __import__("torch").from_numpy(x["decomposition"]["confidence"]), period_library) for x in batch_factory())
    graph_data = ((x["state"], x["neighbors"], x["neighbor_weights"], x["category_context"], x["anchor"]) for x in batch_factory())
    foundation, head = models["foundation_representation"]
    foundation_data = ((x["sequence"], x["mask"]) for x in batch_factory())
    return {
        "dual_axis": predict_dual_axis(models["dual_axis"], dual_data, config.device),
        "time_frequency": predict_time_frequency(models["time_frequency"], tf_data, config.device),
        "relational_graph": predict_graph(models["relational_graph"], graph_data, config.device),
        "foundation_representation": predict_foundation(foundation, head, foundation_data, config.device),
    }


def _expert_matrix(family: dict[str, np.ndarray]) -> np.ndarray:
    names = FAMILIES[:6]
    if any(name not in family for name in names):
        raise RuntimeError("The rank head requires the first six family predictions.")
    return np.stack([family[name] for name in names], axis=1).astype(np.float32)


def _collect_head_inputs(batch_factory, family):
    import torch
    from .state_router import cross_section_state_features
    expert_parts, target_parts, state_parts, group_parts = [], [], [], []
    offset = 0
    matrix = _expert_matrix(family)
    for batch in batch_factory():
        size = batch["stocks"].size
        experts = torch.from_numpy(matrix[offset:offset + size])
        expert_parts.append(experts)
        if batch["target"] is not None: target_parts.append(batch["target"])
        periodic = torch.from_numpy(batch["decomposition"]["confidence"])
        state_parts.append(cross_section_state_features(batch["current"], batch["mask"], experts,
                                                        batch["groups"], periodic))
        group_parts.append(size); offset += size
    if offset != matrix.shape[0]:
        raise RuntimeError("Family predictions and time batches are misaligned.")
    return (torch.cat(expert_parts), None if not target_parts else torch.cat(target_parts),
            torch.cat(state_parts), torch.tensor(group_parts, dtype=torch.long))


def _head_outputs_by_group(head, experts, states, groups):
    import torch
    from .state_router import expand_group_values
    pieces = {"score": [], "top": [], "bottom": [], "confidence": []}
    offset = 0
    head.to("cpu").eval()
    with torch.no_grad():
        for group_index, size_value in enumerate(groups.tolist()):
            size = int(size_value)
            block_state = expand_group_values(states[group_index:group_index + 1], groups[group_index:group_index + 1])
            output = head(experts[offset:offset + size], block_state)
            for name in pieces: pieces[name].append(output[name])
            offset += size
    return {name: torch.cat(values) for name, values in pieces.items()}


def _train_head_and_router(config, family, batch_factory, checkpoint_dir, target_models: bool = False):
    import torch
    from .multi_objective_head import MultiObjectiveRankHead
    from .state_router import StateRouter
    from .training import train_head, train_router

    experts, target, states, groups = _collect_head_inputs(batch_factory, family)
    head = MultiObjectiveRankHead(expert_count=experts.shape[1], state_dim=states.shape[1])
    if target_models:
        head_batches, offset = [], 0
        for group_index, size_value in enumerate(groups.tolist()):
            size = int(size_value)
            state = states[group_index:group_index + 1].repeat(size, 1)
            head_batches.append((experts[offset:offset + size], state, target[offset:offset + size], groups[group_index:group_index + 1]))
            offset += size
        train_head(config, head, head_batches, checkpoint_dir / "multi_objective_head.pt")
    head.to("cpu")
    output = _head_outputs_by_group(head, experts, states, groups)
    family["multi_objective_rank"] = output["score"].numpy().astype(np.float32)
    ranked = torch.from_numpy(np.stack([group_rank(family[name], groups.numpy()) for name in FAMILIES], axis=1))
    desired = []
    offset = 0
    minimum = torch.tensor([MIN_WEIGHTS[name] for name in FAMILIES], dtype=torch.float32)
    base = torch.tensor([BASE_WEIGHTS[name] for name in FAMILIES], dtype=torch.float32)
    for size in groups.tolist():
        block_y, block_p = target[offset:offset + size], ranked[offset:offset + size]
        quality = []
        for col in range(block_p.shape[1]):
            left = block_p[:, col] - block_p[:, col].mean(); right = block_y - block_y.mean()
            quality.append(torch.clamp((left * right).mean() / (left.std(unbiased=False).clamp_min(1e-6) * right.std(unbiased=False).clamp_min(1e-6)), -1, 1))
        score = torch.softmax(torch.stack(quality), dim=0)
        desired.append(minimum + (1.0 - minimum.sum()) * score)
        offset += size
    router = StateRouter(states.shape[1], len(FAMILIES), base, minimum)
    if target_models:
        train_router(config, router, states, torch.stack(desired), checkpoint_dir / "state_router.pt")
    router.to("cpu")
    return head, router


def _predict_head_and_router(head, router, family, batch_factory):
    import torch
    experts, _, states, groups = _collect_head_inputs(batch_factory, family)
    output = _head_outputs_by_group(head, experts, states, groups)
    router.to("cpu").eval()
    with torch.no_grad():
        confidence = torch.stack([part.mean() for part in torch.split(output["confidence"], groups.tolist())])
        weights = router(states, confidence)
    family["multi_objective_rank"] = output["score"].numpy().astype(np.float32)
    return weights.numpy().astype(np.float32), groups.numpy()


def _validate_complete_family(family: dict[str, np.ndarray], rows: int) -> None:
    if set(family) != set(FAMILIES):
        raise RuntimeError(f"Missing family outputs: {sorted(set(FAMILIES) - set(family))}")
    for name, values in family.items():
        values = np.asarray(values)
        if values.shape != (rows,) or not np.isfinite(values).all() or float(np.std(values)) <= 1e-8:
            raise RuntimeError(f"Family {name} failed its prediction contract.")
    if all(np.array_equal(family[FAMILIES[0]], family[name]) for name in FAMILIES[1:]):
        raise RuntimeError("All families are identical; refusing to create a fake unified result.")


def _anchor_to_capped(ctx, split, start, stop, cap, full_values):
    """Select capped row values from a full-split anchor vector."""
    rows, groups = ctx.row_slice(split, start, stop)
    selected = np.concatenate(_capped_positions(groups, cap))
    values = np.asarray(full_values, np.float32)
    if values.size == rows.stop - rows.start:
        return values[selected]
    if split == "test" and values.size == ctx.common["test"]["time"].size:
        return values[selected]
    raise RuntimeError("Anchor cannot be aligned to capped rows.")


def _period_sample(batch_factory, limit: int = 32):
    histories, masks = [], []
    for index, batch in enumerate(batch_factory()):
        histories.append(batch["sequence"].numpy()); masks.append(batch["mask"].numpy())
        if index + 1 >= limit: break
    if not histories:
        raise RuntimeError("Cannot fit period library from an empty stream.")
    return np.concatenate(histories), np.concatenate(masks)


def _build_supervised_family(config, ctx, train_split, train_start, train_stop, pred_split, pred_start, pred_stop,
                             train_X, train_y, train_rel, train_groups, pred_X, pred_groups,
                             train_anchor, pred_anchor, checkpoint_dir, period_library, foundation_state=None):
    tabular_models, _ = _tabular_train_or_load(config, train_X, train_y, train_rel, train_groups, checkpoint_dir / "tabular")
    tabular_pred, _ = predict_tabular_family(tabular_models, pred_X, pred_groups)
    train_factory = lambda: _torch_batches(ctx, train_split, train_start, train_stop, config.stock_cap, train_anchor, period_library)
    models = _train_neural_families(config, train_factory, checkpoint_dir, period_library, foundation_state)
    pred_factory = lambda: _torch_batches(ctx, pred_split, pred_start, pred_stop, config.stock_cap, pred_anchor, period_library)
    return {"exp015_anchor": pred_anchor, "tabular": tabular_pred,
            **_predict_neural(models, pred_factory, config, period_library)}


def _official_validation(config, ctx, cache, checkpoints, head, router, foundation_state):
    """Train on [486,2918), predict official Valid [2918,3161)."""
    from .time_frequency import fit_period_library
    train_X, train_y, train_rel, train_groups, _ = _tabular_arrays(ctx, "train", 486, VALID_START, config.stock_cap)
    valid_X, valid_y, _, valid_groups, _ = _tabular_arrays(ctx, "valid", VALID_START, VALID_STOP, config.stock_cap)
    anchor_model, anchor_reused = _anchor_train_or_load(config, train_X, train_rel, train_groups,
                                                        checkpoints / "official_valid" / "anchor.txt")
    train_anchor = predict_exp015_anchor(anchor_model, train_X); valid_anchor = predict_exp015_anchor(anchor_model, valid_X)
    raw_factory = lambda: _torch_batches(ctx, "train", 486, VALID_START, config.stock_cap, train_anchor)
    history, mask = _period_sample(raw_factory); periods = fit_period_library(history, mask)
    family = _build_supervised_family(config, ctx, "train", 486, VALID_START, "valid", VALID_START, VALID_STOP,
                                      train_X, train_y, train_rel, train_groups, valid_X, valid_groups,
                                      train_anchor, valid_anchor, checkpoints / "official_valid", periods, foundation_state)
    valid_factory = lambda: _torch_batches(ctx, "valid", VALID_START, VALID_STOP, config.stock_cap, valid_anchor, periods)
    weights, routed_groups = _predict_head_and_router(head, router, family, valid_factory)
    _validate_complete_family(family, valid_y.size)
    integrated = dynamic_blend_family_predictions(family, routed_groups, weights)
    scores, offset = [], 0
    for size_value in valid_groups:
        size = int(size_value)
        scores.append(rank_ic(integrated[offset:offset + size], valid_y[offset:offset + size]))
        offset += size
    atomic_npy(cache / "oof_predictions" / "official_valid_prediction.npy", integrated)
    atomic_npy(cache / "oof_predictions" / "official_valid_dynamic_weights.npy", weights)
    atomic_npy(cache / "oof_predictions" / "official_valid_first_six.npy",
               np.stack([family[name] for name in FAMILIES[:6]], axis=1))
    atomic_json(cache / "oof_predictions" / "official_valid.metadata.json",
                {"train": [486, VALID_START], "predict": [VALID_START, VALID_STOP],
                 "rows": int(valid_y.size), "labels_used_for_training": False, "anchor_reused": anchor_reused,
                 "mean_rank_ic": float(np.nanmean(scores))})
    return {"rows": int(valid_y.size), "groups": int(valid_groups.size),
            "mean_rank_ic": float(np.nanmean(scores))}


def execute_full(config: RunConfig) -> dict[str, object]:
    require_training(config, "exp016 full stage")
    from .time_frequency import fit_period_library

    output, cache = RESULT_DIR / "full", CACHE_DIR
    checkpoints = cache / "checkpoints"
    for path in (output, cache / "feature_views", cache / "oof_predictions", cache / "graph", cache / "spectral", checkpoints):
        path.mkdir(parents=True, exist_ok=True)
    ctx = DataContext(load_sequence=True); _write_manifest(ctx, config, output)
    # Local self-supervision is fitted once on [0,2918) without reading labels.
    from .self_supervised import SelfSupervisedEncoder
    from .training import restore_torch_checkpoint, train_foundation
    pretrained_foundation = SelfSupervisedEncoder()
    pretrain_path = checkpoints / "foundation_pretrain_0_2918.pt"
    if pretrain_path.exists():
        pretrain_status = restore_torch_checkpoint(pretrained_foundation, pretrain_path)
    else:
        history = train_foundation(config, pretrained_foundation,
                                   lambda: _self_supervised_pretrain_factory(ctx, config.stock_cap), pretrain_path)
        pretrain_status = {"path": str(pretrain_path), "steps": history.steps, "reused": False}
    foundation_state = {name: value.detach().cpu().clone() for name, value in pretrained_foundation.state_dict().items()}
    oof_families = {name: [] for name in FAMILIES[:6]}
    oof_batch_specs = []
    for fold_name, train_start, train_stop, pred_start, pred_stop in OOF_FOLDS:
        fold_dir = checkpoints / fold_name; fold_dir.mkdir(parents=True, exist_ok=True)
        train_X, train_y, train_rel, train_groups, _ = _tabular_arrays(ctx, "train", train_start, train_stop, config.stock_cap)
        pred_X, pred_y, _, pred_groups, _ = _tabular_arrays(ctx, "train", pred_start, pred_stop, config.stock_cap)
        anchor_model, anchor_reused = _anchor_train_or_load(config, train_X, train_rel, train_groups, fold_dir / "anchor.txt")
        train_anchor = predict_exp015_anchor(anchor_model, train_X); pred_anchor = predict_exp015_anchor(anchor_model, pred_X)
        tabular, tabular_reused = _tabular_train_or_load(config, train_X, train_y, train_rel, train_groups, fold_dir / "tabular")
        tabular_pred, _ = predict_tabular_family(tabular, pred_X, pred_groups)
        raw_train_factory = lambda ts=train_start, te=train_stop, av=train_anchor: _torch_batches(ctx, "train", ts, te, config.stock_cap, av)
        period_history, period_mask = _period_sample(raw_train_factory)
        period_library = fit_period_library(period_history, period_mask)
        train_factory = lambda ts=train_start, te=train_stop, av=train_anchor, pl=period_library: _torch_batches(ctx, "train", ts, te, config.stock_cap, av, pl)
        models = _train_neural_families(config, train_factory, fold_dir, period_library, foundation_state)
        pred_factory = lambda ps=pred_start, pe=pred_stop, av=pred_anchor, pl=period_library: _torch_batches(ctx, "train", ps, pe, config.stock_cap, av, pl)
        neural = _predict_neural(models, pred_factory, config, period_library)
        fold_family = {"exp015_anchor": pred_anchor, "tabular": tabular_pred, **neural}
        for name in FAMILIES[:6]: oof_families[name].append(fold_family[name])
        oof_batch_specs.append((pred_start, pred_stop, pred_anchor, period_library))
        atomic_json(cache / "oof_predictions" / f"{fold_name}.metadata.json", {"train": [train_start, train_stop], "predict": [pred_start, pred_stop], "rows": int(pred_y.size), "strict_oof": True,
                    "anchor_reused": anchor_reused, "tabular_reused": tabular_reused})
    stacked = {name: np.concatenate(parts) for name, parts in oof_families.items()}
    def oof_factory():
        for start, stop, anchor, periods in oof_batch_specs:
            yield from _torch_batches(ctx, "train", start, stop, config.stock_cap, anchor, periods)
    head, router = _train_head_and_router(config, stacked, oof_factory, checkpoints, target_models=True)
    atomic_npy(cache / "oof_predictions" / "family_matrix.npy", np.stack([stacked[name] for name in FAMILIES], axis=1))
    official_validation = _official_validation(config, ctx, cache, checkpoints, head, router, foundation_state)

    final_train_X, final_y, final_rel, final_groups = _combined_supervised_arrays(ctx, config.stock_cap)
    # Inference covers every Test row. stock_cap is a supervised-training
    # resource control and must never silently turn official positions into 0.5.
    test_X, _, _, test_groups, _ = _tabular_arrays(ctx, "test", TEST_START, TEST_STOP, STOCK_COUNT)
    anchor_model, _ = _anchor_train_or_load(config, final_train_X, final_rel, final_groups, checkpoints / "final" / "anchor.txt")
    combined_anchor = predict_exp015_anchor(anchor_model, final_train_X)
    train_rows = int(_tabular_arrays(ctx, "train", 486, VALID_START, config.stock_cap)[0].shape[0])
    train_anchor, valid_anchor = combined_anchor[:train_rows], combined_anchor[train_rows:]
    test_anchor = _anchor_to_capped(ctx, "test", TEST_START, TEST_STOP, STOCK_COUNT, load_exp015_test_anchor(ctx))
    tabular, _ = _tabular_train_or_load(config, final_train_X, final_y, final_rel, final_groups, checkpoints / "final" / "tabular")
    tabular_test, _ = predict_tabular_family(tabular, test_X, test_groups)
    raw_final_factory = lambda: _combined_batch_factory(ctx, config.stock_cap, train_anchor, valid_anchor)
    period_history, period_mask = _period_sample(raw_final_factory)
    period_library = fit_period_library(period_history, period_mask)
    atomic_npy(cache / "spectral" / "final_period_library.npy", period_library.astype(np.int64))
    final_factory = lambda: _combined_batch_factory(ctx, config.stock_cap, train_anchor, valid_anchor, period_library)
    models = _train_neural_families(config, final_factory, checkpoints / "final", period_library, foundation_state)
    test_factory = lambda: _torch_batches(ctx, "test", TEST_START, TEST_STOP, STOCK_COUNT, test_anchor, period_library)
    family = {"exp015_anchor": test_anchor, "tabular": tabular_test, **_predict_neural(models, test_factory, config, period_library)}
    weights, routed_groups = _predict_head_and_router(head, router, family, test_factory)
    _validate_complete_family(family, test_anchor.size)
    final_values = dynamic_blend_family_predictions(family, routed_groups, weights)
    full_test = np.full(ctx.common["test"]["time"].size, 0.5, dtype=np.float32)
    _, complete_groups = ctx.row_slice("test", TEST_START, TEST_STOP)
    full_test[:] = final_values
    grid = vector_to_grid(full_test, ctx.common["test"]["time"], ctx.common["test"]["stock"], TEST_START, TEST_STOP - TEST_START)
    contract = validate_prediction(grid, ctx.test_evaluation_mask())
    # Artifact gate: prediction is the final write, only after all family and contract checks.
    for name, values in family.items(): atomic_npy(output / f"family_{name}.npy", values)
    atomic_npy(output / "dynamic_weights.npy", weights)
    atomic_npy(output / "submitted_prediction.npy", grid)
    atomic_npy(output / "prediction.npy", grid)
    prediction_hash = sha256(output / "prediction.npy")
    immutable = output / f"submission_{prediction_hash[:16]}.npy"
    atomic_npy(immutable, grid)
    atomic_json(output / "online_feedback_template.json", {"prediction_sha256": prediction_hash,
                "submitted_at": None, "online_score": None, "rank": None, "notes": ""})
    atomic_json(output / "metadata.json", {"status": "full_completed", "pretraining": pretrain_status,
                "oof_folds_completed": [x[0] for x in OOF_FOLDS],
                "official_validation": official_validation, "contract": contract, "prediction_sha256": prediction_hash,
                "immutable_submission": str(immutable), "formal_submission_overwritten": False})
    return {"status": "FULL_COMPLETED", "official_validation": official_validation,
            "contract": contract, "formal_submission_overwritten": False}
