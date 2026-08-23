from __future__ import annotations

"""T0.1b：全量 Valid inference 重算（零训练，只读 checkpoints）。

重型截面特征（图 KNN / lead-lag / 时频分解）按组流式计算且每组只算一次，
四家神经专家 + 元头状态在同一趟循环内消费。前六家族预测与状态会缓存到
oof_predictions，重复执行时直接复用缓存，跳过重型推理。

不调用 require_training，不写任何模型文件，只写结果报告与预测矩阵。
"""

import json
import time

import numpy as np
import torch

from ..config import (BASE_WEIGHTS, CACHE_DIR, FAMILIES, MIN_WEIGHTS, RESULT_DIR,
                      RunConfig, STOCK_COUNT, VALID_START, VALID_STOP)
from .artifacts import atomic_json, atomic_npy
from .data_context import DataContext
from .dual_axis import DualAxisExpert
from .full_pipeline import _head_outputs_by_group, _period_sample, _tabular_arrays, _torch_batches, _validate_complete_family
from .multi_objective_head import MultiObjectiveRankHead
from .oof_anchor import load_exp015_anchor, predict_exp015_anchor
from .ranking import dynamic_blend_family_predictions, rank_ic
from .relational_graph import RelationalGraphExpert
from .self_supervised import FoundationRepresentationHead, SelfSupervisedEncoder
from .state_router import StateRouter, cross_section_state_features
from .tabular_experts import load_tabular_family, predict_tabular_family
from .time_frequency import TimeFrequencyExpert, fit_period_library
from .training import restore_torch_checkpoint


def _build_config() -> RunConfig:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return RunConfig(mode="smoke", training_allowed=False, stage="all", device=device, stock_cap=1024)


def _infer_first_six(config: RunConfig, ctx: DataContext, valid_X, valid_anchor, valid_groups, periods):
    """流式计算前六家族（anchor+tabular 已在外部算好前两家）。返回 first_six、states、group_sizes。"""
    device = config.device
    checkpoints = CACHE_DIR / "checkpoints"

    tabular_models = load_tabular_family(checkpoints / "official_valid" / "tabular")
    tabular_pred, _ = predict_tabular_family(tabular_models, valid_X, valid_groups)

    sample = next(_torch_batches(ctx, "valid", VALID_START, VALID_STOP, STOCK_COUNT, valid_anchor, periods))
    dual = DualAxisExpert(sample["current"].shape[1])
    restore_torch_checkpoint(dual, checkpoints / "official_valid" / "dual_axis.pt")
    tf = TimeFrequencyExpert()
    restore_torch_checkpoint(tf, checkpoints / "official_valid" / "time_frequency.pt")
    graph = RelationalGraphExpert(sample["state"].shape[1])
    restore_torch_checkpoint(graph, checkpoints / "official_valid" / "relational_graph.pt")
    foundation = SelfSupervisedEncoder()
    foundation_head = FoundationRepresentationHead()
    restore_torch_checkpoint(foundation, checkpoints / "official_valid" / "foundation.pt")
    restore_torch_checkpoint(foundation_head, checkpoints / "official_valid" / "foundation_head.pt")
    dual.to(device).eval(); tf.to(device).eval(); graph.to(device).eval(); foundation.to(device).eval(); foundation_head.to(device).eval()

    dual_parts, tf_parts, graph_parts, found_parts = [], [], [], []
    state_parts, group_sizes = [], []
    offset = 0
    t0 = time.time()
    n_groups = int(valid_groups.size)
    for i, p in enumerate(_torch_batches(ctx, "valid", VALID_START, VALID_STOP, STOCK_COUNT, valid_anchor, periods)):
        size = int(p["stocks"].size)
        with torch.no_grad():
            d = dual(p["current"].to(device), p["sequence"].to(device), p["mask"].to(device), p["anchor"].to(device))["residual"].cpu().numpy()
            t = tf(p["periodic"].to(device), p["mask"].to(device), p["anchor"].to(device), torch.from_numpy(p["decomposition"]["confidence"]).to(device), periods).cpu().numpy()
            g = graph(p["state"].to(device), p["neighbors"].to(device), p["neighbor_weights"].to(device), p["category_context"].to(device), p["anchor"].to(device)).cpu().numpy()
            emb = foundation(p["sequence"].to(device), p["mask"].to(device))["embedding"]
            f = foundation_head(emb, p["mask"].to(device).mean(1)).cpu().numpy()
        dual_parts.append(d); tf_parts.append(t); graph_parts.append(g); found_parts.append(f)

        group_experts = np.stack([valid_anchor[offset:offset + size], tabular_pred[offset:offset + size], d, t, g, f], axis=1).astype(np.float32)
        state_parts.append(cross_section_state_features(p["current"], p["mask"], torch.from_numpy(group_experts), p["groups"],
                                                        torch.from_numpy(p["decomposition"]["confidence"])))
        group_sizes.append(size)
        offset += size
        if (i + 1) % 30 == 0:
            print(f"[T0.1b] processed {i + 1}/{n_groups} groups ({time.time() - t0:.1f}s)", flush=True)
    print(f"[T0.1b] {n_groups} groups in {time.time() - t0:.1f}s", flush=True)

    family = {"exp015_anchor": valid_anchor, "tabular": tabular_pred,
              "dual_axis": np.concatenate(dual_parts).astype(np.float32),
              "time_frequency": np.concatenate(tf_parts).astype(np.float32),
              "relational_graph": np.concatenate(graph_parts).astype(np.float32),
              "foundation_representation": np.concatenate(found_parts).astype(np.float32)}
    first_six = np.stack([family[name] for name in FAMILIES[:6]], axis=1).astype(np.float32)
    states = torch.cat(state_parts)
    return first_six, states, group_sizes


def recompute_full_valid() -> dict:
    config = _build_config()
    device = config.device
    checkpoints = CACHE_DIR / "checkpoints"
    ctx = DataContext(load_sequence=True)
    print(f"[T0.1b] device={device}", flush=True)

    valid_X, valid_y, _, valid_groups, _ = _tabular_arrays(ctx, "valid", VALID_START, VALID_STOP, STOCK_COUNT)
    print(f"[T0.1b] valid rows={valid_X.shape[0]}, groups={valid_groups.size}", flush=True)

    cache_dir = CACHE_DIR / "oof_predictions"
    cache_dir.mkdir(parents=True, exist_ok=True)
    first_six_path = cache_dir / "full_valid_first_six.npy"
    states_path = cache_dir / "full_valid_states.npy"
    groups_path = cache_dir / "full_valid_group_sizes.npy"

    if first_six_path.exists() and states_path.exists() and groups_path.exists():
        first_six = np.load(first_six_path)
        states = torch.from_numpy(np.load(states_path))
        group_sizes = np.load(groups_path).tolist()
        print("[T0.1b] loaded cached first-six family predictions", flush=True)
    else:
        anchor_model = load_exp015_anchor(checkpoints / "official_valid" / "anchor.txt")
        valid_anchor = predict_exp015_anchor(anchor_model, valid_X)
        train_X, _, train_rel, train_groups, _ = _tabular_arrays(ctx, "train", 486, VALID_START, config.stock_cap)
        train_anchor = predict_exp015_anchor(anchor_model, train_X)
        raw_factory = lambda: _torch_batches(ctx, "train", 486, VALID_START, config.stock_cap, train_anchor)
        history, mask = _period_sample(raw_factory)
        periods = fit_period_library(history, mask)
        print(f"[T0.1b] period library: {periods.tolist()}", flush=True)
        first_six, states, group_sizes = _infer_first_six(config, ctx, valid_X, valid_anchor, valid_groups, periods)
        np.save(first_six_path, first_six)
        np.save(states_path, states.numpy())
        np.save(groups_path, np.asarray(group_sizes, dtype=np.int64))
        print("[T0.1b] cached first-six family predictions", flush=True)

    family = {name: first_six[:, i] for i, name in enumerate(FAMILIES[:6])}

    head = MultiObjectiveRankHead(expert_count=6, state_dim=8)
    restore_torch_checkpoint(head, checkpoints / "multi_objective_head.pt")
    base = torch.tensor([BASE_WEIGHTS[n] for n in FAMILIES], dtype=torch.float32)
    minimum = torch.tensor([MIN_WEIGHTS[n] for n in FAMILIES], dtype=torch.float32)
    router = StateRouter(8, len(FAMILIES), base, minimum)
    restore_torch_checkpoint(router, checkpoints / "state_router.pt")

    experts = torch.from_numpy(first_six)
    groups_t = torch.tensor(group_sizes, dtype=torch.long)

    output = _head_outputs_by_group(head, experts, states, groups_t)
    family["multi_objective_rank"] = output["score"].numpy().astype(np.float32)
    _validate_complete_family(family, int(valid_y.size))

    router.to("cpu").eval()
    with torch.no_grad():
        confidence = torch.stack([part.mean() for part in torch.split(output["confidence"], group_sizes)])
        weights = router(states, confidence)
    weights = weights.numpy().astype(np.float32)
    routed_groups = groups_t.numpy()

    integrated = dynamic_blend_family_predictions(family, routed_groups, weights)

    valid_time = np.asarray(ctx.common["valid"]["time"], dtype=np.int32)
    scores, per_time, offset = [], [], 0
    for size_value in valid_groups:
        size = int(size_value)
        ic = rank_ic(integrated[offset:offset + size], valid_y[offset:offset + size])
        scores.append(ic)
        per_time.append((int(valid_time[offset]), ic))
        offset += size
    scores = np.asarray(scores, dtype=np.float64)
    finite = scores[np.isfinite(scores)]

    report = {
        "experiment": "exp_016_unified_expert_fusion",
        "task": "T0.1b full valid inference",
        "rows": int(valid_y.size),
        "groups": int(valid_groups.size),
        "cap": STOCK_COUNT,
        "mean_rank_ic": float(np.mean(finite)),
        "median_rank_ic": float(np.median(finite)),
        "std_rank_ic": float(np.std(finite)),
        "positive_ratio": float((finite > 0).mean()),
        "worst": float(finite.min()),
        "best": float(finite.max()),
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    out = RESULT_DIR / "full"
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / "official_valid_full_report.json", report)
    atomic_npy(out / "full_valid_family_predictions.npy",
               np.stack([family[name] for name in FAMILIES], axis=1).astype(np.float32))
    atomic_npy(out / "full_valid_dynamic_weights.npy", weights.astype(np.float32))
    with open(out / "per_time_ic_full.csv", "w", encoding="utf-8") as handle:
        handle.write("time,rank_ic\n")
        for t, ic in per_time:
            handle.write(f"{t},{ic:.10f}\n")

    print("[T0.1b] DONE", flush=True)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


if __name__ == "__main__":
    recompute_full_valid()
