from __future__ import annotations

import ast
import json
import tempfile
from pathlib import Path

import numpy as np

from ..config import BASE_WEIGHTS, CACHE_DIR, EXPERIMENT_ID, FAMILIES, FINAL_SUBMISSION, LABEL_GAIN, MIN_WEIGHTS, RELEVANCE_LEVELS, RESULT_DIR, RunConfig, TEST_START
from .artifacts import atomic_json, atomic_npy, sha256
from .data_context import DataContext
from .feature_views import robust_rank_features, state_features as numpy_state_features
from .oof_anchor import anchor_oof_contract, load_exp015_test_anchor
from .prediction_contract import validate_prediction, vector_to_grid
from .ranking import blend_family_predictions, group_rank


def protected_sha() -> str:
    if not FINAL_SUBMISSION.exists():
        raise RuntimeError("正式提交文件不存在。")
    return sha256(FINAL_SUBMISSION)


def assert_protected_unchanged(expected: str) -> None:
    actual = protected_sha()
    if actual != expected:
        raise AssertionError("exp016 不得修改 final_submission。")


def source_root() -> Path:
    return Path(__file__).resolve().parents[1]


def static_check(config: RunConfig) -> dict[str, object]:
    root = source_root()
    compiled = []
    unsafe = []
    for path in sorted((root / "src").glob("*.py")) + [root / "config.py", root / "run_exp016.py"]:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        compile(tree, str(path), "exec")
        compiled.append(str(path.relative_to(root)))
        if path.name not in {"tabular_experts.py", "training.py", "full_pipeline.py", "oof_anchor.py"}:
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and node.attr in {"step", "fit", "train"}:
                    unsafe.append(f"{path.name}:{node.lineno}:{node.attr}")
    protected = {"training.py", "tabular_experts.py", "oof_anchor.py", "full_pipeline.py"}
    missing_guards = []
    for name in protected:
        text = (root / "src" / name).read_text(encoding="utf-8")
        if "require_training(" not in text:
            missing_guards.append(name)
    import os
    previous_mode = os.environ.pop("DSCR_EXP016_MODE", None)
    previous_allow = os.environ.pop("DSCR_EXP016_ALLOW_TRAINING", None)
    try:
        default_mode = RunConfig.from_environment().mode
    finally:
        if previous_mode is not None:
            os.environ["DSCR_EXP016_MODE"] = previous_mode
        if previous_allow is not None:
            os.environ["DSCR_EXP016_ALLOW_TRAINING"] = previous_allow
    notebook = json.loads((root / "experiment.ipynb").read_text(encoding="utf-8"))
    notebook_cells = 0
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            compile("".join(cell.get("source", [])), str(root / "experiment.ipynb"), "exec")
            notebook_cells += 1
    report = {
        "status": "STATIC_CHECK_PASSED", "compiled": compiled, "default_mode": default_mode,
        "notebook_code_cells_compiled": notebook_cells,
        "training_guard": "DSCR_EXP016_MODE=full + DSCR_EXP016_ALLOW_TRAINING=YES",
        "unexpected_training_calls": unsafe,
        "protected_training_modules": ["src/training.py", "src/tabular_experts.py", "src/oof_anchor.py", "src/full_pipeline.py"],
        "protected_modules_missing_guard": missing_guards,
        "families": list(FAMILIES), "full_executed": False,
    }
    if report["default_mode"] != "smoke" or unsafe or missing_guards:
        raise AssertionError(report)
    return report


def smoke(config: RunConfig) -> dict[str, object]:
    import torch
    from .dual_axis import DualAxisExpert
    from .multi_objective_head import MultiObjectiveRankHead
    from .ranking_objectives import multi_objective_loss
    from .relational_graph import RelationalGraphExpert, category_context, knn_graph
    from .self_supervised import SelfSupervisedEncoder, self_supervised_loss
    from .state_router import StateRouter, expand_group_values, state_features
    from .time_frequency import TimeFrequencyExpert, causal_decompose
    before = protected_sha()
    torch.manual_seed(42); rng = np.random.default_rng(42)
    groups = np.asarray([12, 11], np.int32); n, window, feature_count = int(groups.sum()), 240, 40
    current = rng.normal(size=(n, 64)).astype(np.float32)
    sequence = rng.normal(size=(n, window, feature_count)).astype(np.float32)
    mask = (rng.random((n, window)) > 0.20).astype(np.float32); mask[:, -1] = 1.0
    anchor = rng.normal(size=n).astype(np.float32); target = rng.random(n).astype(np.float32)
    features = robust_rank_features(current, numeric_count=40)
    history_state = numpy_state_features(sequence, mask)
    idx, weight = knn_graph(history_state[:, :24], 8)
    categories = rng.integers(0, 5, size=(n, 9)); context = category_context(history_state, categories)
    tf = causal_decompose(sequence, mask)
    ct, st, mt, at = map(torch.from_numpy, (features, sequence, mask, anchor))
    dual = DualAxisExpert(current_dim=features.shape[1])
    parameters_before = {name: value.detach().clone() for name, value in dual.named_parameters()}
    dual_out = dual(ct, st, mt, at)
    permutation = torch.randperm(n); inverse = torch.argsort(permutation)
    permuted = dual(ct[permutation], st[permutation], mt[permutation], at[permutation])["residual"][inverse]
    prototype_equivariant = bool(torch.allclose(permuted, dual_out["residual"], atol=1e-5, rtol=1e-5))
    selfsup = SelfSupervisedEncoder(); ss = selfsup(st, mt)
    ssloss = self_supervised_loss(ss, st[:, -1], torch.zeros(n, dtype=torch.long), torch.zeros(n, dtype=torch.long))
    graph = RelationalGraphExpert(dim=history_state.shape[1]); graph_out = graph(torch.from_numpy(history_state), torch.from_numpy(idx), torch.from_numpy(weight), torch.from_numpy(context), at)
    tf_model = TimeFrequencyExpert(); tf_out = tf_model(st, mt, at, torch.from_numpy(tf["confidence"]))
    experts = torch.stack([at, dual_out["residual"], tf_out, graph_out, ss["embedding"].mean(1), dual_out["direct"]], dim=1)
    group_tensor = torch.from_numpy(groups)
    router_state = state_features(ct, mt, experts, group_tensor, torch.from_numpy(tf["confidence"]))
    expanded_state = expand_group_values(router_state, group_tensor)
    head = MultiObjectiveRankHead(expert_count=6, state_dim=expanded_state.shape[1]); head_out = head(experts, expanded_state)
    loss = multi_objective_loss(head_out, torch.from_numpy(target), group_tensor, anchor=at)
    (ssloss + loss).backward()
    parameters_unchanged = all(torch.equal(parameters_before[name], value.detach()) for name, value in dual.named_parameters())
    base = torch.tensor([BASE_WEIGHTS[x] for x in FAMILIES]); minimum = torch.tensor([MIN_WEIGHTS[x] for x in FAMILIES])
    router = StateRouter(router_state.shape[1], len(FAMILIES), base, minimum)
    group_confidence = torch.stack([part.mean() for part in torch.split(head_out["confidence"], groups.tolist())])
    routed = router(router_state, group_confidence)
    if not bool(torch.all(routed >= minimum - 1e-6)):
        raise AssertionError("路由器未保留正权重下限。")
    combined = blend_family_predictions({name: rng.normal(size=n).astype(np.float32) for name in FAMILIES}, groups, BASE_WEIGHTS)
    if not np.isfinite(combined).all() or not np.all((combined > 0) & (combined <= 1)):
        raise AssertionError("排名融合不合法。")
    tied = group_rank(np.asarray([1.0, 1.0, 2.0], np.float32), [3])
    if tied[0] != tied[1]:
        raise AssertionError("Average-tie percentile ranking contract failed.")
    eval_mask = np.zeros((442, 5282), dtype=bool); eval_mask[:1, :n] = True
    grid = np.full((442, 5282), 0.5, dtype=np.float32); grid[0, :n] = group_rank(combined[:n], [n])
    contract = validate_prediction(grid, eval_mask)
    probe = config.run_dir / "atomic_probe.npy"; atomic_npy(probe, combined); atomic_ok = np.array_equal(np.load(probe), combined); probe.unlink(missing_ok=True)
    assert_protected_unchanged(before)
    return {"status": "SMOKE_PASSED", "training_performed": False, "optimizer_step_called": False, "tree_fit_called": False,
            "dual_axis_forward_backward": True, "prototype_permutation_equivariant": prototype_equivariant,
            "self_supervised_forward_backward": True, "time_frequency_forward": True,
            "relational_graph_forward": True, "multi_objective_forward_backward": True,
            "parameters_unchanged": parameters_unchanged, "rank_average_ties": True, "router_lower_bounds": True,
            "router_is_cross_sectional": list(routed.shape) == [groups.size, len(FAMILIES)],
            "spectral_confidence_finite": bool(np.isfinite(tf["confidence"]).all()), "atomic_write_verified": atomic_ok,
            "prediction_contract": contract, "final_submission_sha256_before": before, "final_submission_sha256_after": protected_sha()}


def preflight(config: RunConfig) -> dict[str, object]:
    before = protected_sha(); ctx = DataContext(load_sequence=True)
    times = np.asarray(ctx.common["valid"]["time"][:], dtype=np.int32)
    unique = np.unique(times)[:config.real_preflight_times]
    rows, groups = ctx.row_slice("valid", int(unique[0]), int(unique[-1]) + 1)
    last_rows, last_groups = ctx.row_slice("valid", int(unique[-1]), int(unique[-1]) + 1)
    selected_count = min(128, int(last_groups[0]))
    selected = np.asarray(ctx.common["valid"]["stock"][last_rows.start:last_rows.start + selected_count], dtype=np.int32)
    history, mask = ctx.causal_history(int(unique[-1]), selected, window=240)
    current = np.asarray(ctx.tree["valid"][last_rows.start:last_rows.start + selected.size, :64], dtype=np.float32)
    features = robust_rank_features(current, numeric_count=40)
    from .relational_graph import knn_graph
    state = numpy_state_features(history, mask); graph_idx, graph_weight = knn_graph(state[:, :24], min(16, selected.size - 1))
    from .time_frequency import causal_decompose
    spectral = causal_decompose(history, mask)
    from .tabular_experts import _catboost_pool, build_tabular_dry_run, lambdarank_params, validate_relevance
    table = build_tabular_dry_run(np.asarray(ctx.tree["valid"][rows.start:rows.start + min(256, rows.stop - rows.start)], dtype=np.float32), np.asarray(ctx.common["valid"]["relevance"][rows.start:rows.start + min(256, rows.stop - rows.start)]), np.asarray([min(256, rows.stop - rows.start)], np.int32))
    cat_pool = _catboost_pool(np.asarray(ctx.tree["valid"][rows.start:rows.start + min(64, rows.stop - rows.start)], dtype=np.float32), None, None)
    import lightgbm as lgb
    import xgboost as xgb
    tree_sample = np.asarray(ctx.tree["valid"][rows.start:rows.start + min(64, rows.stop - rows.start)], dtype=np.float32)
    label_sample = np.asarray(ctx.common["valid"]["relevance"][rows.start:rows.start + tree_sample.shape[0]], dtype=np.int32)
    lgb_dataset = lgb.Dataset(tree_sample, label=label_sample, group=np.asarray([tree_sample.shape[0]], np.int32), free_raw_data=False)
    lgb_dataset.construct()
    xgb_matrix = xgb.DMatrix(tree_sample, label=label_sample, group=np.asarray([tree_sample.shape[0]], np.int32))
    train_relevance = ctx.common["train"]["relevance"]
    valid_relevance = ctx.common["valid"]["relevance"]
    relevance_contract = {
        "train": validate_relevance(np.asarray(train_relevance, np.int32), np.asarray(ctx.common["train"]["groups"], np.int32)),
        "valid": validate_relevance(np.asarray(valid_relevance, np.int32), np.asarray(ctx.common["valid"]["groups"], np.int32)),
        "label_gain_length": len(lambdarank_params()["label_gain"]),
        "label_gain_matches_levels": lambdarank_params()["label_gain"] == list(LABEL_GAIN),
    }
    if relevance_contract["label_gain_length"] != RELEVANCE_LEVELS or not relevance_contract["label_gain_matches_levels"]:
        raise AssertionError("LambdaRank label_gain does not cover the real relevance mapping.")
    checkpoint_contract = {"available": False, "compatible": None, "steps": None}
    pretrain_path = CACHE_DIR / "checkpoints" / "foundation_pretrain_0_2918.pt"
    if pretrain_path.exists():
        from .self_supervised import SelfSupervisedEncoder
        from .training import restore_torch_checkpoint
        restored = restore_torch_checkpoint(SelfSupervisedEncoder(), pretrain_path)
        checkpoint_contract = {"available": True, "compatible": True, "steps": restored["steps"]}
    from .artifacts import atomic_text
    with tempfile.TemporaryDirectory(prefix="exp016_unicode_probe_", dir=CACHE_DIR) as temporary:
        unicode_probe = Path(temporary) / "重复保存.txt"
        atomic_text(unicode_probe, "first")
        atomic_text(unicode_probe, "second")
        unicode_atomic_write = unicode_probe.read_text(encoding="utf-8") == "second"
    if not unicode_atomic_write:
        raise AssertionError("Unicode-path atomic overwrite contract failed.")
    batch_bytes = config.stock_cap * 240 * 40 * 4
    cuda_budget = None
    try:
        import torch
        if torch.cuda.is_available():
            cuda_budget = int(torch.cuda.get_device_properties(0).total_memory)
    except Exception:
        pass
    anchor = load_exp015_test_anchor(ctx)
    test_mask = ctx.test_evaluation_mask()
    try:
        import psutil
        memory_mb = int(psutil.Process().memory_info().rss / (1024 * 1024))
    except Exception:
        memory_mb = None
    assert "y" not in ctx.common["test"]
    assert_protected_unchanged(before)
    return {"status": "PREFLIGHT_PASSED", "training_performed": False, "optimizer_step_called": False, "tree_fit_called": False,
            "context": ctx.summary(), "real_time_points": unique.tolist(), "history_shape": list(history.shape), "feature_shape": list(features.shape),
            "graph_shape": {"indices": list(graph_idx.shape), "weights": list(graph_weight.shape)}, "spectral_confidence_finite": bool(np.isfinite(spectral["confidence"]).all()),
            "tabular_dry_run": table.__dict__, "catboost_pool_constructed": {"rows": cat_pool.num_row(), "columns": cat_pool.num_col()},
            "tree_containers": {"lightgbm_rows": int(lgb_dataset.num_data()), "xgboost_rows": int(xgb_matrix.num_row())},
            "relevance_contract": relevance_contract, "pretraining_checkpoint": checkpoint_contract,
            "unicode_atomic_overwrite": unicode_atomic_write,
            "resource_budget": {"sequence_batch_bytes": int(batch_bytes), "cuda_total_bytes": cuda_budget,
                                "fits_cuda_sequence_budget": cuda_budget is None or batch_bytes < cuda_budget // 2},
            "aligned_preflight_time": int(unique[-1]), "anchor_test_rows": int(anchor.size), "test_label_loaded": "y" in ctx.common["test"],
            "test_evaluation_count": int(test_mask.sum()), "memory_peak_mb": memory_mb, "anchor_oof_contract": anchor_oof_contract(ctx),
            "final_submission_sha256_before": before, "final_submission_sha256_after": protected_sha()}


def full(config: RunConfig) -> None:
    import time
    from .full_pipeline import execute_full
    try:
        return execute_full(config)
    except Exception as error:
        atomic_json(config.run_dir / "failure.json", {
            "status": "FULL_FAILED", "error_type": type(error).__name__, "message": str(error),
            "failed_at": time.strftime("%Y-%m-%d %H:%M:%S"), "prediction_written": (config.run_dir / "prediction.npy").exists(),
        })
        raise


def run(config: RunConfig) -> dict[str, object]:
    if config.mode == "static": result = static_check(config)
    elif config.mode == "smoke": result = smoke(config)
    elif config.mode == "preflight": result = preflight(config)
    else:
        result = full(config)
    config.run_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(config.run_dir / ("static_check_report.json" if config.mode == "static" else f"{config.mode}_report.json"), result)
    return result
