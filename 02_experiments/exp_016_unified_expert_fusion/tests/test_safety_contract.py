"""Dependency-free regression checks; runnable with the project interpreter."""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve()
    for parent in (root, *root.parents):
        if (parent / "data.z").exists():
            sys.path.insert(0, str(parent))
            break
    config = importlib.import_module("02_experiments.exp_016_unified_expert_fusion.config")
    pipeline = importlib.import_module("02_experiments.exp_016_unified_expert_fusion.src.pipeline")
    full_pipeline = importlib.import_module("02_experiments.exp_016_unified_expert_fusion.src.full_pipeline")
    data_context = importlib.import_module("02_experiments.exp_016_unified_expert_fusion.src.data_context")
    tabular = importlib.import_module("02_experiments.exp_016_unified_expert_fusion.src.tabular_experts")
    os.environ.pop("DSCR_EXP016_ALLOW_TRAINING", None)
    for mode in ("static", "smoke", "preflight"):
        os.environ["DSCR_EXP016_MODE"] = mode
        run_config = config.RunConfig.from_environment()
        try:
            config.require_training(run_config, "test guard")
        except PermissionError:
            pass
        else:
            raise AssertionError(f"{mode} 不应允许真实训练")
    assert config.RunConfig.from_environment().mode == "preflight"
    assert not (config.RESULT_DIR / "preflight" / "prediction.npy").exists()
    assert not (config.RESULT_DIR / "smoke" / "prediction.npy").exists()
    before = pipeline.protected_sha()
    assert pipeline.protected_sha() == before
    ctx = data_context.DataContext(load_sequence=False)
    X, target, relevance, groups, indices = full_pipeline._tabular_arrays(ctx, "train", 1459, 1461, 8)
    assert int(ctx.common["train"]["time"][indices[0]]) == 1459
    assert X.shape[0] == target.size == relevance.size == int(groups.sum())
    relevance_info = tabular.validate_relevance(relevance, groups)
    assert relevance_info["maximum"] < relevance_info["levels"] == 64
    assert tabular.lambdarank_params()["label_gain"] == list(range(64))
    sequence_ctx = data_context.DataContext(load_sequence=True)
    batch = next(full_pipeline.time_slice_batches(sequence_ctx, "train", 1459, 1460, 8))
    assert batch["time"] == 1459 and batch["current"].shape == (8, 120)
    from tempfile import TemporaryDirectory
    artifacts = importlib.import_module("02_experiments.exp_016_unified_expert_fusion.src.artifacts")
    with TemporaryDirectory(prefix="exp016_atomic_", dir=config.CACHE_DIR) as temporary:
        unicode_path = Path(temporary) / "模型.txt"
        artifacts.atomic_text(unicode_path, "one")
        artifacts.atomic_text(unicode_path, "two")
        assert unicode_path.read_text(encoding="utf-8") == "two"
    os.environ["DSCR_EXP016_MODE"] = "full"
    denied = config.RunConfig.from_environment()
    try:
        full_pipeline.execute_full(denied)
    except PermissionError:
        pass
    else:
        raise AssertionError("full without explicit authorization must fail before data/model work")
    print("SAFETY_CONTRACT_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
