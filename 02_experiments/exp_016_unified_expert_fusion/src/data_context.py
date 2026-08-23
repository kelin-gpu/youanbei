from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..config import DATASET_DIR, SEQUENCE_FEATURES, STOCK_COUNT, TIME_COUNT, TREE_FEATURES
from .artifacts import sha256


class DataContext:
    """只读 processed_data_v1 访问器；Test 永远不装载监督标签。"""

    def __init__(self, dataset_dir: Path = DATASET_DIR, load_sequence: bool = True):
        self.dataset_dir = Path(dataset_dir)
        ready_path, manifest_path = self.dataset_dir / "READY", self.dataset_dir / "manifest.json"
        if not ready_path.exists() or not manifest_path.exists():
            raise RuntimeError("processed_data_v1 缺少 READY 或 manifest.json。")
        self.ready = json.loads(ready_path.read_text(encoding="utf-8"))
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.manifest_sha256 = sha256(manifest_path)
        if self.ready.get("manifest_sha256") != self.manifest_sha256:
            raise RuntimeError("READY 与 manifest SHA-256 不一致。")
        if self.manifest.get("status") != "ready":
            raise RuntimeError("共享缓存不是 ready 状态。")
        if self.manifest["dimensions"] != {"time": TIME_COUNT, "stock": STOCK_COUNT, "raw_numeric": 99, "raw_category": 9}:
            raise RuntimeError("共享缓存维度契约不匹配。")
        self.common: dict[str, dict[str, np.ndarray]] = {}
        self.tree: dict[str, np.ndarray] = {}
        common_dir, tree_dir = self.dataset_dir / "common", self.dataset_dir / "tree"
        for split in ("train", "valid", "test"):
            entry = {
                "time": np.load(common_dir / f"{split}_time.npy", mmap_mode="r"),
                "stock": np.load(common_dir / f"{split}_stock.npy", mmap_mode="r"),
                "groups": np.load(common_dir / f"{split}_group_sizes.npy", mmap_mode="r"),
            }
            if split != "test":
                entry["y"] = np.load(common_dir / f"{split}_y.npy", mmap_mode="r")
                entry["relevance"] = np.load(common_dir / f"{split}_relevance.npy", mmap_mode="r")
            self.common[split] = entry
            self.tree[split] = np.load(tree_dir / f"{split}_X.npy", mmap_mode="r")
            if self.tree[split].shape != (int(self.manifest["expected_rows"][split]), TREE_FEATURES):
                raise RuntimeError(f"{split} tree 缓存形状不匹配。")
            if int(entry["groups"].sum()) != entry["time"].size:
                raise RuntimeError(f"{split} 分组契约不匹配。")
        self.sequence = None
        self.sequence_mask = None
        if load_sequence:
            self.sequence = np.load(self.dataset_dir / "sequence" / "X.npy", mmap_mode="r")
            self.sequence_mask = np.load(self.dataset_dir / "sequence" / "mask_x.npy", mmap_mode="r")
            if self.sequence.shape != (TIME_COUNT, STOCK_COUNT, SEQUENCE_FEATURES):
                raise RuntimeError("sequence 缓存形状不匹配。")
            if self.sequence_mask.shape != (TIME_COUNT, STOCK_COUNT):
                raise RuntimeError("sequence mask 缓存形状不匹配。")

    def row_slice(self, split: str, start: int, stop: int) -> tuple[slice, np.ndarray]:
        if split not in self.common or start >= stop:
            raise ValueError("split 或时间区间不合法。")
        times = self.common[split]["time"]
        left, right = int(np.searchsorted(times, start, side="left")), int(np.searchsorted(times, stop, side="left"))
        split_start = int(times[0])
        groups = np.asarray(self.common[split]["groups"][start - split_start:stop - split_start], dtype=np.int32)
        if int(groups.sum()) != right - left:
            raise RuntimeError("时间切片与 groups 不一致。")
        return slice(left, right), groups

    def causal_history(self, time_index: int, stocks: np.ndarray, window: int = 240) -> tuple[np.ndarray, np.ndarray]:
        if self.sequence is None or self.sequence_mask is None:
            raise RuntimeError("当前 DataContext 未加载 sequence。")
        if not 0 <= int(time_index) < TIME_COUNT:
            raise ValueError("time_index 越界。")
        stocks = np.asarray(stocks, dtype=np.int32)
        begin = max(0, int(time_index) - int(window) + 1)
        values = np.zeros((stocks.size, window, SEQUENCE_FEATURES), dtype=np.float32)
        masks = np.zeros((stocks.size, window), dtype=np.float32)
        actual = np.asarray(self.sequence[begin:time_index + 1, stocks, :], dtype=np.float32).transpose(1, 0, 2)
        observed = np.asarray(self.sequence_mask[begin:time_index + 1, stocks], dtype=np.float32).T
        values[:, -actual.shape[1]:] = actual
        masks[:, -observed.shape[1]:] = observed
        return values, masks

    def test_evaluation_mask(self) -> np.ndarray:
        from .prediction_contract import evaluation_mask
        return evaluation_mask(self.common["test"]["time"], self.common["test"]["stock"])

    def summary(self) -> dict[str, object]:
        return {
            "manifest_sha256": self.manifest_sha256,
            "splits": {
                split: {"rows": int(v["time"].size), "groups": int(v["groups"].size), "has_labels": "y" in v}
                for split, v in self.common.items()
            },
            "test_has_labels": "y" in self.common["test"],
        }
