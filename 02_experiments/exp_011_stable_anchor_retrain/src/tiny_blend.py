"""补跑：小权重 TCN/线性 融合（官方 Valid，探索性）。"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(r"d:\google_dl\book\youanbei")
sys.path.insert(0, str(PROJECT_ROOT / "02_experiments" / "exp_011_stable_anchor_retrain" / "src"))

from dscr_exp011_lib import (  # noqa: E402
    BASE_ROUNDS, Dataset, S, VALID_START, VALID_STOP, feature_cols_for_config,
    get_cached_prediction, group_rank_transform, interval_target, model_fingerprint,
    score_prediction,
)

RESULT_DIR = PROJECT_ROOT / "04_results" / "exp_011_stable_anchor_retrain"
exp004_dir = PROJECT_ROOT / "04_results" / "exp_004_model_ensemble"
ds = Dataset(PROJECT_ROOT / "03_cache" / "processed_data_v1", check_sha256=False)

tcn_valid = np.load(exp004_dir / "component_valid_tcn.npy").astype(np.float32)
lin_valid = np.load(exp004_dir / "component_valid_linear.npy").astype(np.float32)
assert tcn_valid.shape == (243, S) and lin_valid.shape == (243, S)

valid_times = np.asarray(ds.common["valid"]["time"], dtype=np.int32)
valid_stocks = np.asarray(ds.common["valid"]["stock"], dtype=np.int32)
tcn_vec = tcn_valid[valid_times - VALID_START, valid_stocks]
lin_vec = lin_valid[valid_times - VALID_START, valid_stocks]
y_valid = interval_target(ds, "valid", VALID_START, VALID_STOP)
groups_v = np.asarray(ds.common["valid"]["groups"], dtype=np.int32)

key_base = model_fingerprint("anchor", [("train", 486, VALID_START)], BASE_ROUNDS, 42, "full_328")
cols = feature_cols_for_config("full_328")
cols_hash = hashlib.sha256(np.ascontiguousarray(cols).tobytes()).hexdigest()[:8]
anchor_vec = get_cached_prediction(RESULT_DIR / "runtime_cache",
                                   f"{key_base}_{cols_hash}_valid_{VALID_START}_{VALID_STOP}", int(y_valid.size))
assert anchor_vec is not None, "锚点 valid 缓存缺失"

a_rank = group_rank_transform(anchor_vec, groups_v)
t_rank = group_rank_transform(tcn_vec, groups_v)
l_rank = group_rank_transform(lin_vec, groups_v)
base_m = score_prediction(a_rank, y_valid, groups_v)
rows = []
for wa, wt, wl in ((0.85, 0.10, 0.05), (0.85, 0.15, 0.00), (0.90, 0.05, 0.05), (0.80, 0.15, 0.05)):
    blend = wa * a_rank + wt * t_rank + wl * l_rank
    mb = score_prediction(blend, y_valid, groups_v)
    rows.append({"anchor_w": wa, "tcn_w": wt, "linear_w": wl, "valid_rankic": mb["mean_rankic"],
                 "delta": mb["mean_rankic"] - base_m["mean_rankic"],
                 "second_half": mb["second_half_rankic"], "worst_quarter": mb["worst_quarter_rankic"]})
    print(f"blend ({wa},{wt},{wl}): {mb['mean_rankic']:.6f} delta {mb['mean_rankic'] - base_m['mean_rankic']:+.6f}")
rows.append({"anchor_w": 1.0, "tcn_w": 0.0, "linear_w": 0.0, "valid_rankic": base_m["mean_rankic"],
             "delta": 0.0, "second_half": base_m["second_half_rankic"], "worst_quarter": base_m["worst_quarter_rankic"]})
pd.DataFrame(rows).to_csv(RESULT_DIR / "tiny_blend_results.csv", index=False, encoding="utf-8-sig")
print("tiny_blend_results.csv 已保存。")
