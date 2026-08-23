from __future__ import annotations

"""exp_023c：exp021 融合栈 x exp023b 递归自举 的截面 rank 混合。

valid 选 w（栈=exp016 full_valid 动态融合输出），test 应用（栈=exp021 线上 0.116568 网格）。
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, r"D:\google_dl\book\youanbei\02_experiments")
from exp_016_unified_expert_fusion.src.ranking import group_rank, dynamic_blend_family_predictions
from exp_016_unified_expert_fusion.src.prediction_contract import vector_to_grid, validate_prediction

ROOT = Path(r"D:\google_dl\book\youanbei")
C = ROOT / "03_cache" / "processed_data_v1" / "common"
F16 = ROOT / "04_results" / "exp_016_unified_expert_fusion" / "full"
P021 = ROOT / "04_results" / "exp_021_retrain_head_router" / "prediction_1.npy"
P023B = ROOT / "04_results" / "exp_023b_recursion"
RESULT = ROOT / "04_results" / "exp_023c_recursion_blend"
RESULT.mkdir(parents=True, exist_ok=True)

TEST_START, TEST_TIME_POINTS, STOCK_COUNT = 3161, 442, 5282
FAMILIES = ("exp015_anchor", "tabular", "dual_axis", "time_frequency",
            "relational_graph", "foundation_representation", "multi_objective_rank")

va_g = np.load(C / "valid_group_sizes.npy")
va_y = np.load(C / "valid_y.npy")
fam = np.load(F16 / "full_valid_family_predictions.npy")
wts = np.load(F16 / "full_valid_dynamic_weights.npy")
assert fam.shape == (va_y.size, 7) and wts.shape == (len(va_g), 7), (fam.shape, wts.shape)

stack_va = dynamic_blend_family_predictions({n: fam[:, k] for k, n in enumerate(FAMILIES)}, va_g, wts)
rec_va = np.load(P023B / "valid_pred.npy").astype(np.float32)
assert rec_va.size == va_y.size


def ics_of(vec):
    out = []
    for size in va_g:
        size = int(size)
    off = 0
    for size in va_g:
        size = int(size)
        out.append(np.corrcoef(np.argsort(np.argsort(vec[off:off + size])), np.argsort(np.argsort(va_y[off:off + size])))[0, 1])
        off += size
    return np.asarray(out)


rs = group_rank(stack_va, va_g)
rr = group_rank(rec_va, va_g)
base_ics = ics_of(rs)
rec_ics = ics_of(rr)
corr = np.corrcoef(rs, rr)[0, 1]
print(f"stack valid IC = {base_ics.mean():+.4f}   rec valid IC = {rec_ics.mean():+.4f}   corr = {corr:.3f}", flush=True)

best_w, best_ic = 0.0, base_ics.mean()
grid = {}
for w in np.arange(0.05, 0.65, 0.05):
    v = ics_of((1 - w) * rs + w * rr)
    grid[f"{w:.2f}"] = float(v.mean())
    print(f"w={w:.2f}: {v.mean():+.4f}  pos={np.mean(v > 0):.2f}", flush=True)
    if v.mean() > best_ic:
        best_ic, best_w = float(v.mean()), float(w)
print(f"best w = {best_w:.2f} -> valid IC {best_ic:+.4f} (stack {base_ics.mean():+.4f}, +{best_ic - base_ics.mean():.4f})", flush=True)

# ---- test ----
te_g = np.load(C / "test_group_sizes.npy")
te_time = np.load(C / "test_time.npy").astype(np.int32)
te_stock = np.load(C / "test_stock.npy").astype(np.int32)
grid021 = np.load(P021)
stack_te = grid021[te_time - TEST_START, te_stock]
rec_te = np.load(P023B / "test_pred.npy").astype(np.float32)
assert stack_te.size == rec_te.size == te_time.size

blend_te = (1 - best_w) * group_rank(stack_te, te_g) + best_w * group_rank(rec_te, te_g)
final = group_rank(blend_te, te_g)
out = vector_to_grid(final, te_time, te_stock, TEST_START, TEST_TIME_POINTS)
mask = np.zeros((TEST_TIME_POINTS, STOCK_COUNT), dtype=bool)
mask[te_time - TEST_START, te_stock] = True
contract = validate_prediction(out, mask)
np.save(RESULT / "prediction_1.npy", out)

meta = {
    "experiment": "exp_023c_recursion_blend",
    "stack_valid_ic": float(base_ics.mean()), "recursion_valid_ic": float(rec_ics.mean()),
    "rank_corr_stack_rec": float(corr), "w_grid": grid, "best_w": float(best_w),
    "blend_valid_ic": float(best_ic), "delta_vs_stack": float(best_ic - base_ics.mean()),
    "stack_test_source": "exp021 prediction_1.npy (online 0.116568)",
    "recursion": "exp023b alpha=0.3 (valid 0.0926)",
    "contract": contract, "compliance": "第 t 截面仅用 X(<=t) 与历史自有预测；无 t 后数据",
}
(RESULT / "metrics.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({k: meta[k] for k in ("best_w", "blend_valid_ic", "delta_vs_stack", "contract")}, ensure_ascii=False), flush=True)
