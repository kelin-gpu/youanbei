"""Read-only project audit for environment, artifacts, causal markers and contracts."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata as metadata
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "03_cache" / "processed_data_v1"
FORMAL = ROOT / "04_results" / "final_submission" / "prediction.npy"
BASELINE_PREDICTIONS = {
    "exp021_stack": ROOT / "04_results" / "exp_021_retrain_head_router" / "prediction_1.npy",
    "exp023h_previous_best": ROOT / "04_results" / "exp_023h_ultimate_surgery" / "prediction_1.npy",
    "exp024b_best_candidate": ROOT / "04_results" / "exp_024b_retrieval_exploratory" / "prediction_1.npy",
    "formal_submission": FORMAL,
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def package_version(distribution: str, module: str | None = None) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        try:
            return str(getattr(importlib.import_module(module or distribution.replace("-", "_")), "__version__", "installed"))
        except Exception:
            return "missing"


def prediction_contract(path: Path) -> dict[str, Any]:
    try:
        import numpy as np
        array = np.load(path, mmap_mode="r")
        contract: dict[str, Any] = {
            "exists": True,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "finite": bool(np.isfinite(array).all()),
            "minimum": float(array.min()),
            "maximum": float(array.max()),
        }
        expected_shape = (442, 5282)
        contract["shape_matches"] = tuple(array.shape) == expected_shape
        contract["dtype_matches"] = str(array.dtype) == "float32"
        time_path = DATASET / "common" / "test_time.npy"
        stock_path = DATASET / "common" / "test_stock.npy"
        if tuple(array.shape) == expected_shape and time_path.exists() and stock_path.exists():
            mask = np.zeros(expected_shape, dtype=bool)
            times = np.load(time_path, mmap_mode="r").astype(np.int64, copy=False)
            stocks = np.load(stock_path, mmap_mode="r").astype(np.int64, copy=False)
            mask[times - 3161, stocks] = True
            contract["evaluation_count"] = int(mask.sum())
            contract["non_evaluation_count"] = int((~mask).sum())
            contract["non_evaluation_all_0_5"] = bool(np.all(array[~mask] == 0.5))
        else:
            contract["mask_check"] = "missing_inputs"
        return contract
    except Exception as exc:
        return {"exists": path.exists(), "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-safety", action="store_true", help="run the existing dependency-free safety test")
    args = parser.parse_args()

    manifest = DATASET / "manifest.json"
    ready = DATASET / "READY"
    ready_data = json.loads(ready.read_text(encoding="utf-8")) if ready.exists() else {}
    manifest_hash = digest(manifest) if manifest.exists() else "missing"
    status_path = ROOT / "project_status.json"
    status_data = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
    result: dict[str, Any] = {
        "project_root": str(ROOT),
        "python": sys.version,
        "platform": platform.platform(),
        "interpreter": sys.executable,
        "packages": {
            "numpy": package_version("numpy"),
            "scipy": package_version("scipy"),
            "pandas": package_version("pandas"),
            "scikit-learn": package_version("scikit-learn", "sklearn"),
            "torch": package_version("torch"),
            "lightgbm": package_version("lightgbm"),
            "catboost": package_version("catboost"),
            "xgboost": package_version("xgboost"),
            "zstandard": package_version("zstandard"),
        },
        "cache": {
            "ready_exists": ready.exists(),
            "manifest_exists": manifest.exists(),
            "manifest_sha256": manifest_hash,
            "ready_manifest_sha256": ready_data.get("manifest_sha256", "missing"),
            "manifest_matches_ready": manifest_hash == ready_data.get("manifest_sha256"),
        },
        "formal_submission": {
            "path": str(FORMAL.relative_to(ROOT)).replace("\\", "/"),
            "sha256": digest(FORMAL) if FORMAL.exists() else "missing",
            "contract": prediction_contract(FORMAL),
        },
        "baselines": {
            name: {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": digest(path) if path.exists() else "missing",
                "contract": prediction_contract(path),
            }
            for name, path in BASELINE_PREDICTIONS.items()
        },
        "source_checks": {
            "test_labels_not_declared_in_data_context": "test_y" not in (ROOT / "02_experiments" / "exp_016_unified_expert_fusion" / "src" / "data_context.py").read_text(encoding="utf-8"),
            "exp023a_invalid_registered": "invalid_future_information_probe" in (ROOT / "project_status.json").read_text(encoding="utf-8"),
            "training_guard_present": "DSCR_EXP016_ALLOW_TRAINING" in (ROOT / "02_experiments" / "exp_016_unified_expert_fusion" / "config.py").read_text(encoding="utf-8"),
            "status_hashes_match_artifacts": all(
                status_data.get("baselines", {}).get(section, {}).get("sha256") == digest(path)
                for section, path in {
                    "stack": BASELINE_PREDICTIONS["exp021_stack"],
                    "best_candidate": BASELINE_PREDICTIONS["exp024b_best_candidate"],
                    "formal_submission": BASELINE_PREDICTIONS["formal_submission"],
                }.items()
                if path.exists()
            ),
        },
        "causal_policy": {
            "exp023a": "invalid_future_information_probe",
            "new_experiments": "must_pass_causal_audit",
        },
        "training_performed": False,
    }
    if args.run_safety:
        import subprocess
        test = ROOT / "02_experiments" / "exp_016_unified_expert_fusion" / "tests" / "test_safety_contract.py"
        completed = subprocess.run([sys.executable, str(test)], cwd=ROOT, text=True, capture_output=True, check=False)
        result["safety_contract"] = {
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    required = ("finite", "shape_matches", "dtype_matches", "non_evaluation_all_0_5")
    baseline_contracts = result["baselines"].values()
    valid_contract = all(all(contract.get(key, False) for key in required) for contract in (item["contract"] for item in baseline_contracts))
    source_checks = all(result["source_checks"].values())
    return 0 if result["cache"]["manifest_matches_ready"] and valid_contract and source_checks else 1


if __name__ == "__main__":
    raise SystemExit(main())
