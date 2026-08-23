from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "04_results" / "exp_027a_retrieval_attribution"
EXPECTED_FORMAL_HASH = "6ff796c7c222bb3e9a55077014c1fc885dca368e3ac1bd29aecafb4e58e2be55"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    protocol = json.loads((RESULT / "protocol.json").read_text(encoding="utf-8"))
    metrics = json.loads((RESULT / "metrics.json").read_text(encoding="utf-8"))
    metadata = json.loads((RESULT / "metadata.json").read_text(encoding="utf-8"))
    gates = protocol["decision_gates"]

    assert metrics["test_arrays_loaded"] is False
    assert metrics["prediction_generated"] is False
    assert metrics["online_submission_used"] is False
    assert metadata["test_prediction_generated"] is False
    assert metrics["protected_unchanged"] is True
    assert metrics["retrieval_reproduction_max_abs_error"] <= gates["retrieval_reproduction_tolerance"]
    assert metrics["fingerprint_decomposition_max_abs_error"] < gates["decomposition_tolerance"]
    assert not (RESULT / "prediction.npy").exists()
    assert not (RESULT / "prediction_1.npy").exists()

    with (RESULT / "per_time_metrics.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    retrieval = [row for row in rows if row["method"] == "retrieval"]
    assert len(retrieval) == 1702
    assert all(int(row["max_library_time"]) < int(row["library_cutoff_exclusive"]) for row in retrieval)

    formal = ROOT / "04_results" / "final_submission" / "prediction.npy"
    assert digest(formal) == EXPECTED_FORMAL_HASH
    print("EXP027A_CONTRACT_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
