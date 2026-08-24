from __future__ import annotations

"""exp032 step 1: extract raw 99-feature slices from data.z into a cache.

Decompresses data.z via a temp file to cap peak memory (~11GB), slices
num_x[1885:3161) (diagnostic windows plus rolling history), and saves to
03_cache/exp_032_raw_feature_bank/raw_num.npy for mmap consumption by the
diagnostic script. Also records an alignment check between raw stock grid and
the processed cache's per-section stock ids.
"""

import json
import pickle
import time
from pathlib import Path

import numpy as np
import zstandard as zstd

ROOT = Path(r"d:\google_dl\book\youanbei")
DATA_Z = ROOT / "data.z"
CACHE = ROOT / "03_cache" / "exp_032_raw_feature_bank"
TEMP = CACHE / "_decompressed_payload.pkl"
SLICE_START, SLICE_STOP = 1885, 3161

EXPECTED_KEYS = {"num_x", "cat_x", "y1", "mask_x", "mask_y",
                 "train_start_idx", "valid_start_idx", "test_start_idx"}


def main() -> int:
    started = time.time()
    CACHE.mkdir(parents=True, exist_ok=True)
    out_path = CACHE / "raw_num.npy"
    if out_path.exists():
        print(f"[extract] cache already exists: {out_path}", flush=True)
        return 0

    with DATA_Z.open("rb") as handle:
        compressed = pickle.load(handle)
    print(f"[extract] compressed payload loaded: {len(compressed) / 1e9:.2f} GB", flush=True)
    decompressed = zstd.ZstdDecompressor().decompress(compressed)
    print(f"[extract] decompressed: {len(decompressed) / 1e9:.2f} GB", flush=True)
    del compressed
    with TEMP.open("wb") as handle:
        handle.write(decompressed)
    del decompressed
    import gc
    gc.collect()
    print("[extract] temp file written, memory freed", flush=True)

    with TEMP.open("rb") as handle:
        data = pickle.load(handle)
    missing = EXPECTED_KEYS - set(data.keys())
    if missing:
        raise RuntimeError(f"payload missing keys: {missing}")
    num_x = data["num_x"]
    print(f"[extract] num_x shape={num_x.shape} dtype={num_x.dtype}", flush=True)

    raw_slice = np.ascontiguousarray(num_x[SLICE_START:SLICE_STOP])
    del data, num_x
    gc.collect()
    np.save(out_path, raw_slice)
    print(f"[extract] saved {out_path}: shape={raw_slice.shape}", flush=True)

    del raw_slice
    gc.collect()
    TEMP.unlink(missing_ok=True)

    loaded = np.load(out_path, mmap_mode="r")
    alignment = verify_alignment(loaded)
    (CACHE / "extraction_metadata.json").write_text(json.dumps({
        "source": str(DATA_Z),
        "slice": [SLICE_START, SLICE_STOP],
        "shape": list(loaded.shape),
        "dtype": str(loaded.dtype),
        "alignment": alignment,
        "elapsed_s": round(time.time() - started, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[extract] alignment check: {alignment}", flush=True)
    print(f"[extract] done in {time.time() - started:.0f}s", flush=True)
    return 0


def verify_alignment(raw_slice: np.ndarray) -> dict:
    common = ROOT / "03_cache" / "processed_data_v1" / "common"
    train_groups = np.asarray(np.load(common / "train_group_sizes.npy"), dtype=np.int64)
    train_times = np.asarray(np.load(common / "train_time.npy", mmap_mode="r"), dtype=np.int32)
    train_stocks = np.asarray(np.load(common / "train_stock.npy", mmap_mode="r"), dtype=np.int32)
    offsets = np.concatenate([[0], np.cumsum(train_groups)])
    checks = []
    for probe in (1945, 2200, 2432, 2700, 2917):
        group_index = int(np.flatnonzero(train_times[offsets[:-1]] == probe)[0])
        left, right = int(offsets[group_index]), int(offsets[group_index + 1])
        stocks = train_stocks[left:right]
        raw_t = int(probe - SLICE_START)
        finite_all = np.isfinite(raw_slice[raw_t][stocks]).mean(axis=0)
        checks.append({
            "time": probe,
            "rows": int(stocks.size),
            "stock_id_range": [int(stocks.min()), int(stocks.max())],
            "mean_finite_rate_all99": round(float(finite_all.mean()), 4),
        })
    return {"probes": checks}


if __name__ == "__main__":
    raise SystemExit(main())
