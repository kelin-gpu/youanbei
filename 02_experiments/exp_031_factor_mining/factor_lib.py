from __future__ import annotations

"""Shared causal cross-sectional factor library for exp031.

compute_factor builds a factor vector from one section's feature block using only
same-section X(t) information (numeric pool columns and categorical columns 408-416).
extended_features assembles the exp024b container's feature matrix: the frozen 31
registry-allowed base columns plus mined factor columns. Both exp031a (diagnostic)
and exp031b (candidate generation) import this module so the deployed factor
definitions are bit-identical to the validated ones.
"""

from typing import Any

import numpy as np
from scipy.stats import rankdata

CAT_SLICE = slice(408, 417)


def rank01(values: np.ndarray) -> np.ndarray:
    ranked = rankdata(values, method="average")
    return (ranked - 1.0) / max(1, ranked.size - 1)


def group_demean(values: np.ndarray, cats: np.ndarray) -> np.ndarray:
    out = values - float(np.mean(values))
    finite = np.isfinite(cats)
    if int(finite.sum()) < 2:
        return out
    for value in np.unique(cats[finite]):
        members = finite & (cats == value)
        if int(members.sum()) >= 2:
            out[members] = values[members] - float(np.mean(values[members]))
    return out


def compute_factor(x_block: np.ndarray, spec: dict[str, Any],
                   base_positions: list[int]) -> np.ndarray:
    kind = spec["kind"]
    if kind == "base":
        return np.asarray(x_block[:, base_positions[spec["col"]]], dtype=np.float32)
    if kind == "rank":
        return rank01(x_block[:, base_positions[spec["col"]]]).astype(np.float32)
    if kind == "diff":
        return (rank01(x_block[:, base_positions[spec["a"]]])
                - rank01(x_block[:, base_positions[spec["b"]]])).astype(np.float32)
    if kind == "prod":
        return ((rank01(x_block[:, base_positions[spec["a"]]]) - 0.5)
                * (rank01(x_block[:, base_positions[spec["b"]]]) - 0.5)).astype(np.float32)
    if kind == "concept":
        ranks = rank01(x_block[:, base_positions[spec["col"]]])
        cats = np.asarray(x_block[:, 408 + spec["cat"]], dtype=np.float64)
        return group_demean(ranks, cats).astype(np.float32)
    if kind == "absdev":
        return np.abs(rank01(x_block[:, base_positions[spec["col"]]]) - 0.5).astype(np.float32)
    raise ValueError(f"unknown factor kind {kind}")


def factor_name(spec: dict[str, Any]) -> str:
    kind = spec["kind"]
    if kind == "base":
        return f"base_{spec['col']}"
    if kind == "rank":
        return f"rank_{spec['col']}"
    if kind == "diff":
        return f"diff_{spec['a']}_{spec['b']}"
    if kind == "prod":
        return f"prod_{spec['a']}_{spec['b']}"
    if kind == "concept":
        return f"concept_{spec['col']}_cat{spec['cat']}"
    if kind == "absdev":
        return f"absdev_{spec['col']}"
    raise ValueError(f"unknown factor kind {kind}")


def extended_features(x_block: np.ndarray, base_positions: list[int],
                      factor_specs: list[dict[str, Any]]) -> np.ndarray:
    columns = [np.asarray(x_block[:, base_positions], dtype=np.float32)]
    columns.extend(compute_factor(x_block, spec, base_positions).reshape(-1, 1)
                   for spec in factor_specs)
    return np.concatenate(columns, axis=1)
