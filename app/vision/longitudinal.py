"""Deterministic longitudinal change detection helpers."""

from __future__ import annotations

import numpy as np


def aligned_difference_map(current: np.ndarray, previous: np.ndarray) -> np.ndarray:
    """Return absolute aligned difference map for two same-shape arrays."""
    if current.shape != previous.shape:
        raise ValueError("shape_mismatch")
    return np.abs(current.astype(np.float32) - previous.astype(np.float32))


def zscore_change_map(current: np.ndarray, baseline_stack: np.ndarray) -> np.ndarray:
    """Return per-pixel z-score map against baseline stack [N,H,W]."""
    mu = np.mean(baseline_stack.astype(np.float32), axis=0)
    sigma = np.std(baseline_stack.astype(np.float32), axis=0) + 1e-6
    return (current.astype(np.float32) - mu) / sigma


def per_region_trend_slopes(series_values: dict[str, list[float]]) -> dict[str, float]:
    """Compute linear trend slope per region over index-based time."""
    out: dict[str, float] = {}
    for key, vals in series_values.items():
        if len(vals) < 2:
            out[key] = 0.0
            continue
        x = np.arange(len(vals), dtype=np.float32)
        y = np.asarray(vals, dtype=np.float32)
        slope = float(np.polyfit(x, y, 1)[0])
        out[key] = slope
    return out
