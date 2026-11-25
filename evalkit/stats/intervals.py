"""Confidence intervals: Wilson score and (paired) bootstrap (Plan.md §10)."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np


def wilson(passes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion; (0, 0) for n = 0."""
    if n <= 0:
        return (0.0, 0.0)
    p = passes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_ci(
    values: Sequence[float],
    stat: Callable[[np.ndarray], float] = np.mean,  # type: ignore[assignment]
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap CI of `stat` over `values`."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    stats = np.apply_along_axis(stat, 1, arr[idx]) if stat is not np.mean else arr[idx].mean(1)
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return (float(lo), float(hi))


def paired_bootstrap_diff(
    a: Sequence[float],
    b: Sequence[float],
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float]:
    """Mean(b) − mean(a) with a CI from resampling case indices jointly.

    Returns diff, lo, hi and a two-sided bootstrap p-value for diff = 0.
    """
    xa, xb = np.asarray(list(a), dtype=float), np.asarray(list(b), dtype=float)
    if xa.size != xb.size:
        raise ValueError("paired bootstrap needs equal-length samples")
    if xa.size == 0:
        return {"diff": 0.0, "lo": 0.0, "hi": 0.0, "p": 1.0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, xa.size, size=(n_boot, xa.size))
    diffs = xb[idx].mean(1) - xa[idx].mean(1)
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    diff = float(xb.mean() - xa.mean())
    p = 1.0 if diff == 0 else float(min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())))
    return {"diff": diff, "lo": float(lo), "hi": float(hi), "p": p}
