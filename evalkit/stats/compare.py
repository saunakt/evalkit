"""Paired comparisons: exact McNemar, slice deltas (Plan.md §10)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from scipy.stats import binomtest

from evalkit.stats.intervals import paired_bootstrap_diff


def mcnemar_exact(a_pass: Sequence[bool], b_pass: Sequence[bool]) -> dict[str, Any]:
    """Exact McNemar on matched cases.

    b = A pass & B fail, c = A fail & B pass; p = two-sided binomtest(min(b,c), b+c, 0.5).
    """
    if len(a_pass) != len(b_pass):
        raise ValueError("McNemar needs matched samples")
    b = sum(1 for x, y in zip(a_pass, b_pass, strict=True) if x and not y)
    c = sum(1 for x, y in zip(a_pass, b_pass, strict=True) if not x and y)
    n = b + c
    p = 1.0 if n == 0 else float(binomtest(min(b, c), n, 0.5).pvalue)
    return {"b": b, "c": c, "p": p, "n_discordant": n}


def slice_delta(
    base: dict[str, bool], cand: dict[str, bool], seed: int = 42, n_boot: int = 2000
) -> dict[str, Any]:
    """Candidate − base pass rate on matched case ids, with bootstrap CI and McNemar."""
    ids = sorted(set(base) & set(cand))
    a = [base[i] for i in ids]
    b = [cand[i] for i in ids]
    boot = paired_bootstrap_diff([float(x) for x in a], [float(x) for x in b], n_boot, seed=seed)
    mc = mcnemar_exact(a, b)
    return {
        "n": len(ids),
        "base_rate": sum(a) / len(a) if a else 0.0,
        "cand_rate": sum(b) / len(b) if b else 0.0,
        "delta": boot["diff"],
        "lo": boot["lo"],
        "hi": boot["hi"],
        "p_mcnemar": mc["p"],
        "fixed": [i for i, x, y in zip(ids, a, b, strict=True) if not x and y],
        "broken": [i for i, x, y in zip(ids, a, b, strict=True) if x and not y],
    }
