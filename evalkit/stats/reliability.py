"""pass@k, pass^k and flakiness (Plan.md §10)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def pass_at_k_empirical(
    results_per_case: Mapping[str, Sequence[bool]], k: int | None = None
) -> float:
    """Share of cases where any of the first k repeats passed."""
    if not results_per_case:
        return 0.0
    hits = [any(list(r)[:k] if k else r) for r in results_per_case.values()]
    return sum(hits) / len(hits)


def pass_hat_k_empirical(
    results_per_case: Mapping[str, Sequence[bool]], k: int | None = None
) -> float:
    """Share of cases where all of the first k repeats passed."""
    if not results_per_case:
        return 0.0
    hits = [bool(r) and all(list(r)[:k] if k else r) for r in results_per_case.values()]
    return sum(hits) / len(hits)


def pass_at_k_estimate(p: float, k: int) -> float:
    return 1 - (1 - p) ** k


def pass_hat_k_estimate(p: float, k: int) -> float:
    return float(p**k)


def case_pass_rates(results_per_case: Mapping[str, Sequence[bool]]) -> dict[str, float]:
    return {cid: (sum(r) / len(r) if r else 0.0) for cid, r in sorted(results_per_case.items())}


def flaky(case_pass_rates_: Mapping[str, float]) -> list[str]:
    """Case ids with 0 < pass rate < 1, sorted."""
    return sorted(cid for cid, r in case_pass_rates_.items() if 0 < r < 1)
