"""Statistics reproduce the guide's hand-computed numbers (Plan.md §10)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from evalkit.stats.agreement import cohen_kappa, judge_agreement, krippendorff_alpha_nominal
from evalkit.stats.compare import mcnemar_exact, slice_delta
from evalkit.stats.intervals import bootstrap_ci, paired_bootstrap_diff, wilson
from evalkit.stats.reliability import (
    case_pass_rates,
    flaky,
    pass_at_k_empirical,
    pass_at_k_estimate,
    pass_hat_k_empirical,
    pass_hat_k_estimate,
)
from evalkit.stats.sample_size import cases_needed, slice_table


def test_wilson_45_of_50() -> None:
    lo, hi = wilson(45, 50)
    assert lo == pytest.approx(0.786, abs=1e-3)
    assert hi == pytest.approx(0.957, abs=1e-3)
    assert wilson(0, 0) == (0.0, 0.0)


def _mcnemar(b: int, c: int) -> dict[str, float]:
    a_pass = [True] * b + [False] * c + [True] * 10
    b_pass = [False] * b + [True] * c + [True] * 10
    return mcnemar_exact(a_pass, b_pass)


def test_mcnemar_guide_values() -> None:
    r = _mcnemar(9, 3)
    assert (r["b"], r["c"], r["n_discordant"]) == (9, 3, 12)
    assert r["p"] == pytest.approx(0.146, abs=1e-3)
    assert _mcnemar(19, 3)["p"] < 0.001
    assert mcnemar_exact([True, False], [True, False])["p"] == 1.0


def test_kappa_guide_example() -> None:
    human = ["F", "P", "P", "F", "P", "F", "P", "P"]
    judge = ["F", "P", "F", "F", "P", "P", "P", "P"]
    assert cohen_kappa(human, judge) == pytest.approx(0.47, abs=0.005)


def test_cases_needed() -> None:
    assert cases_needed(0.9, 0.05) == 139


def test_judge_agreement_fail_class() -> None:
    human = ["FAIL", "PASS", "PASS", "FAIL", "PASS", "FAIL", "PASS", "PASS"]
    judge = ["FAIL", "PASS", "FAIL", "FAIL", "PASS", "PASS", "PASS", "PASS"]
    r = judge_agreement(human, judge)
    assert r["accuracy"] == pytest.approx(0.75)
    assert r["fail_recall"] == pytest.approx(2 / 3)
    assert r["fail_precision"] == pytest.approx(2 / 3)
    assert r["confusion"] == {"tp": 2, "fp": 1, "fn": 1, "tn": 4}


def test_krippendorff_alpha() -> None:
    assert krippendorff_alpha_nominal([["a", "b", "a"], ["a", "b", "a"]]) == pytest.approx(1.0)
    # Two coders disagreeing on every unit with balanced labels → alpha < 0.
    assert krippendorff_alpha_nominal([["a", "b"], ["b", "a"]]) < 0
    # Matches a hand calculation: 2 coders, 4 units, one disagreement.
    alpha = krippendorff_alpha_nominal([["a", "a", "b", "b"], ["a", "a", "b", "a"]])
    assert alpha == pytest.approx(0.5333, abs=1e-3)


def test_pass_at_k_family() -> None:
    res = {"a": [True, True, True], "b": [False, True, False], "c": [False, False, False]}
    assert pass_at_k_empirical(res) == pytest.approx(2 / 3)
    assert pass_hat_k_empirical(res) == pytest.approx(1 / 3)
    assert pass_at_k_empirical(res, k=1) == pytest.approx(1 / 3)
    assert pass_at_k_estimate(0.5, 3) == pytest.approx(0.875)
    assert pass_hat_k_estimate(0.5, 3) == pytest.approx(0.125)
    assert flaky(case_pass_rates(res)) == ["b"]


def test_bootstrap_is_seeded_and_brackets_mean() -> None:
    vals = [0, 1, 1, 1, 0, 1, 1, 1, 1, 0]
    lo, hi = bootstrap_ci(vals, seed=1, n_boot=2000)
    assert lo < 0.7 < hi
    assert (lo, hi) == bootstrap_ci(vals, seed=1, n_boot=2000)
    diff = paired_bootstrap_diff([1, 1, 1, 1], [0, 0, 1, 1], n_boot=500)
    assert diff["diff"] == pytest.approx(-0.5) and diff["hi"] <= 0


def test_slice_delta_lists_fixed_and_broken() -> None:
    d = slice_delta({"a": True, "b": False, "c": True}, {"a": False, "b": True, "c": True})
    assert d["fixed"] == ["b"] and d["broken"] == ["a"] and d["n"] == 3


def test_slice_table() -> None:
    rows = [
        SimpleNamespace(passed=True, skipped=None, tags=["x"], language="hi"),
        SimpleNamespace(passed=False, skipped=None, tags=["x", "y"], language="en"),
        SimpleNamespace(passed=None, skipped=None, tags=["x"], language="en"),
    ]
    by_tag = {r["slice"]: r for r in slice_table(rows, "tags")}
    assert by_tag["x"]["n"] == 2 and by_tag["x"]["pass_rate"] == 0.5
    assert by_tag["y"]["pass_rate"] == 0.0
    assert [r["slice"] for r in slice_table(rows, "language")] == ["en", "hi"]
