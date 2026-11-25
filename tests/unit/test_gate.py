from __future__ import annotations

from pathlib import Path
from typing import Any

from evalkit.config import Bound, RegressionConfig, Thresholds
from evalkit.gate import evaluate, must_pass_failures
from evalkit.reporting.aggregate import summarize_suite
from evalkit.reporting.store import (
    ResultStore,
    find_baseline,
    list_runs,
    load_run,
    resolve_run,
    set_baseline,
)
from evalkit.schema import ResultRow, RunInfo


def row(
    case: str | None,
    metric: str,
    passed: bool | None,
    value: float | None = None,
    repeat: int = 0,
    must: bool = False,
    skipped: str | None = None,
    suite: str = "quality",
    **details: Any,
) -> ResultRow:
    return ResultRow(
        run_id="r",
        suite=suite,
        case_id=case,
        repeat=repeat if case else None,
        metric=metric,
        value=value if value is not None else (None if passed is None else float(passed)),
        passed=passed,
        threshold=None,
        skipped=skipped,
        details=details,
        tags=["t"],
        language=None,
        split="dev",
        must_pass=must,
        latency_s=None,
        tokens=None,
        cost_usd=None,
        trace_id=None,
    )


def gate_for(
    rows: list[ResultRow], thr: Thresholds, metrics: list[str], base: list[ResultRow] | None = None
) -> Any:
    summ = {"quality": summarize_suite("quality", rows, metrics)}
    return evaluate(thr, summ, rows, base)


def test_default_pass_rate_threshold() -> None:
    rows = [row(f"c{i}", "contains", i < 9) for i in range(10)]
    g = gate_for(rows, Thresholds(defaults={"pass_rate": 0.9}), ["contains"])
    assert g.passed and g.suites["quality"].status == "PASS"
    rows[0] = row("c0", "contains", False)
    g = gate_for(rows, Thresholds(defaults={"pass_rate": 0.9}), ["contains"])
    assert not g.passed and "contains pass_rate 0.800 < 0.9" in g.suites["quality"].failures[0]


def test_explicit_min_and_max_bounds() -> None:
    rows = [row(None, "attack_success_rate", None, value=0.05)]
    thr = Thresholds(suites={"quality": {"attack_success_rate": Bound(max=0.02)}})
    g = gate_for(rows, thr, ["attack_success_rate"])
    assert g.suites["quality"].status == "FAIL"
    thr = Thresholds(suites={"quality": {"attack_success_rate": Bound(max=0.1)}})
    assert gate_for(rows, thr, ["attack_success_rate"]).passed


def test_must_pass_fails_gate_even_when_rate_ok() -> None:
    rows = [row(f"c{i}", "contains", True) for i in range(20)]
    rows.append(row("c0", "contains", False, repeat=1, must=True))
    g = gate_for(rows, Thresholds(defaults={"pass_rate": 0.5}), ["contains"])
    assert not g.passed
    assert must_pass_failures(rows) == [
        {"suite": "quality", "case_id": "c0", "metric": "contains", "repeat": 1}
    ]


def test_warn_for_not_tested_uncalibrated_and_low_n() -> None:
    rows = [
        row("c1", "bleu", None, skipped="sacrebleu not installed"),
        row("c1", "judge_binary", True, uncalibrated=True),
    ]
    g = gate_for(rows, Thresholds(), ["bleu", "judge_binary"])
    sg = g.suites["quality"]
    assert sg.status == "WARN" and g.passed and g.exit_code == 0
    text = " ".join(sg.warnings)
    assert "bleu not tested: sacrebleu not installed" in text
    assert "uncalibrated" in text and "low n" in text


def test_regression_vs_baseline_requires_significance() -> None:
    base = [row(f"c{i}", "contains", True) for i in range(30)]
    cur = [row(f"c{i}", "contains", i >= 12) for i in range(30)]
    thr = Thresholds(defaults={"pass_rate": 0.0}, regression=RegressionConfig(max_slice_drop=0.05))
    g = gate_for(cur, thr, ["contains"], base)
    comp = g.suites["quality"].comparison["contains"]
    assert comp["broken"][:2] == ["c0", "c1"] and comp["regressed"]
    assert not g.passed
    small = [row(f"c{i}", "contains", i >= 2) for i in range(30)]
    g2 = gate_for(small, thr, ["contains"], base)
    assert not g2.suites["quality"].comparison["contains"]["regressed"]  # p >= 0.05


def test_store_roundtrip_and_baseline(tmp_path: Path) -> None:
    for rid in ("run-a", "run-b"):
        st = ResultStore(tmp_path / rid)
        st.write_info(RunInfo(run_id=rid, started="now", suites={"quality": ["contains"]}))
        st.write_rows([row("c1", "contains", True)])
    assert list_runs(tmp_path) == ["run-a", "run-b"]
    assert resolve_run(tmp_path, "last").name == "run-b"
    loaded = load_run(tmp_path / "run-a")
    assert loaded.rows[0].metric == "contains" and loaded.suites() == ["quality"]
    assert find_baseline(tmp_path, "auto", "run-b") is None
    set_baseline(tmp_path, "run-a")
    base = find_baseline(tmp_path, "auto", "run-b")
    assert base is not None and base.run_id == "run-a"
    assert find_baseline(tmp_path, "auto", "run-a") is None
