"""Quality gate: thresholds, must-pass, regression vs baseline (Plan.md §11.1)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from evalkit.config import Bound, Thresholds
from evalkit.reporting.aggregate import MetricSummary, SuiteSummary
from evalkit.schema import ResultRow
from evalkit.stats.compare import slice_delta

Status = Literal["PASS", "FAIL", "WARN"]
MIN_N = 3


@dataclass
class Check:
    metric: str
    value: float | None
    bound: str
    target: float
    ok: bool
    basis: str  # "value" or "pass_rate"


@dataclass
class SuiteGate:
    suite: str
    status: Status = "PASS"
    checks: list[Check] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    regressions: list[dict[str, Any]] = field(default_factory=list)
    comparison: dict[str, Any] = field(default_factory=dict)


@dataclass
class GateResult:
    passed: bool
    suites: dict[str, SuiteGate]
    must_pass_failures: list[dict[str, Any]]

    @property
    def exit_code(self) -> int:
        return 0 if self.passed else 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "suites": {k: asdict(v) for k, v in self.suites.items()},
            "must_pass_failures": self.must_pass_failures,
        }


def threshold_for(thr: Thresholds, suite: str, ms: MetricSummary) -> tuple[str, float, str] | None:
    """(bound, target, basis) for a metric, or None when it is informational."""
    explicit = thr.suites.get(suite, {}).get(ms.metric)
    if explicit is not None:
        if isinstance(explicit, Bound):
            if explicit.min is not None:
                return ("min", explicit.min, "value")
            if explicit.max is not None:
                return ("max", explicit.max, "value")
            return None
        return ("min" if ms.direction == "higher" else "max", float(explicit), "value")
    if ms.scope == "case" and ms.n_scored:
        return ("min", thr.defaults.get("pass_rate", 0.9), "pass_rate")
    return None


def _check(suite: str, ms: MetricSummary, thr: Thresholds) -> Check | None:
    spec = threshold_for(thr, suite, ms)
    if spec is None:
        if ms.scope == "dataset" and ms.passed is not None:
            return Check(ms.metric, ms.value, "self", 1.0, bool(ms.passed), "metric")
        return None
    bound, target, basis = spec
    value = ms.pass_rate if basis == "pass_rate" else ms.value
    if value is None:
        return None
    ok = value >= target if bound == "min" else value <= target
    return Check(ms.metric, value, bound, target, ok, basis)


def must_pass_failures(rows: list[ResultRow]) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        if r.must_pass and r.case_id is not None and r.skipped is None and r.passed is False:
            out.append(
                {"suite": r.suite, "case_id": r.case_id, "metric": r.metric, "repeat": r.repeat}
            )
    return sorted(out, key=lambda d: (d["suite"], d["case_id"], d["metric"], d["repeat"] or 0))


def _case_pass(rows: list[ResultRow], metric: str) -> dict[str, bool]:
    """Case passes a metric when every repeat passed."""
    acc: dict[str, bool] = defaultdict(lambda: True)
    for r in rows:
        if r.metric == metric and r.case_id and r.skipped is None and r.passed is not None:
            acc[r.case_id] &= bool(r.passed)
    return dict(acc)


def regression_checks(
    suite: str, cur: list[ResultRow], base: list[ResultRow], thr: Thresholds
) -> dict[str, Any]:
    """Per metric: fixed / newly broken cases, delta with CI and McNemar p."""
    out: dict[str, Any] = {}
    metrics = sorted({r.metric for r in cur if r.case_id and r.passed is not None})
    for m in metrics:
        a, b = _case_pass(base, m), _case_pass(cur, m)
        if not set(a) & set(b):
            continue
        d = slice_delta(a, b)
        drop = -d["delta"]
        significant = d["p_mcnemar"] < 0.05 or not thr.regression.require_significance
        d["regressed"] = drop > thr.regression.max_slice_drop and significant
        out[m] = d
    return out


def evaluate(
    thr: Thresholds,
    summaries: dict[str, SuiteSummary],
    rows: list[ResultRow],
    baseline_rows: list[ResultRow] | None = None,
) -> GateResult:
    suites: dict[str, SuiteGate] = {}
    mp = must_pass_failures(rows)
    for name, summ in summaries.items():
        sg = SuiteGate(name)
        for ms in summ.metrics.values():
            _apply_metric(sg, name, ms, thr)
        for f in (f for f in mp if f["suite"] == name):
            sg.failures.append(
                f"must-pass case {f['case_id']} failed {f['metric']} (repeat {f['repeat']})"
            )
        if baseline_rows is not None:
            base = [r for r in baseline_rows if r.suite == name]
            sg.comparison = regression_checks(name, [r for r in rows if r.suite == name], base, thr)
            for m, d in sg.comparison.items():
                if d["regressed"]:
                    sg.failures.append(
                        f"{m} regressed vs baseline: {d['base_rate']:.2f} → "
                        f"{d['cand_rate']:.2f} (p={d['p_mcnemar']:.3f})"
                    )
        sg.status = "FAIL" if sg.failures else ("WARN" if sg.warnings else "PASS")
        suites[name] = sg
    passed = all(s.status != "FAIL" for s in suites.values()) and not mp
    return GateResult(passed, suites, mp)


def _apply_metric(sg: SuiteGate, suite: str, ms: MetricSummary, thr: Thresholds) -> None:
    if ms.not_tested:
        reasons = ", ".join(ms.skipped_reasons) or "no applicable cases"
        sg.warnings.append(f"{ms.metric} not tested: {reasons}")
        return
    if ms.uncalibrated:
        sg.warnings.append(f"{ms.metric} uses an uncalibrated judge")
    if ms.scope == "case" and 0 < ms.n_scored < MIN_N:
        sg.warnings.append(f"{ms.metric}: low n ({ms.n_scored})")
    check = _check(suite, ms, thr)
    if check is None:
        return
    sg.checks.append(check)
    if not check.ok:
        val = f"{check.value:.3f}" if check.value is not None else "n/a"
        sg.failures.append(
            f"{ms.metric} {check.basis} {val} {'<' if check.bound == 'min' else '>'} {check.target}"
            if check.bound != "self"
            else f"{ms.metric} failed"
        )
