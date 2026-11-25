"""Cross-cutting statistics available in every suite's report (Plan.md §9.19).

Registered with `suites=("*",)` for the catalog; the reporting layer computes them for every
suite (overall pass rate CI, flaky rate, slices, sample-size advice, judge agreement).
`pass_at_k`, `pass_hat_k` and `mcnemar` are registered by the agent / migration suites.
"""

from __future__ import annotations

from evalkit.metrics.common import na, passes_by_case, result
from evalkit.registry import metric
from evalkit.schema import CaseRun, MetricResult
from evalkit.stats.intervals import bootstrap_ci, wilson
from evalkit.stats.reliability import case_pass_rates, flaky
from evalkit.stats.sample_size import cases_needed
from evalkit.suites.context import MetricContext

EVERY = ("*",)


def _all_passes(rows: list[CaseRun]) -> list[bool]:
    return [
        bool(r.passed)
        for cr in rows
        for r in cr.results
        if r.passed is not None and r.skipped is None
    ]


@metric("pass_rate_ci", suites=EVERY, kind="statistical", scope="dataset")
def pass_rate_ci(rows: list[CaseRun], ctx: MetricContext) -> MetricResult:
    """Overall pass rate with a Wilson 95% interval."""
    vals = _all_passes(rows)
    if not vals:
        return na("no scored results")
    lo, hi = wilson(sum(vals), len(vals))
    return result(sum(vals) / len(vals), None, lo=lo, hi=hi, n=len(vals))


@metric("flaky_rate", suites=EVERY, kind="statistical", scope="dataset", direction="lower")
def flaky_rate(rows: list[CaseRun], ctx: MetricContext) -> MetricResult:
    """Share of cases with 0 < pass rate < 1 across repeats."""
    per_case: dict[str, list[bool]] = {}
    for cr in rows:
        by_rep: dict[int, bool] = {}
        for r, rep in zip(cr.results, cr.repeats, strict=True):
            if r.passed is not None and r.skipped is None:
                by_rep[rep] = by_rep.get(rep, True) and bool(r.passed)
        if by_rep:
            per_case[cr.case.id] = list(by_rep.values())
    if not per_case:
        return na("no scored results")
    ids = flaky(case_pass_rates(per_case))
    return result(len(ids) / len(per_case), None, flaky=ids)


@metric("bootstrap_ci", suites=EVERY, kind="statistical", scope="dataset")
def bootstrap_ci_metric(rows: list[CaseRun], ctx: MetricContext) -> MetricResult:
    """Bootstrap 95% CI of the mean metric value."""
    vals = [r.value for cr in rows for r in cr.results if r.value is not None and not r.skipped]
    if not vals:
        return na("no values")
    lo, hi = bootstrap_ci(vals, seed=ctx.seed, n_boot=2000)
    return result(sum(vals) / len(vals), None, lo=lo, hi=hi, n=len(vals))


@metric("slice_report", suites=EVERY, kind="statistical", scope="dataset")
def slice_report(rows: list[CaseRun], ctx: MetricContext) -> MetricResult:
    """Pass rate per tag (full slice tables are in every report)."""
    by_tag: dict[str, list[bool]] = {}
    for cr in rows:
        for tag in cr.case.tags or ["(none)"]:
            by_tag.setdefault(tag, []).extend(
                bool(r.passed) for r in cr.results if r.passed is not None and not r.skipped
            )
    table = {t: sum(v) / len(v) for t, v in sorted(by_tag.items()) if v}
    return result(None, None, by_tag=table) if table else na("no scored results")


@metric("judge_agreement", suites=EVERY, kind="statistical", scope="dataset")
def judge_agreement_metric(rows: list[CaseRun], ctx: MetricContext) -> MetricResult:
    """Calibration status of the judges used (from calibration records)."""
    used = ctx.judge.used
    if not used:
        return na("no judges used")
    return result(None, None, judges=used)


@metric("sample_size", suites=EVERY, kind="statistical", scope="dataset", unit="count")
def sample_size(rows: list[CaseRun], ctx: MetricContext) -> MetricResult:
    """Cases needed for ±params.margin (default 0.05) at the observed pass rate."""
    per_case = passes_by_case(rows, ctx.metric_id) or {
        cr.case.id: [bool(r.passed) for r in cr.results if r.passed is not None] for cr in rows
    }
    vals = [v for lst in per_case.values() for v in lst]
    if not vals:
        return na("no scored results")
    p = sum(vals) / len(vals)
    margin = float(ctx.p("margin", 0.05))
    return result(
        float(cases_needed(p, margin)),
        None,
        observed_rate=p,
        margin=margin,
        current_cases=len(per_case),
    )
