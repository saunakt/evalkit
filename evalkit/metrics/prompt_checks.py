"""Prompt-suite metrics (Plan.md §9.2)."""

from __future__ import annotations

from typing import Any

from evalkit.metrics.common import boolean, contains_any, na, result
from evalkit.metrics.deterministic import json_schema, length, regex
from evalkit.metrics.judged import baseline_output, per_criterion, sections
from evalkit.registry import metric
from evalkit.schema import Case, CaseRun, MetricResult, Observation
from evalkit.suites.context import MetricContext
from evalkit.util import mean

POSITIVE = ("positive", "boundary")


@metric("instruction_positive", suites=("prompt",), kind="judge", needs=("criteria",))
def instruction_positive(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """judge_binary on criteria for cases with extra.polarity positive|boundary."""
    if case.extra.get("polarity", "positive") not in POSITIVE:
        return na("polarity is not positive/boundary")
    return per_criterion(ctx.judge, "generic_binary", case.criteria, case, obs)


@metric("instruction_negative", suites=("prompt",), kind="judge", needs=("criteria",))
def instruction_negative(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """judge_binary on criteria for cases with extra.polarity negative."""
    if case.extra.get("polarity") != "negative":
        return na("polarity is not negative")
    return per_criterion(ctx.judge, "generic_binary", case.criteria, case, obs)


@metric(
    "out_of_scope_redirect",
    suites=("prompt", "security"),
    kind="judge",
    needs=("tag:out_of_scope",),
)
def out_of_scope_redirect(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """Judge: declines politely + redirects to params.scope; plus not_contains."""
    present = contains_any(obs.text, case.must_not_include)
    crit = ctx.judge.get("out_of_scope_redirect").criterion.format(
        scope=ctx.p("scope", "the assistant's supported topics")
    )
    v = ctx.judge.binary(
        "out_of_scope_redirect", criterion=crit, sections=sections(case, obs), case_id=case.id
    )
    ok = bool(v.passed) and not present
    return result(
        1.0 if ok else 0.0,
        ok,
        reasoning=v.reasoning,
        forbidden_present=present,
        **({"uncalibrated": True} if v.details.get("uncalibrated") else {}),
    )


@metric("format_compliance", suites=("prompt",), kind="deterministic")
def format_compliance(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """All declared deterministic format checks on the case (schema, regex, length, language)."""
    checks: dict[str, Any] = {}
    if case.schema_ is not None:
        checks["json_schema"] = json_schema(case, obs, ctx).passed
    if case.regex:
        checks["regex"] = regex(case, obs, ctx).passed
    if case.extra.get("max_words") or ctx.p("max_words"):
        checks["length"] = length(case, obs, ctx).passed
    if case.language:
        from evalkit.metrics.language import dominant_script, script_of_language

        want = script_of_language(case.language)
        checks["language"] = want is None or dominant_script(obs.text) == want
    if not checks:
        return na("no format checks declared")
    ok = all(checks.values())
    return result(sum(bool(v) for v in checks.values()) / len(checks), ok, checks=checks)


@metric("instruction_conflict", suites=("prompt",), kind="judge", needs=("criteria",))
def instruction_conflict(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """Judge: the higher-priority instruction was followed (per criteria)."""
    tmpl = ctx.judge.get("instruction_conflict").criterion
    return per_criterion(ctx.judge, "instruction_conflict", case.criteria, case, obs, tmpl)


def args_subset(expected: dict[str, Any] | None, actual: dict[str, Any], tol: float = 1e-6) -> bool:
    """Every expected arg present and equal (strings normalised, numbers with tolerance)."""
    for key, want in (expected or {}).items():
        got = actual.get(key)
        if isinstance(want, int | float) and isinstance(got, int | float):
            if abs(float(want) - float(got)) > tol:
                return False
        elif str(want).strip().lower() != str(got).strip().lower():
            return False
    return True


@metric("tool_definition_choice", suites=("prompt",), kind="deterministic", needs=("expect_tools",))
def tool_definition_choice(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """Re-run with max_steps=1: first tool call == expect_tools[0] (+ args subset)."""
    first = ctx.suite.recall(case, max_steps=1)
    if not first.tool_calls:
        return boolean(
            False, reason="no tool call in the first step", expected=case.expect_tools[0]
        )
    call = first.tool_calls[0]
    want_args = case.expect_trajectory[0].args if case.expect_trajectory else None
    ok = call.name == case.expect_tools[0] and args_subset(want_args, call.args)
    return boolean(ok, called=call.name, args=call.args, expected=case.expect_tools[0])


def _case_pass(cr: CaseRun, metrics: set[str]) -> bool | None:
    vals = [
        r.passed
        for r in cr.results
        if r.metric in metrics and r.passed is not None and r.skipped is None
    ]
    return all(vals) if vals else None


@metric(
    "fixed_vs_broken",
    suites=("prompt",),
    kind="statistical",
    scope="dataset",
    direction="lower",
    unit="count",
)
def fixed_vs_broken(rows: list[CaseRun], ctx: MetricContext) -> MetricResult:
    """Vs the baseline run on the same case ids: fixed / broken / unchanged; value = broken."""
    base = ctx.baseline
    if base is None:
        return na("no baseline run")
    suite = ctx.suite.name
    base_pass: dict[str, bool] = {}
    for r in base.rows:
        if r.suite == suite and r.case_id and r.passed is not None and r.skipped is None:
            base_pass[r.case_id] = base_pass.get(r.case_id, True) and bool(r.passed)
    metrics = {r.metric for cr in rows for r in cr.results}
    fixed: list[str] = []
    broken: list[str] = []
    same: list[str] = []
    for cr in rows:
        now = _case_pass(cr, metrics)
        if now is None or cr.case.id not in base_pass:
            continue
        before = base_pass[cr.case.id]
        (fixed if now and not before else broken if before and not now else same).append(cr.case.id)
    return result(
        float(len(broken)), not broken, 0.0, fixed=fixed, broken=broken, unchanged=len(same)
    )


@metric("pairwise_win_rate", suites=("prompt", "migration"), kind="judge", scope="dataset")
def pairwise_win_rate(rows: list[CaseRun], ctx: MetricContext) -> MetricResult:
    """Mean position-swapped pairwise score of current outputs vs the baseline's."""
    scores, unstable = [], 0
    from evalkit.metrics.migration import matched

    pairs = [
        (cand.case, cand.observations[0].text, cur.observations[0].text)
        for _, cur, cand in matched(rows)
        if cur.observations and cand.observations
    ]
    items = (
        pairs
        if pairs
        else [
            (cr.case, cr.observations[0].text, baseline_output(cr.case, ctx))
            for cr in rows
            if cr.observations
        ]
    )
    for case, a, b in items:
        if b is None:
            continue
        v = ctx.judge.pairwise(
            "pairwise", a=a, b=b, sections={"User input": case.text_input()}, case_id=case.id
        )
        scores.append(v.score or 0.0)
        unstable += bool(v.details.get("position_unstable"))
    if not scores:
        return na("no baseline outputs to compare")
    value = mean(scores) or 0.0
    return result(
        value, value >= float(ctx.p("min_win_rate", 0.5)), n=len(scores), position_unstable=unstable
    )
