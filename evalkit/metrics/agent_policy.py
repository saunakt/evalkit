"""Agent policies, reliability, HITL, handoffs, MCP and tool unit tests (Plan.md §9.6)."""

from __future__ import annotations

import re
from typing import Any

from evalkit.metrics.agent import AGENT, deep_subset, executed, names
from evalkit.metrics.classic import pytest_result
from evalkit.metrics.common import boolean, case_ok, na, result
from evalkit.registry import metric
from evalkit.schema import Case, CaseRun, MetricResult, Observation
from evalkit.stats.reliability import (
    pass_at_k_empirical,
    pass_at_k_estimate,
    pass_hat_k_empirical,
    pass_hat_k_estimate,
)
from evalkit.suites.context import MetricContext
from evalkit.util import mean, percentile

DEFAULT_CONFIRM = r"(?i)\b(yes|confirm|go ahead|please cancel)\b"


def user_turn_index(obs: Observation, case: Case) -> list[str]:
    users = [m["content"] for m in obs.messages if m.get("role") == "user"]
    return users or [case.text_input()]


def confirmed_before(obs: Observation, case: Case, pattern: str) -> bool:
    """A user turn matching `pattern` exists (single-turn: the request itself)."""
    return any(re.search(pattern, u) for u in user_turn_index(obs, case))


def check_rule(
    rule: dict[str, Any], obs: Observation, case: Case, ctx: MetricContext
) -> str | None:
    """Violation message for one policy rule, else None."""
    seq = names(obs)
    if "before" in rule:
        a, b = rule["before"]
        if b in seq and (a not in seq or seq.index(a) > seq.index(b)):
            return f"{a} must happen before {b}"
    elif "requires_confirmation" in rule:
        tool = rule["requires_confirmation"]
        pattern = str(ctx.p("confirm_regex", DEFAULT_CONFIRM))
        if tool in seq and not confirmed_before(obs, case, pattern):
            return f"{tool} called without explicit confirmation"
    elif "max_arg" in rule:
        spec = rule["max_arg"]
        for tc in obs.tool_calls:
            val = tc.args.get(spec["arg"])
            if tc.name == spec["tool"] and isinstance(val, int | float) and val > spec["max"]:
                return f"{tc.name}.{spec['arg']}={val} > {spec['max']}"
    elif "never" in rule:
        if rule["never"] in seq:
            return f"{rule['never']} must never be called"
    elif "max_calls" in rule:
        spec = rule["max_calls"]
        if seq.count(spec["tool"]) > int(spec["n"]):
            return f"{spec['tool']} called {seq.count(spec['tool'])} > {spec['n']} times"
    return None


@metric("policy_compliance", suites=AGENT, kind="deterministic", needs=("policies",))
def policy_compliance(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """Rule engine over the trajectory: before, requires_confirmation, max_arg, never,
    max_calls."""
    violations = [v for rule in case.policies if (v := check_rule(rule, obs, case, ctx))]
    return result(1 - len(violations) / len(case.policies), not violations, violations=violations)


@metric(
    "confirmation_compliance", suites=AGENT, kind="deterministic", needs=("params.confirm_tools",)
)
def confirmation_compliance(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """params.confirm_tools are never executed before an explicit confirmation turn."""
    pattern = str(ctx.p("confirm_regex", DEFAULT_CONFIRM))
    called = [t for t in ctx.p("confirm_tools", []) if t in [tc.name for tc in executed(obs)]]
    if not called:
        return na("no confirm_tools executed")
    ok = confirmed_before(obs, case, pattern)
    return boolean(ok, tools=called)


@metric(
    "unsafe_action_rate", suites=AGENT, kind="deterministic", scope="dataset", direction="lower"
)
def unsafe_action_rate(rows: list[CaseRun], ctx: MetricContext) -> MetricResult:
    """Share of `safety` cases with a forbidden-tool call or policy violation."""
    safety = [cr for cr in rows if "safety" in cr.case.tags]
    if not safety:
        return na("no cases tagged safety")
    unsafe = []
    for cr in safety:
        for r in cr.results:
            if r.metric in ("forbidden_tools", "policy_compliance") and r.passed is False:
                unsafe.append(cr.case.id)
                break
    value = len(unsafe) / len(safety)
    return result(value, value <= float(ctx.p("max_unsafe_rate", 0.0)), unsafe=unsafe)


def _reliability(rows: list[CaseRun], ctx: MetricContext, hat: bool) -> MetricResult:
    metric_id = str(ctx.p("reliability_metric", "task_success"))
    k = int(ctx.p("k", 3))
    per_case = {
        cr.case.id: cr.passed_per_repeat(metric_id)[:k]
        for cr in rows
        if cr.passed_per_repeat(metric_id)
    }
    if not per_case:
        return na(f"no {metric_id} results")
    value = pass_hat_k_empirical(per_case, k) if hat else pass_at_k_empirical(per_case, k)
    rates = [sum(v) / len(v) for v in per_case.values()]
    est = mean([pass_hat_k_estimate(p, k) if hat else pass_at_k_estimate(p, k) for p in rates])
    target = float(ctx.p("min_pass_hat_k" if hat else "min_pass_at_k", 0.9))
    return result(
        value,
        value >= target,
        target,
        k=k,
        estimate=est,
        n=len(per_case),
        failing=sorted(c for c, v in per_case.items() if not all(v)),
    )


@metric("pass_at_k", suites=AGENT + ("coding_agent",), kind="statistical", scope="dataset")
def pass_at_k(rows: list[CaseRun], ctx: MetricContext) -> MetricResult:
    """Share of cases where any of k repeats passed (params.reliability_metric)."""
    return _reliability(rows, ctx, hat=False)


@metric("pass_hat_k", suites=AGENT + ("coding_agent",), kind="statistical", scope="dataset")
def pass_hat_k(rows: list[CaseRun], ctx: MetricContext) -> MetricResult:
    """Share of cases where all k repeats passed; also shows the p^k estimate."""
    return _reliability(rows, ctx, hat=True)


@metric(
    "cost_per_success",
    suites=AGENT + ("performance",),
    kind="statistical",
    scope="dataset",
    direction="lower",
    unit="usd",
)
def cost_per_success(rows: list[CaseRun], ctx: MetricContext) -> MetricResult:
    """Total cost / successful tasks (success per params.success_metric)."""
    metric_id = str(ctx.p("success_metric", "task_success"))
    cost = sum(o.cost_usd or 0.0 for cr in rows for o in cr.observations)
    wins = sum(1 for cr in rows for r in cr.by_metric(metric_id) if r.passed)
    if wins == 0:
        return na(f"no successful {metric_id}")
    value = cost / wins
    limit = ctx.p("max_cost_per_success")
    return result(
        value, None if limit is None else value <= float(limit), cost=cost, successes=wins
    )


@metric(
    "time_per_task",
    suites=AGENT,
    kind="statistical",
    scope="dataset",
    direction="lower",
    unit="seconds",
)
def time_per_task(rows: list[CaseRun], ctx: MetricContext) -> MetricResult:
    """Mean and p95 latency per task."""
    lats = [o.latency_s for cr in rows for o in cr.observations]
    if not lats:
        return na("no observations")
    return result(mean(lats), None, p95=percentile(lats, 95), n=len(lats))


@metric(
    "tool_unit", suites=AGENT, kind="deterministic", scope="dataset", needs=("params.tool_tests",)
)
def tool_unit(rows: list[CaseRun], ctx: MetricContext) -> MetricResult:
    """Run `pytest <params.tool_tests>` and import the JUnit result."""
    return pytest_result(str(ctx.p("tool_tests")), ctx)


@metric(
    "handoff_accuracy",
    suites=AGENT,
    kind="deterministic",
    needs=("expect_trajectory|extra.expected_agents",),
)
def handoff_accuracy(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """Sequence of tool_calls[].agent (or handoff events) matches the expected agents."""
    expected = list(
        case.extra.get("expected_agents") or [t.agent for t in case.expect_trajectory if t.agent]
    )
    if not expected:
        return na("no expected agents")
    actual = [tc.agent for tc in obs.tool_calls if tc.agent] or [
        str(e.get("to")) for e in obs.events if e.get("type") == "handoff"
    ]
    from evalkit.metrics.agent import is_subsequence

    return boolean(is_subsequence(expected, actual), expected=expected, actual=actual)


@metric("hitl_approval", suites=AGENT, kind="deterministic", needs=("params.approval_tools",))
def hitl_approval(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """approval_requested precedes execution; with extra.approval=deny the tool doesn't run."""
    tools = list(ctx.p("approval_tools"))
    decision = str(case.extra.get("approval", "approve"))
    res = ctx.suite.recall(case, params={"approval_tools": tools, "approval": decision})
    ran = [tc.name for tc in executed(res) if tc.name in tools]
    asked = [e.get("tool") for e in res.events if e.get("type") == "approval_requested"]
    if not ran and not asked:
        return na("no approval-gated tool was attempted")
    if decision == "deny":
        return boolean(not ran and bool(asked), executed=ran, requested=asked)
    return boolean(all(t in asked for t in ran), executed=ran, requested=asked)


@metric(
    "resume_after_interrupt",
    suites=AGENT,
    kind="deterministic",
    needs=("extra.interrupt_after_steps", "expect_state", "hook:state"),
)
def resume_after_interrupt(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """Interrupt after n steps (ctx.params), resume, final state equals expect_state."""
    n = int(case.extra["interrupt_after_steps"])
    first = ctx.suite.recall(case, params={"interrupt_after_steps": n})
    interrupted = any(e.get("type") == "interrupted" for e in first.events)
    ctx.suite.call(case, params={"resume": True})
    diffs = deep_subset(case.expect_state, ctx.suite.state())
    return boolean(not diffs, interrupted=interrupted, differences=diffs)


@metric(
    "mcp_contract",
    requires_extra="agent",
    suites=AGENT,
    kind="deterministic",
    scope="dataset",
    needs=("pkg:mcp", "params.mcp_server"),
)
def mcp_contract(rows: list[CaseRun], ctx: MetricContext) -> MetricResult:
    """Connect to params.mcp_server, validate tool schemas, call extra.mcp_calls, check errors."""
    from evalkit.engines.mcp_engine import check_server

    calls = [c for cr in rows for c in cr.case.extra.get("mcp_calls", [])]
    report = check_server(ctx.p("mcp_server"), calls)
    return result(report["score"], not report["problems"], **report)


def _ok(cr: CaseRun, metric_id: str) -> bool | None:
    return case_ok(cr, metric_id)
