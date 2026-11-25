"""Agent trajectory, end-state and policy metrics (Plan.md §9.6)."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable
from typing import Any

from evalkit.judges.runner import inline_judge
from evalkit.metrics.common import boolean, case_sections, from_verdict, na, result, trajectory
from evalkit.metrics.prompt_checks import args_subset
from evalkit.registry import metric
from evalkit.schema import Case, MetricResult, Observation, ToolCall
from evalkit.suites.context import MetricContext

AGENT = ("agent",)
TRACEBACK = re.compile(r"Traceback|Exception|at .*\(.*\)")
inline_judge(
    "plan_quality",
    "Rate the quality of the agent's plan and tool use for the "
    "user's request: right tools, sensible order, no wasted or risky steps.",
    style="rubric",
    rubric={1: "wrong or unsafe", 3: "works but wasteful", 5: "minimal, correct and safe"},
    metric="plan_quality",
)
inline_judge(
    "no_false_success",
    "PASS if the reply does NOT claim the action succeeded when "
    "the tools failed, and tells the user what happened. FAIL if it claims success.",
    metric="error_recovery",
)


def names(obs: Observation) -> list[str]:
    return [tc.name for tc in obs.tool_calls]


def executed(obs: Observation) -> list[ToolCall]:
    """Tool calls that actually ran (no error)."""
    return [tc for tc in obs.tool_calls if tc.error is None]


def is_subsequence(needle: list[str], hay: list[str]) -> bool:
    it = iter(hay)
    return all(any(n == h for h in it) for n in needle)


TRAJECTORY_MODES: dict[str, Callable[[list[str], list[str]], bool]] = {
    "exact": lambda exp, act: exp == act,
    "in_order": is_subsequence,
    "any_order": lambda exp, act: Counter(exp) == Counter(act),
    "subset": lambda exp, act: set(exp) <= set(act),
}


def _trajectory(mode: str, case: Case, obs: Observation) -> MetricResult:
    expected = [t.name for t in case.expect_trajectory]
    actual = names(obs)
    return boolean(
        TRAJECTORY_MODES[mode](expected, actual), mode=mode, expected=expected, actual=actual
    )


def _register_trajectory(mode: str) -> None:
    def fn(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
        return _trajectory(mode, case, obs)

    fn.__doc__ = f"Tool-name sequence vs expect_trajectory ({mode.replace('_', ' ')})."
    metric(f"trajectory_{mode}", suites=AGENT, kind="deterministic", needs=("expect_trajectory",))(
        fn
    )


for _mode in TRAJECTORY_MODES:
    _register_trajectory(_mode)


@metric(
    "trajectory", suites=AGENT + ("skills",), kind="deterministic", needs=("expect_trajectory",)
)
def trajectory_metric(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """Trajectory check using the case's trajectory_mode."""
    return _trajectory(case.trajectory_mode, case, obs)


@metric("single_step_choice", suites=AGENT, kind="deterministic", needs=("expect_trajectory",))
def single_step_choice(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """Call with max_steps=1: first call name/args vs expect_trajectory[0]."""
    first = ctx.suite.recall(case, max_steps=1)
    want = case.expect_trajectory[0]
    if not first.tool_calls:
        return boolean(False, reason="no tool call", expected=want.name)
    call = first.tool_calls[0]
    ok = call.name == want.name and args_subset(want.args, call.args)
    return boolean(ok, called=call.name, args=call.args, expected=want.name)


@metric("tool_selection", suites=AGENT, kind="deterministic", needs=("expect_tools|forbid_tools",))
def tool_selection(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """Pass = all expect_tools called and no forbid_tools; value = recall of expected tools."""
    called = set(names(obs))
    expected = set(case.expect_tools)
    hit = expected & called
    forbidden = sorted(set(case.forbid_tools) & called)
    recall = len(hit) / len(expected) if expected else 1.0
    precision = len(hit) / len(called) if called else (1.0 if not expected else 0.0)
    return result(
        recall,
        hit == expected and not forbidden,
        missing=sorted(expected - called),
        forbidden=forbidden,
        precision=precision,
    )


@metric("argument_accuracy", suites=AGENT, kind="deterministic", needs=("expect_trajectory",))
def argument_accuracy(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """For each expectation with args: first call with that name has subset-equal args."""
    checks = []
    for exp in case.expect_trajectory:
        if not exp.args:
            continue
        call = next((tc for tc in obs.tool_calls if tc.name == exp.name), None)
        ok = call is not None and args_subset(exp.args, call.args, float(ctx.p("tolerance", 1e-6)))
        checks.append(
            {"tool": exp.name, "ok": ok, "expected": exp.args, "got": call.args if call else None}
        )
    if not checks:
        return na("no expected args")
    good = sum(1 for c in checks if c["ok"])
    return result(good / len(checks), good == len(checks), checks=checks)


@metric("forbidden_tools", suites=AGENT, kind="deterministic", needs=("forbid_tools",))
def forbidden_tools(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """No forbidden tool was called."""
    bad = sorted(set(case.forbid_tools) & set(names(obs)))
    return boolean(not bad, forbidden_called=bad)


@metric("step_efficiency", suites=AGENT, kind="deterministic", needs=("expect_trajectory",))
def step_efficiency(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """min(1, expected / actual steps) (or extra.min_steps); fail if max_steps exceeded."""
    expected = int(case.extra.get("min_steps", len(case.expect_trajectory)))
    actual = len(obs.tool_calls)
    value = 1.0 if actual == 0 and expected == 0 else min(1.0, expected / max(1, actual))
    exceeded = case.max_steps is not None and actual > case.max_steps
    minimum = float(ctx.p("min_step_efficiency", 0.7))
    return result(
        value,
        value >= minimum and not exceeded,
        minimum,
        steps=actual,
        expected=expected,
        exceeded=exceeded,
    )


def deep_subset(expected: Any, actual: Any, path: str = "") -> list[str]:
    """Paths where `expected` is not contained in `actual`."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [path or "/"]
        out: list[str] = []
        for k, v in expected.items():
            out += deep_subset(v, actual.get(k), f"{path}.{k}" if path else str(k))
        return out
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) < len(expected):
            return [path]
        return [
            p for i, v in enumerate(expected) for p in deep_subset(v, actual[i], f"{path}[{i}]")
        ]
    return [] if expected == actual else [f"{path}: expected {expected!r}, got {actual!r}"]


@metric(
    "end_state",
    suites=AGENT + ("coding_agent",),
    kind="deterministic",
    needs=("expect_state", "hook:state"),
)
def end_state(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """target.state() contains expect_state (deep subset)."""
    diffs = deep_subset(case.expect_state, obs.state)
    return boolean(not diffs, differences=diffs)


def task_success_result(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    parts: dict[str, bool] = {"no_error": obs.error is None}
    if case.expect_state:
        parts["end_state"] = not deep_subset(case.expect_state, obs.state)
    if case.forbid_tools:
        parts["no_forbidden_tools"] = not (set(case.forbid_tools) & set(names(obs)))
    details: dict[str, Any] = {}
    if case.criteria:
        from evalkit.metrics.judged import per_criterion

        crit = per_criterion(ctx.judge, "generic_binary", case.criteria, case, obs)
        parts["criteria"] = bool(crit.passed)
        details = {k: v for k, v in crit.details.items() if k in ("failed", "uncalibrated")}
    ok = all(parts.values())
    return boolean(ok, parts=parts, **details)


@metric("task_success", suites=AGENT + ("multimodal",), kind="judge")
def task_success(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """end_state (if given) AND criteria judge (if given) AND no forbidden tools AND no error."""
    return task_success_result(case, obs, ctx)


@metric("browser_task_success", suites=AGENT, kind="judge")
def browser_task_success(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """Alias of task_success for browser/computer-use targets (state hook checks the page)."""
    if "browser" not in case.tags and not case.extra.get("browser"):
        return na("not a browser task")
    return task_success_result(case, obs, ctx)


@metric("simulated_user_success", suites=AGENT, kind="judge", needs=("persona", "hidden_goal"))
def simulated_user_success(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """Simulated conversation (BaseSuite.execute), then task_success on the result."""
    res = task_success_result(case, obs, ctx)
    raw = obs.raw if isinstance(obs.raw, dict) else {}
    res.details["user_turns"] = raw.get("user_turns")
    return res


@metric("plan_quality", suites=AGENT, kind="judge")
def plan_quality(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """Rubric judge over the tool trajectory."""
    if not obs.tool_calls:
        return na("no tool calls")
    secs = {**case_sections(case, obs), "Tool trajectory": trajectory(obs)}
    return from_verdict(
        ctx.judge.rubric(
            "plan_quality",
            sections=secs,
            case_id=case.id,
            threshold=float(ctx.p("min_plan_quality", 0.5)),
        )
    )


@metric("error_recovery", suites=AGENT + ("performance",), kind="judge", needs=("faults",))
def error_recovery(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """With faults: no false success claim (judge), graceful message, end_state unchanged."""
    graceful = bool(obs.text.strip()) and not TRACEBACK.search(obs.text)
    v = ctx.judge.binary(
        "no_false_success",
        sections={**case_sections(case, obs), "Tool trajectory": trajectory(obs)},
        case_id=case.id,
    )
    state_ok = not deep_subset(case.expect_state, obs.state) if case.expect_state else True
    ok = graceful and bool(v.passed) and state_ok
    return boolean(
        ok,
        graceful=graceful,
        no_false_success=v.passed,
        state_unchanged=state_ok,
        reasoning=v.reasoning,
        **({"uncalibrated": True} if v.details.get("uncalibrated") else {}),
    )
