"""Default suite behaviour (Plan.md §9 "Default suite behaviour")."""

from __future__ import annotations

from typing import Any, ClassVar

from evalkit.env.simulator import simulate
from evalkit.planner import load_suite_cases
from evalkit.schema import Case, CaseRun, Observation, ResultRow, ToolCall
from evalkit.suites.context import SuiteContext
from evalkit.util import normalize_text


class BaseSuite:
    """Load cases → prepare → run each case × repeats → metrics → panels.

    Subclasses override only what differs (at most one level of inheritance).
    """

    name: ClassVar[str] = ""
    default_metrics: ClassVar[tuple[str, ...]] = ()

    def load_cases(self, sctx: SuiteContext) -> list[Case]:
        return load_suite_cases(sctx.project, sctx.name, sctx.sc)

    def prepare(self, cases: list[Case], sctx: SuiteContext) -> list[Case]:
        return cases

    def skip_reason(self, case: Case, sctx: SuiteContext) -> str | None:
        """Skip every metric of a case without calling the target (e.g. no media)."""
        return None

    def case_metric_ids(self, case: Case, ids: list[str], sctx: SuiteContext) -> list[str]:
        """Case-scope metric ids to compute for this case (suites may add base metrics)."""
        return ids

    def execute(self, case: Case, repeat: int, sctx: SuiteContext) -> Observation:
        if case.turns:
            return run_scripted_turns(case, sctx)
        if case.persona and case.hidden_goal and case.input is None:
            return run_simulated(case, sctx)
        return sctx.call(case)

    def panels(
        self, rows: list[ResultRow], runs: list[CaseRun], sctx: SuiteContext
    ) -> list[dict[str, Any]]:
        return []


def merge_observations(
    observations: list[Observation], messages: list[dict[str, str]]
) -> Observation:
    """Fold per-turn observations into one transcript-level Observation."""
    if not observations:
        return Observation(messages=messages)
    last = observations[-1]
    tool_calls: list[ToolCall] = [tc for o in observations for tc in o.tool_calls]
    return last.model_copy(
        update={
            "messages": messages,
            "tool_calls": tool_calls,
            "retrieved": [c for o in observations for c in o.retrieved],
            "events": [e for o in observations for e in o.events],
            "spans": [s for o in observations for s in o.spans],
            "tokens_in": sum(o.tokens_in for o in observations),
            "tokens_out": sum(o.tokens_out for o in observations),
            "latency_s": sum(o.latency_s for o in observations),
            "error": next((o.error for o in observations if o.error), None),
        }
    )


def turn_checks(expect: dict[str, Any], text: str) -> dict[str, Any]:
    """Deterministic per-turn `expect` checks (must_include / must_not_include / regex)."""
    import re

    low = normalize_text(text)
    missing = [s for s in expect.get("must_include", []) if normalize_text(s) not in low]
    present = [s for s in expect.get("must_not_include", []) if normalize_text(s) in low]
    rx = expect.get("regex")
    rx_ok = True if not rx else bool(re.search(rx, text, re.I))
    return {
        "missing": missing,
        "forbidden": present,
        "regex_ok": rx_ok,
        "passed": not missing and not present and rx_ok,
        "criteria": list(expect.get("criteria", [])),
    }


def run_scripted_turns(case: Case, sctx: SuiteContext, **over: Any) -> Observation:
    """Each turn sends the full history; per-turn expect results go to raw['turns']."""
    messages: list[dict[str, str]] = list(over.pop("messages", []))
    observations: list[Observation] = []
    turns: list[dict[str, Any]] = []
    for i, turn in enumerate(case.turns or []):
        messages.append({"role": "user", "content": turn.user})
        turn_case = case.model_copy(update={"input": None})
        obs = sctx.call(turn_case, messages=list(messages), **over)
        observations.append(obs)
        messages.append({"role": "assistant", "content": obs.text})
        turns.append(
            {"turn": i, "user": turn.user, "reply": obs.text, **turn_checks(turn.expect, obs.text)}
        )
    merged = merge_observations(observations, messages)
    merged.raw = {"turns": turns}
    return merged


def run_simulated(case: Case, sctx: SuiteContext, **over: Any) -> Observation:
    def call_target(msgs: list[dict[str, str]]) -> Observation:
        return sctx.call(case.model_copy(update={"input": None}), messages=msgs, **over)

    res = simulate(
        persona=case.persona or "",
        hidden_goal=case.hidden_goal or "",
        max_turns=case.max_turns,
        llm=sctx.llm("simulator"),
        call_target=call_target,
        first_message=case.extra.get("first_message"),
        case_id=case.id,
    )
    merged = merge_observations(res.observations, res.messages)
    merged.raw = {"simulated": True, "user_turns": res.user_turns, "done": res.done}
    return merged
