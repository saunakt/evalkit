"""Multi-turn conversation metrics (Plan.md §9.4)."""

from __future__ import annotations

from typing import Any

from evalkit.judges.runner import inline_judge
from evalkit.metrics.common import contains_all, from_verdict, na, result, transcript
from evalkit.registry import metric
from evalkit.schema import Case, MetricResult, Observation
from evalkit.suites.context import MetricContext

CONV = ("conversation",)
CRITERIA = {
    "knowledge_retention": "PASS if the assistant remembers and correctly uses facts the user "
    "gave in earlier turns, and never asks for them again.",
    "reference_resolution": "PASS if the assistant correctly resolves references such as 'it', "
    "'that order' or 'the same one' to the right earlier entity.",
    "consistency": "PASS if the assistant never contradicts itself across turns.",
    "role_adherence": "PASS if the assistant stays in its support-agent role and policies for "
    "the whole conversation, even when the user pushes it elsewhere.",
    "conversation_completeness": "PASS if, by the end, every request the user made was "
    "addressed or explicitly handed off.",
    "recovery": "PASS if, after a misunderstanding, error or correction, the assistant "
    "acknowledges it and gets the conversation back on track.",
}
for _mid, _crit in CRITERIA.items():
    inline_judge(f"conv_{_mid}", _crit, metric=_mid)
inline_judge(
    "conv_turn_relevancy",
    "PASS if this assistant turn is relevant to the user turn "
    "before it and to the conversation so far.",
    metric="turn_relevancy",
)
inline_judge(
    "conv_memory",
    "PASS if the reply correctly recalls this fact from an earlier session: {fact}.",
    metric="long_term_memory",
)


def _conv_judge(metric_id: str, case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    if len(obs.messages) < 2:
        return na("no multi-turn transcript")
    secs = {"Transcript": transcript(obs)}
    if case.hidden_goal:
        secs["User goal"] = case.hidden_goal
    extra = [c for c in case.criteria] if case.criteria else []
    crit = CRITERIA[metric_id] + (f" Also: {'; '.join(extra)}" if extra else "")
    v = ctx.judge.binary(f"conv_{metric_id}", criterion=crit, sections=secs, case_id=case.id)
    return from_verdict(v)


def _register(metric_id: str) -> None:
    def fn(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
        return _conv_judge(metric_id, case, obs, ctx)

    fn.__doc__ = f"Transcript judge: {CRITERIA[metric_id]}"
    metric(metric_id, suites=CONV, kind="judge", engines=("native", "deepeval"))(fn)


for _mid in CRITERIA:
    _register(_mid)


@metric("turn_relevancy", suites=CONV, kind="judge", engines=("native", "deepeval"))
def turn_relevancy(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """Per assistant turn: relevance judge + the turn's deterministic `expect` checks."""
    turns = (obs.raw or {}).get("turns") if isinstance(obs.raw, dict) else None
    if not turns:
        return na("no scripted turns")
    rows: list[dict[str, Any]] = []
    for t in turns:
        v = ctx.judge.binary(
            "conv_turn_relevancy",
            sections={"User turn": t["user"], "Assistant turn": t["reply"]},
            case_id=case.id,
        )
        rows.append(
            {
                "turn": t["turn"],
                "relevant": bool(v.passed),
                "checks": t["passed"],
                "missing": t["missing"],
                "forbidden": t["forbidden"],
            }
        )
    ok_turns = sum(1 for r in rows if r["relevant"] and r["checks"])
    return result(ok_turns / len(rows), ok_turns == len(rows), turns=rows)


@metric(
    "turns_to_resolution",
    suites=CONV,
    kind="deterministic",
    direction="lower",
    unit="count",
    needs=("hidden_goal",),
)
def turns_to_resolution(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """User turns until the simulated user said <DONE> with the goal met."""
    raw = obs.raw if isinstance(obs.raw, dict) else {}
    if not raw.get("simulated"):
        return na("not a simulated conversation")
    turns = int(raw.get("user_turns", 0))
    limit = int(case.extra.get("max_turns_to_resolution", case.max_turns))
    done = bool(raw.get("done"))
    return result(float(turns), done and turns <= limit, float(limit), done=done)


FILLER = ("Thanks.", "Okay, noted.", "Sure, one moment.", "Got it.")


@metric("context_overflow", suites=CONV, kind="deterministic", needs=("turns",))
def context_overflow(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """Pad history with params.pad_turns filler turns; final checks must still pass, no error."""
    pad = int(ctx.p("pad_turns", 40))
    history: list[dict[str, str]] = []
    for i in range(pad):
        history += [
            {"role": "user", "content": FILLER[i % len(FILLER)]},
            {"role": "assistant", "content": "Happy to help."},
        ]
    from evalkit.suites.base import run_scripted_turns

    ctx.suite.reset(ctx.seed)
    padded = run_scripted_turns(case, ctx.suite, messages=history)
    last = (case.turns or [])[-1].expect if case.turns else {}
    missing = contains_all(padded.text, list(last.get("must_include", [])))
    ok = padded.error is None and not missing
    return result(1.0 if ok else 0.0, ok, pad_turns=pad, missing=missing, error=padded.error)


@metric("long_term_memory", suites=CONV, kind="judge", needs=("extra.memory",))
def long_term_memory(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """Session 1 states a fact; a new session (new session_id) asks for it; judge checks recall."""
    spec = dict(case.extra["memory"])
    ctx.suite.reset(ctx.seed)
    ctx.suite.call(
        case.model_copy(update={"input": spec["tell"]}),
        session_id="s1",
        params={"customer_id": spec.get("customer_id", "c-memory")},
    )
    later = ctx.suite.call(
        case.model_copy(update={"input": spec["ask"]}),
        session_id="s2",
        params={"customer_id": spec.get("customer_id", "c-memory")},
    )
    if later.error and "session" in later.error.lower():
        return na("target does not support sessions")
    crit = f"PASS if the reply correctly recalls this fact from an earlier session: {spec['fact']}"
    v = ctx.judge.binary(
        "conv_memory",
        criterion=crit,
        sections={"Question": spec["ask"], "Reply": later.text},
        case_id=case.id,
    )
    found = not contains_all(later.text, [str(spec["fact"])])
    return result(
        1.0 if (v.passed and found) else 0.0,
        bool(v.passed and found),
        reply=later.text[:200],
        fact_present=found,
    )
