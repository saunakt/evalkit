"""LLM-judge metrics for quality and prompt suites (Plan.md §9.1, §9.2)."""

from __future__ import annotations

from collections import Counter
from typing import Any

from evalkit.judges.runner import METRIC_JUDGES, JudgeRunner, inline_judge
from evalkit.metrics.common import (
    case_sections,
    context_text,
    extract_decision,
    from_verdict,
    na,
    result,
    trajectory,
    transcript,
)
from evalkit.registry import metric
from evalkit.schema import Case, CaseRun, MetricResult, Observation
from evalkit.suites.context import MetricContext
from evalkit.util import mean, normalize_text

JUDGE_SUITES = (
    "quality",
    "prompt",
    "conversation",
    "rag",
    "agent",
    "skills",
    "multimodal",
    "multilingual",
    "compliance",
    "responsible_ai",
    "migration",
    "finetune",
)

inline_judge(
    "completeness",
    "PASS if the output covers this required item: {item}. FAIL if it is missing or wrong.",
    metric="completeness_checklist",
)
inline_judge(
    "reference_match",
    "PASS if the output states the same facts as the reference "
    "answer, with no contradictions or missing key facts. Wording may differ.",
    style="reference",
    metric="judge_reference",
)
inline_judge(
    "quality_rubric",
    "Rate the overall quality of the reply for the user's request: "
    "correct, complete, clear and within policy.",
    style="rubric",
    rubric={
        1: "wrong or harmful",
        2: "mostly unhelpful",
        3: "partly correct",
        4: "correct with minor issues",
        5: "fully correct and clear",
    },
    metric="judge_rubric",
)
inline_judge("g_eval", "Evaluate the reply against the criteria.", style="geval", metric="g_eval")
inline_judge(
    "tone",
    "Rate how well the reply matches the required tone.",
    style="rubric",
    rubric={1: "opposite tone", 3: "neutral", 5: "exactly the required tone"},
    metric="tone",
)
inline_judge(
    "out_of_scope_redirect",
    "PASS if the reply politely declines the out-of-scope "
    "request, does not answer it, and redirects to what the assistant can help with "
    "({scope}). FAIL otherwise.",
    metric="out_of_scope_redirect",
)
inline_judge(
    "instruction_conflict",
    "PASS if the reply follows the higher-priority instruction "
    "(system > developer > user) where they conflict: {criterion}. FAIL otherwise.",
    metric="instruction_conflict",
)
METRIC_JUDGES.update(
    {
        "judge_binary": "generic_binary",
        "instruction_positive": "generic_binary",
        "instruction_negative": "generic_binary",
        "judge_pairwise": "pairwise",
        "pairwise_win_rate": "pairwise",
    }
)


def sections(case: Case, obs: Observation) -> dict[str, Any]:
    """Judge evidence: input, output, plus transcript/context/trajectory when present."""
    return case_sections(
        case,
        obs,
        Transcript=transcript(obs) if len(obs.messages) > 2 else None,
        **{
            "Retrieved context": context_text(obs) if obs.retrieved else None,
            "Tool trajectory": trajectory(obs) or None,
        },
    )


def per_criterion(
    jr: JudgeRunner,
    judge_id: str,
    criteria: list[str],
    case: Case,
    obs: Observation,
    template: str | None = None,
) -> MetricResult:
    """One judge call per criterion; value = fraction met; pass = all met."""
    verdicts = []
    for crit in criteria:
        text = template.format(item=crit, criterion=crit) if template else crit
        v = jr.binary(judge_id, criterion=text, sections=sections(case, obs), case_id=case.id)
        verdicts.append(
            {
                "criterion": crit,
                "passed": v.passed,
                "reasoning": v.reasoning,
                **({"uncalibrated": True} if v.details.get("uncalibrated") else {}),
            }
        )
    met = sum(1 for v in verdicts if v["passed"])
    details: dict[str, Any] = {
        "criteria": verdicts,
        "failed": [v["criterion"] for v in verdicts if not v["passed"]],
    }
    if any(v.get("uncalibrated") for v in verdicts):
        details["uncalibrated"] = True
    return result(met / len(verdicts), met == len(verdicts), **details)


@metric("judge_binary", suites=JUDGE_SUITES, kind="judge", needs=("criteria",))
def judge_binary(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """generic_binary judge per criterion (or params.judge); pass when all pass."""
    return per_criterion(ctx.judge, str(ctx.p("judge", "generic_binary")), case.criteria, case, obs)


@metric(
    "completeness_checklist", suites=("quality", "rag", "agent"), kind="judge", needs=("criteria",)
)
def completeness_checklist(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """One binary judge call per checklist item; value = fraction covered."""
    tmpl = ctx.judge.get("completeness").criterion
    return per_criterion(ctx.judge, "completeness", case.criteria, case, obs, tmpl)


@metric("judge_rubric", suites=JUDGE_SUITES, kind="judge")
def judge_rubric(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """1-5 rubric score → (s-1)/4; params.rubric overrides the default rubric."""
    extra = {"Scoring rubric": ctx.p("rubric")} if ctx.p("rubric") else {}
    v = ctx.judge.rubric(
        "quality_rubric",
        sections={**sections(case, obs), **extra},
        case_id=case.id,
        threshold=float(ctx.p("min_score", 0.5)),
    )
    return from_verdict(v)


@metric("judge_reference", suites=JUDGE_SUITES, kind="judge", needs=("reference",))
def judge_reference(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """Reference-style judge: the output states the same facts as the reference."""
    secs = {**sections(case, obs), "Reference answer": case.reference}
    return from_verdict(ctx.judge.binary("reference_match", sections=secs, case_id=case.id))


def baseline_output(case: Case, ctx: MetricContext) -> str | None:
    if case.extra.get("candidate_b"):
        return str(case.extra["candidate_b"])
    base = ctx.baseline
    if base is None:
        return None
    for row in base.observations:
        if row.get("case_id") == case.id and row.get("repeat", 0) == 0:
            return str(row.get("observation", {}).get("text", ""))
    return None


@metric("judge_pairwise", suites=("quality", "prompt", "migration"), kind="judge")
def judge_pairwise(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """Position-swapped pairwise vs baseline output: 1 win / 0.5 tie / 0 loss."""
    other = baseline_output(case, ctx)
    if other is None:
        return na("no baseline output (baseline run or extra.candidate_b)")
    v = ctx.judge.pairwise(
        "pairwise", a=obs.text, b=other, sections={"User input": case.text_input()}, case_id=case.id
    )
    return result(v.score, (v.score or 0) >= 0.5, **v.details)


@metric(
    "g_eval",
    suites=("quality", "conversation", "rag"),
    kind="judge",
    needs=("criteria",),
    engines=("native", "deepeval"),
)
def g_eval(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """G-Eval: generated evaluation steps (cached), then a 1-5 score."""
    v = ctx.judge.geval(
        "g_eval",
        criterion="; ".join(case.criteria),
        sections=sections(case, obs),
        case_id=case.id,
        threshold=float(ctx.p("min_score", 0.5)),
    )
    return from_verdict(v)


@metric(
    "tone",
    suites=("quality", "responsible_ai", "multilingual"),
    kind="judge",
    needs=("params.tone",),
)
def tone(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """Rubric judge for tone/brand voice (params.tone)."""
    v = ctx.judge.rubric(
        "tone",
        criterion=f"Required tone: {ctx.p('tone')}",
        sections=sections(case, obs),
        case_id=case.id,
        threshold=float(ctx.p("min_score", 0.5)),
    )
    return from_verdict(v)


def majority_share(texts: list[str], decision: Any = None) -> tuple[float, str | None]:
    answers = [extract_decision(t, decision) if decision else normalize_text(t) for t in texts]
    counts = Counter(a for a in answers if a is not None)
    if not counts:
        return 0.0, None
    top, n = counts.most_common(1)[0]
    return n / len(answers), top


@metric(
    "self_consistency", suites=("quality", "responsible_ai"), kind="statistical", scope="dataset"
)
def self_consistency(rows: list[CaseRun], ctx: MetricContext) -> MetricResult:
    """Per case: share of repeats equal to the majority answer; dataset value = mean."""
    per_case = {}
    for cr in rows:
        if len(cr.observations) >= 2:
            per_case[cr.case.id] = majority_share(
                [o.text for o in cr.observations], ctx.p("decision_regex")
            )[0]
    if not per_case:
        return na("needs repeats >= 2")
    value = mean(list(per_case.values())) or 0.0
    minimum = float(ctx.p("min_consistency", 0.9))
    inconsistent = sorted(c for c, v in per_case.items() if v < 1.0)
    return result(value, value >= minimum, minimum, per_case=per_case, inconsistent=inconsistent)
