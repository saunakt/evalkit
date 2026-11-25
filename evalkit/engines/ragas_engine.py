"""Ragas metric implementations and test-set generation (extra `[ragas]`).

Checked against ragas 0.4.4: collection metrics (`ragas.metrics.collections.Faithfulness`)
take an `InstructorBaseRagasLLM` (abstract: generate(prompt, response_model), agenerate);
`metric.score(user_input=, response=, retrieved_contexts=)` returns an object with `.value`.
AnswerRelevancy also needs an embeddings model, which evalkit does not configure.
"""

from __future__ import annotations

from typing import Any

from evalkit.engines.judge_bridge import JudgeBridge, disable_telemetry
from evalkit.metrics.common import na, result
from evalkit.registry import engine, metric_impl
from evalkit.schema import Case, MetricResult, Observation
from evalkit.suites.context import MetricContext
from evalkit.util import optional_import


@engine("ragas", requires_extra="ragas")
class RagasEngine:
    """Engine marker: faithfulness (and test-set generation via `evalctl generate`)."""

    package = "ragas"


def judge_llm(ctx: MetricContext, case_id: str | None) -> Any:
    """An InstructorBaseRagasLLM whose calls go through evalkit's judge role."""
    disable_telemetry()
    base = optional_import("ragas.llms", "ragas").InstructorBaseRagasLLM
    bridge = JudgeBridge(ctx.judge.runtime.client("judge"), "ragas", case_id)

    class EvalkitRagasJudge(base):  # type: ignore[misc, valid-type]
        def generate(self, prompt: str, response_model: Any) -> Any:
            return bridge.generate(prompt, response_model)

        async def agenerate(self, prompt: str, response_model: Any) -> Any:
            return bridge.generate(prompt, response_model)

    return EvalkitRagasJudge()


@metric_impl("faithfulness", engine="ragas")
def faithfulness(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """Ragas Faithfulness (collections API) with the evalkit judge."""
    if not obs.retrieved:
        return na("no retrieved context")
    coll = optional_import("ragas.metrics.collections", "ragas")
    metric_obj = coll.Faithfulness(llm=judge_llm(ctx, case.id))
    res = metric_obj.score(
        user_input=case.text_input(),
        response=obs.text,
        retrieved_contexts=[c.text for c in obs.retrieved],
    )
    value = float(getattr(res, "value", res) or 0.0)
    thr = float(ctx.p("min_faithfulness", 0.8))
    return result(value, value >= thr, thr, reason=str(getattr(res, "reason", "") or ""))


@metric_impl("answer_relevancy", engine="ragas")
def answer_relevancy(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """Ragas AnswerRelevancy needs an embeddings model; reported as not tested."""
    optional_import("ragas")
    return MetricResult(
        metric="",
        skipped="ragas answer_relevancy needs an embeddings model "
        "(not configured); use the native or deepeval engine",
    )
