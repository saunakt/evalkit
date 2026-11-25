"""DeepEval metric implementations (extra `[deepeval]`), judged by evalkit's judge role.

Checked against deepeval 4.3.5: custom models subclass `deepeval.models.DeepEvalBaseLLM`
(abstract: load_model, generate, a_generate, get_model_name); metrics take `model=` and
`async_mode=`; `metric.measure(test_case)` sets `.score` / `.reason`.
"""

from __future__ import annotations

from typing import Any

from evalkit.engines.judge_bridge import JudgeBridge, disable_telemetry
from evalkit.metrics.common import result, transcript
from evalkit.registry import engine, metric_impl
from evalkit.schema import Case, MetricResult, Observation
from evalkit.suites.context import MetricContext
from evalkit.util import optional_import


@engine("deepeval", requires_extra="deepeval")
class DeepEvalEngine:
    """Engine marker: faithfulness, answer_relevancy, g_eval and conversational metrics."""

    package = "deepeval"


def judge_model(ctx: MetricContext, case_id: str | None) -> Any:
    """A DeepEvalBaseLLM whose calls go through evalkit's judge role."""
    disable_telemetry()
    base = optional_import("deepeval.models", "deepeval").DeepEvalBaseLLM
    bridge = JudgeBridge(ctx.judge.runtime.client("judge"), "deepeval", case_id)

    class EvalkitJudge(base):  # type: ignore[misc, valid-type]
        def __init__(self) -> None:
            super().__init__(model=f"evalkit-judge:{bridge.client.model}")

        def load_model(self) -> Any:
            return bridge

        def generate(self, prompt: str, schema: Any = None, **_: Any) -> Any:
            return bridge.generate(prompt, schema)

        async def a_generate(self, prompt: str, schema: Any = None, **_: Any) -> Any:
            return bridge.generate(prompt, schema)

        def get_model_name(self) -> str:
            return f"evalkit-judge:{bridge.client.model}"

    return EvalkitJudge()


def _measure(metric_obj: Any, test_case: Any, threshold: float) -> MetricResult:
    metric_obj.measure(test_case)
    score = float(metric_obj.score or 0.0)
    return result(score, score >= threshold, threshold, reason=getattr(metric_obj, "reason", ""))


def _single(case: Case, obs: Observation) -> Any:
    tc = optional_import("deepeval.test_case", "deepeval")
    return tc.LLMTestCase(
        input=case.text_input(),
        actual_output=obs.text,
        expected_output=case.reference,
        retrieval_context=[c.text for c in obs.retrieved] or None,
    )


@metric_impl("faithfulness", engine="deepeval")
def faithfulness(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """DeepEval FaithfulnessMetric with the evalkit judge."""
    m = optional_import("deepeval.metrics", "deepeval")
    thr = float(ctx.p("min_faithfulness", 0.8))
    obj = m.FaithfulnessMetric(threshold=thr, model=judge_model(ctx, case.id), async_mode=False)
    return _measure(obj, _single(case, obs), thr)


@metric_impl("answer_relevancy", engine="deepeval")
def answer_relevancy(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """DeepEval AnswerRelevancyMetric with the evalkit judge."""
    m = optional_import("deepeval.metrics", "deepeval")
    obj = m.AnswerRelevancyMetric(threshold=0.5, model=judge_model(ctx, case.id), async_mode=False)
    return _measure(obj, _single(case, obs), 0.5)


@metric_impl("g_eval", engine="deepeval")
def g_eval(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """DeepEval GEval over the case criteria."""
    m = optional_import("deepeval.metrics", "deepeval")
    tc = optional_import("deepeval.test_case", "deepeval")
    params = [tc.SingleTurnParams.INPUT, tc.SingleTurnParams.ACTUAL_OUTPUT]
    thr = float(ctx.p("min_score", 0.5))
    obj = m.GEval(
        name="evalkit_g_eval",
        criteria="; ".join(case.criteria),
        evaluation_params=params,
        model=judge_model(ctx, case.id),
        threshold=thr,
        async_mode=False,
    )
    return _measure(obj, _single(case, obs), thr)


CONVERSATIONAL = {
    "knowledge_retention": "KnowledgeRetentionMetric",
    "role_adherence": "RoleAdherenceMetric",
    "conversation_completeness": "ConversationCompletenessMetric",
    "turn_relevancy": "TurnRelevancyMetric",
}


def _conversational(metric_id: str, cls_name: str) -> None:
    def fn(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
        m = optional_import("deepeval.metrics", "deepeval")
        tc = optional_import("deepeval.test_case", "deepeval")
        turns = [tc.Turn(role=msg["role"], content=msg["content"]) for msg in obs.messages]
        conv = tc.ConversationalTestCase(
            turns=turns, chatbot_role=str(ctx.p("chatbot_role", "ShopEase support assistant"))
        )
        obj = getattr(m, cls_name)(threshold=0.5, model=judge_model(ctx, case.id), async_mode=False)
        res = _measure(obj, conv, 0.5)
        res.details["transcript"] = transcript(obs)[:500]
        return res

    fn.__doc__ = f"DeepEval {cls_name} with the evalkit judge."
    metric_impl(metric_id, engine="deepeval")(fn)


for _mid, _cls in CONVERSATIONAL.items():
    _conversational(_mid, _cls)
