"""Ranked-retrieval metrics over `obs.retrieved` vs `relevant_ids` (Plan.md §9.5)."""

from __future__ import annotations

import math

from evalkit.metrics.common import result, scored
from evalkit.registry import metric
from evalkit.schema import Case, MetricResult, Observation
from evalkit.suites.context import MetricContext

RET_SUITES = ("rag", "multilingual")
RELEVANT = ("relevant_ids|relevance",)


def top_k(obs: Observation, ctx: MetricContext) -> list[str]:
    k = int(ctx.p("k", 5))
    return [c.id for c in obs.retrieved[:k]]


def relevant(case: Case) -> set[str]:
    return set(case.relevant_ids) or {d for d, g in case.relevance.items() if g > 0}


def recall_at_k_value(case: Case, ids: list[str]) -> float:
    rel = relevant(case)
    return len(rel & set(ids)) / len(rel) if rel else 0.0


@metric("recall_at_k", suites=RET_SUITES, kind="deterministic", needs=RELEVANT)
def recall_at_k(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """|relevant ∩ top-k| / |relevant|."""
    ids = top_k(obs, ctx)
    value = recall_at_k_value(case, ids)
    return scored(
        value,
        float(ctx.p("min_recall", 1.0)),
        retrieved=ids,
        missing=sorted(relevant(case) - set(ids)),
    )


@metric("hit_rate_at_k", suites=RET_SUITES, kind="deterministic", needs=RELEVANT)
def hit_rate_at_k(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """1 if any relevant id is in the top-k."""
    hit = bool(relevant(case) & set(top_k(obs, ctx)))
    return result(1.0 if hit else 0.0, hit)


@metric("precision_at_k", suites=RET_SUITES, kind="deterministic", needs=RELEVANT)
def precision_at_k(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """|relevant ∩ top-k| / k (informational)."""
    k = int(ctx.p("k", 5))
    return result(len(relevant(case) & set(top_k(obs, ctx))) / k, None, k=k)


@metric("mrr", suites=RET_SUITES, kind="deterministic", needs=RELEVANT)
def mrr(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """1 / rank of the first relevant id (0 if none); dataset mean is reported."""
    rel = relevant(case)
    for rank, cid in enumerate(top_k(obs, ctx), start=1):
        if cid in rel:
            return result(1.0 / rank, None, rank=rank)
    return result(0.0, None, rank=None)


@metric("ndcg", suites=RET_SUITES, kind="deterministic", needs=RELEVANT)
def ndcg(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """Standard nDCG@k with graded `relevance` (binary from relevant_ids if absent)."""
    grades = dict(case.relevance) or {d: 1 for d in case.relevant_ids}
    ids = top_k(obs, ctx)
    dcg = sum((2 ** grades.get(cid, 0) - 1) / math.log2(i + 2) for i, cid in enumerate(ids))
    ideal = sorted(grades.values(), reverse=True)[: len(ids) or int(ctx.p("k", 5))]
    idcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return result(dcg / idcg if idcg else 0.0, None)
