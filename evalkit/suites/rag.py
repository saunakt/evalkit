"""rag — retrieval-augmented generation: ingestion, retrieval and generation (ch. 17–18)."""

from __future__ import annotations

from typing import Any

from evalkit.metrics.rag import ingested
from evalkit.registry import suite
from evalkit.schema import Case, CaseRun, Observation, ResultRow
from evalkit.suites.base import BaseSuite
from evalkit.suites.context import MetricContext, SuiteContext

INGESTION = {"parse_fidelity", "chunk_integrity"}
RETRIEVAL = {
    "recall_at_k",
    "hit_rate_at_k",
    "precision_at_k",
    "mrr",
    "ndcg",
    "access_control",
    "freshness",
    "context_precision",
    "context_recall",
    "noise_sensitivity",
}


def stage(case: Case) -> str:
    return str(case.extra.get("stage", "qa"))


@suite("rag")
class RagSuite(BaseSuite):
    """Retrieval-augmented generation: ingestion, retrieval and generation (ch. 17–18)."""

    name = "rag"
    default_metrics = (
        "recall_at_k",
        "mrr",
        "ndcg",
        "faithfulness",
        "answer_relevancy",
        "citation_accuracy",
        "unanswerable_handling",
    )

    def prepare(self, cases: list[Case], sctx: SuiteContext) -> list[Case]:
        """Keep only cases whose stage is selected in params.stages."""
        stages = set(sctx.params.get("stages", ["ingestion", "retrieval", "generation"]))
        keep_qa = bool(stages & {"retrieval", "generation"})
        return [
            c
            for c in cases
            if (stage(c) == "ingestion" and "ingestion" in stages)
            or (stage(c) != "ingestion" and keep_qa)
        ]

    def execute(self, case: Case, repeat: int, sctx: SuiteContext) -> Observation:
        if stage(case) == "ingestion":
            mctx = MetricContext(sctx.params, sctx.judge, sctx)
            return Observation(retrieved=ingested(mctx), text="")
        return super().execute(case, repeat, sctx)

    def case_metric_ids(self, case: Case, ids: list[str], sctx: SuiteContext) -> list[str]:
        """Ingestion cases get ingestion metrics only; generation metrics need the stage."""
        stages = set(sctx.params.get("stages", ["ingestion", "retrieval", "generation"]))
        if stage(case) == "ingestion":
            return [m for m in ids if m in INGESTION]
        out = [m for m in ids if m not in INGESTION]
        if "generation" not in stages:
            out = [m for m in out if m in RETRIEVAL]
        if "retrieval" not in stages:
            out = [m for m in out if m not in RETRIEVAL - {"freshness", "access_control"}]
        return out

    def panels(
        self, rows: list[ResultRow], runs: list[CaseRun], sctx: SuiteContext
    ) -> list[dict[str, Any]]:
        panels: list[dict[str, Any]] = []
        stats = next((r for r in rows if r.metric == "index_stats" and not r.skipped), None)
        if stats:
            panels.append(
                {
                    "title": "Index statistics",
                    "kind": "kv",
                    "items": {k: v for k, v in stats.details.items()},
                }
            )
        cmp_row = next((r for r in rows if r.metric == "config_compare" and not r.skipped), None)
        if cmp_row:
            cfgs = cmp_row.details["configs"]
            panels.append(
                {
                    "title": "Retriever configuration comparison",
                    "kind": "table",
                    "columns": ["config", "recall@k", "MRR", "n"],
                    "rows": [[n, v["recall_at_k"], v["mrr"], v["n"]] for n, v in cfgs.items()],
                    "note": f"best: {cmp_row.details.get('best')}",
                }
            )
        return panels
