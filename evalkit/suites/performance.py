"""performance — latency, TTFT, throughput, errors, cost and resilience (ch. 28.1–28.4)."""

from __future__ import annotations

from typing import Any

from evalkit.registry import suite
from evalkit.schema import Case, CaseRun, ResultRow
from evalkit.suites.base import BaseSuite
from evalkit.suites.context import SuiteContext


@suite("performance")
class PerformanceSuite(BaseSuite):
    """Latency, TTFT, throughput, errors, cost and resilience (ch. 28.1–28.4).

    params.mode: sequential (default) or load (thread-pool load per params.concurrency level;
    `engine: locust` runs Locust headless against http targets).
    """

    name = "performance"
    default_metrics = (
        "ttft",
        "latency_p50",
        "latency_p95",
        "throughput",
        "error_rate",
        "cost_per_request",
        "retry_backoff",
        "graceful_degradation",
    )

    def case_metric_ids(self, case: Case, ids: list[str], sctx: SuiteContext) -> list[str]:
        """Also score the success metric (cost_per_success) and base metrics (fallback)."""
        extra = [str(sctx.params.get("success_metric", "contains"))]
        extra += [str(m) for m in sctx.params.get("base_metrics", [])]
        return ids + [m for m in dict.fromkeys(extra) if m not in ids]

    def panels(
        self, rows: list[ResultRow], runs: list[CaseRun], sctx: SuiteContext
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        load = sctx.memo.get("load")
        if load:
            out.append(
                {
                    "title": "Load test by concurrency",
                    "kind": "table",
                    "columns": [
                        "concurrency",
                        "requests",
                        "throughput req/s",
                        "p50 s",
                        "p95 s",
                        "p99 s",
                        "errors",
                    ],
                    "rows": [
                        [
                            k,
                            v["requests"],
                            v["throughput"],
                            v["p50"],
                            v["p95"],
                            v["p99"],
                            v["errors"],
                        ]
                        for k, v in load.items()
                    ],
                }
            )
        comp = next((r for r in rows if r.metric == "component_latency" and not r.skipped), None)
        if comp:
            out.append(
                {
                    "title": "Component latency (p95, s)",
                    "kind": "kv",
                    "items": comp.details.get("p95", {}),
                }
            )
        return out
