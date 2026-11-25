"""compliance — requirement coverage, AI disclosure, data rights, evidence (ch. 27)."""

from __future__ import annotations

from typing import Any

from evalkit.registry import suite
from evalkit.schema import CaseRun, ResultRow
from evalkit.suites.base import BaseSuite
from evalkit.suites.context import SuiteContext


@suite("compliance")
class ComplianceSuite(BaseSuite):
    """Requirement coverage, AI disclosure, data rights, evidence (ch. 27). Runs last."""

    name = "compliance"
    default_metrics = (
        "requirement_coverage",
        "ai_disclosure",
        "data_rights",
        "evidence_completeness",
    )

    def panels(
        self, rows: list[ResultRow], runs: list[CaseRun], sctx: SuiteContext
    ) -> list[dict[str, Any]]:
        cov = next((r for r in rows if r.metric == "requirement_coverage" and not r.skipped), None)
        if cov is None:
            return []
        status = {
            **{r: "covered" for r in cov.details["covered"]},
            **{r: "FAILING" for r in cov.details["failing"]},
            **{r: "NOT COVERED" for r in cov.details["uncovered"]},
        }
        reqs = cov.details.get("requirements", {})
        return [
            {
                "title": "Requirement coverage",
                "kind": "table",
                "columns": ["requirement", "regulation", "text", "owner", "status"],
                "rows": [
                    [
                        rid,
                        reqs.get(rid, {}).get("regulation"),
                        reqs.get(rid, {}).get("text"),
                        reqs.get(rid, {}).get("owner"),
                        st,
                    ]
                    for rid, st in sorted(status.items())
                ],
            }
        ]
