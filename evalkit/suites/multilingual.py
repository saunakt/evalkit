"""multilingual — Indian languages, scripts and code-mixing (ch. 30)."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from evalkit.registry import suite
from evalkit.schema import Case, CaseRun, ResultRow
from evalkit.suites.base import BaseSuite
from evalkit.suites.context import SuiteContext


@suite("multilingual")
class MultilingualSuite(BaseSuite):
    """Indian languages, scripts and code-mixing (ch. 30)."""

    name = "multilingual"
    default_metrics = (
        "language_match",
        "script_match",
        "entity_preservation",
        "correctness_parity",
        "guardrail_indic",
    )

    def case_metric_ids(self, case: Case, ids: list[str], sctx: SuiteContext) -> list[str]:
        """Also score params.base_metrics (used by correctness_parity)."""
        base = [str(m) for m in sctx.params.get("base_metrics", ["contains"])]
        return ids + [m for m in base if m not in ids]

    def panels(
        self, rows: list[ResultRow], runs: list[CaseRun], sctx: SuiteContext
    ) -> list[dict[str, Any]]:
        per: dict[str, list[bool]] = defaultdict(list)
        for r in rows:
            if r.case_id and r.passed is not None and r.skipped is None:
                per[r.language or "en"].append(bool(r.passed))
        return [
            {
                "title": "Pass rate by language",
                "kind": "bars",
                "items": [
                    {"label": k, "value": sum(v) / len(v), "n": len(v)}
                    for k, v in sorted(per.items())
                ],
            }
        ]
