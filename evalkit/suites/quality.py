"""quality — correctness, format and judged quality of single replies (ch. 5–7)."""

from __future__ import annotations

from typing import Any

from evalkit.registry import suite
from evalkit.schema import CaseRun, ResultRow
from evalkit.suites.base import BaseSuite
from evalkit.suites.context import SuiteContext


@suite("quality")
class QualitySuite(BaseSuite):
    """Correctness, format and judged quality of single replies (ch. 5–7)."""

    name = "quality"
    default_metrics = (
        "exact_match",
        "contains",
        "not_contains",
        "regex",
        "json_schema",
        "numeric_tolerance",
        "length",
        "judge_binary",
    )

    def panels(
        self, rows: list[ResultRow], runs: list[CaseRun], sctx: SuiteContext
    ) -> list[dict[str, Any]]:
        cm = next((r for r in rows if r.metric == "confusion_matrix" and not r.skipped), None)
        if cm is None:
            return []
        labels = cm.details["labels"]
        return [
            {
                "title": "Confusion matrix (rows = true label)",
                "kind": "table",
                "columns": ["true \\ predicted", *labels],
                "rows": [
                    [lab, *vals] for lab, vals in zip(labels, cm.details["matrix"], strict=False)
                ],
            }
        ]
