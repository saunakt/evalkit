"""guardrails — guardrail precision/recall, false blocks, latency and fail mode (ch. 26)."""

from __future__ import annotations

from typing import Any

from evalkit.metrics.guardrails import confusion
from evalkit.registry import suite
from evalkit.schema import CaseRun, ResultRow
from evalkit.suites.base import BaseSuite
from evalkit.suites.context import SuiteContext


@suite("guardrails")
class GuardrailsSuite(BaseSuite):
    """Guardrail precision/recall, false blocks, added latency, fail mode, bypass (ch. 26)."""

    name = "guardrails"
    default_metrics = (
        "guard_precision",
        "guard_recall",
        "guard_f1",
        "false_block_rate",
        "added_latency",
        "fail_mode",
        "bypass_rate",
    )

    def panels(
        self, rows: list[ResultRow], runs: list[CaseRun], sctx: SuiteContext
    ) -> list[dict[str, Any]]:
        c = confusion(runs)
        return [
            {
                "title": "Confusion matrix (positive = harmful / blocked)",
                "kind": "table",
                "columns": ["", "blocked", "allowed"],
                "rows": [["harmful", c["tp"], c["fn"]], ["benign", c["fp"], c["tn"]]],
            }
        ]
