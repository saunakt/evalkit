"""prompt — system-prompt instruction following, scope and format (ch. 14)."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from evalkit.registry import suite
from evalkit.schema import CaseRun, ResultRow
from evalkit.suites.base import BaseSuite
from evalkit.suites.context import SuiteContext

INSTRUCTION_METRICS = (
    "instruction_positive",
    "instruction_negative",
    "format_compliance",
    "instruction_conflict",
    "out_of_scope_redirect",
)


@suite("prompt")
class PromptSuite(BaseSuite):
    """System-prompt instruction following, scope handling and format (ch. 14)."""

    name = "prompt"
    default_metrics = (
        "instruction_positive",
        "instruction_negative",
        "out_of_scope_redirect",
        "format_compliance",
        "instruction_conflict",
        "tool_definition_choice",
    )

    def panels(
        self, rows: list[ResultRow], runs: list[CaseRun], sctx: SuiteContext
    ) -> list[dict[str, Any]]:
        """Instruction × polarity pass-rate table."""
        cases = {cr.case.id: cr.case for cr in runs}
        cells: dict[tuple[str, str], list[bool]] = defaultdict(list)
        for r in rows:
            case = cases.get(r.case_id or "")
            if case is None or r.metric not in INSTRUCTION_METRICS or r.passed is None:
                continue
            key = (str(case.extra.get("instruction", "?")), str(case.extra.get("polarity", "?")))
            cells[key].append(bool(r.passed))
        if not cells:
            return []
        instructions = sorted({k[0] for k in cells})
        polarities = ["positive", "negative", "boundary"]
        table = []
        for ins in instructions:
            row: list[Any] = [ins]
            for pol in polarities:
                vals = cells.get((ins, pol))
                row.append(f"{sum(vals) / len(vals):.0%} (n={len(vals)})" if vals else "–")
            table.append(row)
        return [
            {
                "title": "Instruction × polarity pass rate",
                "kind": "table",
                "columns": ["instruction", *polarities],
                "rows": table,
            }
        ]
