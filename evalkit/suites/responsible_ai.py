"""responsible_ai — hallucination, bias, toxicity, refusals and PII (ch. 25)."""

from __future__ import annotations

from typing import Any

from evalkit.metrics.safety import variant_cases
from evalkit.registry import suite
from evalkit.schema import Case, CaseRun, ResultRow
from evalkit.suites.base import BaseSuite
from evalkit.suites.context import SuiteContext


@suite("responsible_ai")
class ResponsibleAiSuite(BaseSuite):
    """Hallucination, bias, toxicity, refusals and PII (ch. 25)."""

    name = "responsible_ai"
    default_metrics = (
        "unanswerable",
        "false_premise",
        "fake_entity",
        "number_grounding",
        "bias_gap",
        "harmful_refusal_rate",
        "over_refusal_rate",
        "pii_leakage_rate",
        "toxicity_rate",
    )

    def prepare(self, cases: list[Case], sctx: SuiteContext) -> list[Case]:
        """Expand counterfactual templates (template + variants dict) into groups."""
        combine = str(sctx.params.get("combine", "zip"))
        out: list[Case] = []
        for case in cases:
            if case.template and isinstance(case.variants, dict):
                out += variant_cases(case, combine)
            else:
                out.append(case)
        return out

    def panels(
        self, rows: list[ResultRow], runs: list[CaseRun], sctx: SuiteContext
    ) -> list[dict[str, Any]]:
        bias = next((r for r in rows if r.metric == "bias_gap" and not r.skipped), None)
        if not bias:
            return []
        items = []
        for gid, g in bias.details.get("groups", {}).items():
            for key, rate in g["rates"].items():
                items.append({"label": f"{gid}: {key}", "value": rate, "n": None})
        return [
            {
                "title": f"Decision rate by counterfactual group (positive = "
                f"{bias.details.get('positive')!r})",
                "kind": "bars",
                "items": items,
            }
        ]
