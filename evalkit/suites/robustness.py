"""robustness — decisions stay stable under perturbations and rephrasing (ch. 14.3, 15)."""

from __future__ import annotations

from typing import Any

from evalkit.env import perturb
from evalkit.registry import suite
from evalkit.schema import Case, CaseRun, ResultRow
from evalkit.suites.base import BaseSuite
from evalkit.suites.context import SuiteContext


def variant(case: Case, n: int, text: str, transform: str, relation: str) -> Case:
    safe = transform.replace(":", "-")
    return case.model_copy(
        update={
            "id": f"{case.id}~{n:02d}-{safe}",
            "input": text,
            "variants": None,
            "extra": {**case.extra, "group": case.id, "transform": transform, "relation": relation},
        }
    )


@suite("robustness")
class RobustnessSuite(BaseSuite):
    """Decisions stay stable under perturbations and rephrasing (ch. 14.3, 15)."""

    name = "robustness"
    default_metrics = (
        "metamorphic_invariance",
        "metamorphic_change",
        "perturbation_drop",
        "rephrase_consistency",
    )

    def prepare(self, cases: list[Case], sctx: SuiteContext) -> list[Case]:
        """Expand every case into a group: original + explicit variants + perturbations."""
        transforms = [str(t) for t in sctx.params.get("perturbations", [])]
        llm = None
        if any(t.split(":")[0] in perturb.LLM_TRANSFORMS for t in transforms):
            llm = sctx.llm("perturb")
        out: list[Case] = []
        for case in cases:
            text = case.input or case.text_input()
            relation = case.relation or "same_decision"
            out.append(
                case.model_copy(
                    update={"extra": {**case.extra, "group": case.id, "transform": "original"}}
                )
            )
            explicit = case.variants if isinstance(case.variants, list) else []
            for i, v in enumerate(explicit):
                out.append(variant(case, i, v, "explicit", relation))
            if case.relation == "different_decision":
                continue  # perturbations should never change the decision
            for j, t in enumerate(transforms, start=len(explicit)):
                vtext, name = perturb.apply(t, text, sctx.seed, llm=llm)
                if vtext != text:
                    out.append(variant(case, j, vtext, name, "same_decision"))
        return out

    def case_metric_ids(self, case: Case, ids: list[str], sctx: SuiteContext) -> list[str]:
        """Also score params.base_metrics (used by perturbation_drop)."""
        base = [str(m) for m in sctx.params.get("base_metrics", ["contains"])]
        return ids + [m for m in base if m not in ids]

    def panels(
        self, rows: list[ResultRow], runs: list[CaseRun], sctx: SuiteContext
    ) -> list[dict[str, Any]]:
        out = []
        for r in rows:
            if r.metric in ("metamorphic_invariance", "metamorphic_change") and r.details.get(
                "failures"
            ):
                out.append(
                    {
                        "title": f"Decision flips ({r.metric})",
                        "kind": "table",
                        "columns": ["group", "variant", "transform", "original", "variant"],
                        "rows": [
                            [
                                f["group"],
                                f["variant"],
                                f["transform"],
                                f["original_decision"],
                                f["variant_decision"],
                            ]
                            for f in r.details["failures"]
                        ],
                    }
                )
        return out
