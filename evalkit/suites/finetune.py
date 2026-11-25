"""finetune — fine-tuned vs prompted base model comparisons (ch. 29.4)."""

from __future__ import annotations

from typing import Any

from evalkit.registry import suite
from evalkit.schema import Case, CaseRun, Observation, ResultRow
from evalkit.suites.base import BaseSuite
from evalkit.suites.context import SuiteContext

VARIANTS = ("base_prompted", "tuned", "tuned_prompted")


@suite("finetune")
class FinetuneSuite(BaseSuite):
    """Fine-tuned vs prompted base model comparisons (ch. 29.4).

    params.targets maps base_prompted / tuned / tuned_prompted to target yaml files; each case
    runs once per configured variant.
    """

    name = "finetune"
    default_metrics = (
        "gain_vs_prompted_base",
        "heldout_generalisation",
        "train_test_overlap",
        "forgetting_regression",
        "safety_regression",
        "memorisation_canary",
    )

    def prepare(self, cases: list[Case], sctx: SuiteContext) -> list[Case]:
        targets: dict[str, Any] = dict(sctx.params.get("targets", {}))
        variants = [v for v in VARIANTS if v in targets] or ["tuned"]
        return [
            c.model_copy(
                update={"id": f"{c.id}@{v}", "extra": {**c.extra, "variant": v, "base_id": c.id}}
            )
            for c in cases
            for v in variants
        ]

    def execute(self, case: Case, repeat: int, sctx: SuiteContext) -> Observation:
        ref = dict(sctx.params.get("targets", {})).get(str(case.extra.get("variant")))
        if ref is None:
            return super().execute(case, repeat, sctx)
        tgt = sctx.resolve_target(ref)
        sctx.reset(sctx.seed, tgt)
        return sctx.call(case, target=tgt)

    def case_metric_ids(self, case: Case, ids: list[str], sctx: SuiteContext) -> list[str]:
        base = [str(m) for m in sctx.params.get("base_metrics", ["contains"])]
        return ids + [m for m in base if m not in ids]

    def panels(
        self, rows: list[ResultRow], runs: list[CaseRun], sctx: SuiteContext
    ) -> list[dict[str, Any]]:
        per: dict[str, list[bool]] = {}
        for r in rows:
            if r.case_id and r.passed is not None and "@" in r.case_id:
                per.setdefault(r.case_id.rsplit("@", 1)[1], []).append(bool(r.passed))
        return [
            {
                "title": "Pass rate by model variant",
                "kind": "bars",
                "items": [
                    {"label": k, "value": sum(v) / len(v), "n": len(v)}
                    for k, v in sorted(per.items())
                ],
            }
        ]
