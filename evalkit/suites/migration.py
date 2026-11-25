"""migration — current vs candidate model comparison across suites (ch. 28.5)."""

from __future__ import annotations

import dataclasses
from typing import Any

from evalkit.planner import load_suite_cases, selected_metric_ids
from evalkit.registry import METRICS, SUITES, suite
from evalkit.schema import Case, CaseRun, Observation, ResultRow
from evalkit.suites.base import BaseSuite
from evalkit.suites.context import SuiteContext
from evalkit.util import ConfigError

ARMS = (("current", "sut"), ("candidate", "candidate"))


def arm_model(sctx: SuiteContext, role: str) -> str | None:
    rm = sctx.project.config.models.get(role)
    if rm is None:
        if role == "candidate":
            raise ConfigError(
                "migration needs models.candidate (e.g. --role candidate=<provider>:<model>)"
            )
        return None
    return rm.model


@suite("migration")
class MigrationSuite(BaseSuite):
    """Current vs candidate model comparison across suites (ch. 28.5)."""

    name = "migration"
    default_metrics = (
        "slice_delta",
        "pairwise_win_rate",
        "mcnemar",
        "latency_delta",
        "cost_delta",
        "refusal_delta",
    )

    def prepare(self, cases: list[Case], sctx: SuiteContext) -> list[Case]:
        """Load params.suites' cases; duplicate each into a current and a candidate arm."""
        subs = list(
            sctx.params.get("suites", ["quality", "rag", "agent", "security", "performance"])
        )
        out: list[Case] = []
        for sub in subs:
            sc = sctx.project.config.suites.get(sub)
            if sc is None or not sc.dataset:
                continue
            for case in load_suite_cases(sctx.project, sub, sc):
                for arm, role in ARMS:
                    out.append(
                        case.model_copy(
                            update={
                                "id": f"{sub}/{case.id}@{arm}",
                                "extra": {
                                    **case.extra,
                                    "arm": arm,
                                    "role": role,
                                    "base_id": f"{sub}/{case.id}",
                                    "sub_suite": sub,
                                    "model": arm_model(sctx, role),
                                },
                            }
                        )
                    )
        return out

    def _child(self, case: Case, sctx: SuiteContext) -> SuiteContext:
        sub = str(case.extra["sub_suite"])
        sc = sctx.project.config.suites[sub]
        return dataclasses.replace(
            sctx, name=sub, sc=sc, params=dict(sc.params), model=case.extra.get("model"), memo={}
        )

    def execute(self, case: Case, repeat: int, sctx: SuiteContext) -> Observation:
        sub = str(case.extra["sub_suite"])
        child = self._child(case, sctx)
        original = case.model_copy(update={"id": str(case.extra["base_id"]).split("/", 1)[1]})
        return SUITES.get(sub)().execute(original, repeat, child)  # type: ignore[no-any-return]

    def case_metric_ids(self, case: Case, ids: list[str], sctx: SuiteContext) -> list[str]:
        """The sub-suite's selected case-scope metrics."""
        sub = str(case.extra["sub_suite"])
        sc = sctx.project.config.suites[sub]
        chosen = selected_metric_ids(sctx.project, sub, sc)
        return [m for m in chosen if METRICS.get(m).scope == "case"]

    def panels(
        self, rows: list[ResultRow], runs: list[CaseRun], sctx: SuiteContext
    ) -> list[dict[str, Any]]:
        sd = next((r for r in rows if r.metric == "slice_delta" and not r.skipped), None)
        if sd is None:
            return []
        table = [
            [
                k,
                v["n"],
                round(v["base_rate"], 3),
                round(v["cand_rate"], 3),
                round(v["delta"], 3),
                f"{v['lo']:.2f}…{v['hi']:.2f}",
                round(v["p_mcnemar"], 3),
                ", ".join(v["broken"]),
            ]
            for k, v in sd.details["slices"].items()
        ]
        return [
            {
                "title": "Candidate vs current by suite/metric/tag",
                "kind": "table",
                "columns": [
                    "slice",
                    "n",
                    "current",
                    "candidate",
                    "Δ",
                    "95% CI",
                    "McNemar p",
                    "newly broken",
                ],
                "rows": table,
                "note": f"flagged drops: {', '.join(sd.details['flagged']) or 'none'}",
            }
        ]
