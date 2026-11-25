"""agent — tool-using agents: trajectories, end state, policies, faults, reliability (ch. 19–20)."""

from __future__ import annotations

from typing import Any

from evalkit.registry import suite
from evalkit.schema import CaseRun, ResultRow
from evalkit.suites.base import BaseSuite
from evalkit.suites.context import SuiteContext

RELIABILITY = ("pass_at_k", "pass_hat_k")


@suite("agent")
class AgentSuite(BaseSuite):
    """Tool-using agents: trajectories, end state, policies, faults, reliability (ch. 19–20)."""

    name = "agent"
    default_metrics = (
        "tool_selection",
        "argument_accuracy",
        "trajectory",
        "forbidden_tools",
        "step_efficiency",
        "end_state",
        "task_success",
        "policy_compliance",
        "error_recovery",
        "pass_hat_k",
    )

    def panels(
        self, rows: list[ResultRow], runs: list[CaseRun], sctx: SuiteContext
    ) -> list[dict[str, Any]]:
        panels: list[dict[str, Any]] = []
        rel = [r for r in rows if r.metric in RELIABILITY and not r.skipped]
        if rel:
            panels.append(
                {
                    "title": "Reliability over repeats",
                    "kind": "table",
                    "columns": [
                        "metric",
                        "k",
                        "empirical",
                        "estimate from p",
                        "n",
                        "not always passing",
                    ],
                    "rows": [
                        [
                            r.metric,
                            r.details.get("k"),
                            r.value,
                            r.details.get("estimate"),
                            r.details.get("n"),
                            ", ".join(r.details.get("failing", [])),
                        ]
                        for r in rel
                    ],
                }
            )
        traj = []
        for cr in runs:
            if cr.observations:
                calls = " → ".join(tc.name for tc in cr.observations[0].tool_calls) or "(none)"
                traj.append([cr.case.id, calls, cr.observations[0].text[:80]])
        if traj:
            panels.append(
                {
                    "title": "Trajectories (repeat 0)",
                    "kind": "table",
                    "columns": ["case", "tool calls", "reply"],
                    "rows": traj,
                }
            )
        return panels
