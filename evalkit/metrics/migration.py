"""Two-model comparison metrics for the migration suite (Plan.md §9.14).

Cases carry `extra.arm` ("current" | "candidate"), `extra.base_id` and `extra.sub_suite`.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from evalkit.metrics.common import na, result
from evalkit.registry import metric
from evalkit.schema import CaseRun, MetricResult
from evalkit.stats.compare import mcnemar_exact, slice_delta
from evalkit.suites.context import MetricContext
from evalkit.util import mean, percentile

MIG = ("migration",)
REFUSAL = re.compile(r"(?i)\b(can't help|cannot help|sorry, i can't|not able to)\b")


def arms(rows: list[CaseRun]) -> dict[str, dict[str, CaseRun]]:
    """arm → base_id → CaseRun."""
    out: dict[str, dict[str, CaseRun]] = defaultdict(dict)
    for cr in rows:
        arm = cr.case.extra.get("arm")
        if arm:
            out[str(arm)][str(cr.case.extra.get("base_id", cr.case.id))] = cr
    return out


def _passes(cr: CaseRun, metric_id: str) -> bool | None:
    vals = cr.passed_per_repeat(metric_id)
    return all(vals) if vals else None


def matched(rows: list[CaseRun]) -> list[tuple[str, CaseRun, CaseRun]]:
    a = arms(rows)
    cur, cand = a.get("current", {}), a.get("candidate", {})
    return [(k, cur[k], cand[k]) for k in sorted(set(cur) & set(cand))]


def per_metric(rows: list[CaseRun]) -> dict[str, tuple[dict[str, bool], dict[str, bool]]]:
    """metric (prefixed by sub-suite) → (current passes, candidate passes) by base id."""
    out: dict[str, tuple[dict[str, bool], dict[str, bool]]] = {}
    for key, cur, cand in matched(rows):
        sub = str(cur.case.extra.get("sub_suite", "?"))
        for mid in sorted({r.metric for r in cur.results} & {r.metric for r in cand.results}):
            a, b = _passes(cur, mid), _passes(cand, mid)
            if a is None or b is None:
                continue
            pair = out.setdefault(f"{sub}.{mid}", ({}, {}))
            pair[0][key], pair[1][key] = a, b
    return out


@metric("slice_delta", suites=MIG, kind="statistical", scope="dataset")
def slice_delta_metric(rows: list[CaseRun], ctx: MetricContext) -> MetricResult:
    """Per suite/metric/tag: candidate − current with paired bootstrap CI; flag big drops."""
    max_drop = ctx.suite.project.thresholds.regression.max_slice_drop
    table: dict[str, Any] = {}
    for name, (cur_pass, cand_pass) in per_metric(rows).items():
        table[name] = slice_delta(cur_pass, cand_pass, seed=ctx.seed)
    tags: dict[str, tuple[dict[str, bool], dict[str, bool]]] = defaultdict(lambda: ({}, {}))
    for key, cur, cand in matched(rows):
        for tag in cur.case.tags:
            a = [r.passed for r in cur.results if r.passed is not None]
            b = [r.passed for r in cand.results if r.passed is not None]
            if a and b:
                tags[f"tag:{tag}"][0][key], tags[f"tag:{tag}"][1][key] = all(a), all(b)
    for name, (cur_t, cand_t) in tags.items():
        table[name] = slice_delta(cur_t, cand_t, seed=ctx.seed)
    if not table:
        return na("no matched current/candidate results")
    drops = {k: v for k, v in table.items() if -v["delta"] > max_drop}
    worst = min(v["delta"] for v in table.values())
    return result(worst, not drops, -max_drop, slices=table, flagged=sorted(drops))


@metric("mcnemar", suites=MIG + ("finetune",), kind="statistical", scope="dataset")
def mcnemar(rows: list[CaseRun], ctx: MetricContext) -> MetricResult:
    """Exact McNemar per pass/fail metric on matched cases (current = A, candidate = B)."""
    tests = {}
    for name, (cur, cand) in per_metric(rows).items():
        ids = sorted(set(cur) & set(cand))
        tests[name] = mcnemar_exact([cur[i] for i in ids], [cand[i] for i in ids])
    if not tests:
        return na("no matched pass/fail results")
    worse = {k: v for k, v in tests.items() if v["p"] < 0.05 and v["b"] > v["c"]}
    min_p = min(v["p"] for v in tests.values())
    return result(min_p, not worse, 0.05, tests=tests, significantly_worse=sorted(worse))


def _arm_values(rows: list[CaseRun], fn: Any) -> tuple[list[float], list[float]]:
    a = arms(rows)
    cur = [
        v
        for cr in a.get("current", {}).values()
        for o in cr.observations
        if (v := fn(o)) is not None
    ]
    cand = [
        v
        for cr in a.get("candidate", {}).values()
        for o in cr.observations
        if (v := fn(o)) is not None
    ]
    return cur, cand


@metric(
    "latency_delta",
    suites=MIG,
    kind="statistical",
    scope="dataset",
    direction="lower",
    unit="seconds",
)
def latency_delta(rows: list[CaseRun], ctx: MetricContext) -> MetricResult:
    """p95 latency candidate − current."""
    cur, cand = _arm_values(rows, lambda o: o.latency_s)
    if not cur or not cand:
        return na("needs both arms")
    delta = (percentile(cand, 95) or 0.0) - (percentile(cur, 95) or 0.0)
    limit = ctx.p("max_latency_delta_s")
    return result(
        delta,
        None if limit is None else delta <= float(limit),
        current_p95=percentile(cur, 95),
        candidate_p95=percentile(cand, 95),
    )


@metric(
    "cost_delta", suites=MIG, kind="statistical", scope="dataset", direction="lower", unit="usd"
)
def cost_delta(rows: list[CaseRun], ctx: MetricContext) -> MetricResult:
    """Mean cost candidate − current."""
    cur, cand = _arm_values(rows, lambda o: o.cost_usd or 0.0)
    if not cur or not cand:
        return na("needs both arms")
    delta = (mean(cand) or 0.0) - (mean(cur) or 0.0)
    return result(delta, None, current=mean(cur), candidate=mean(cand))


@metric("refusal_delta", suites=MIG, kind="statistical", scope="dataset", direction="lower")
def refusal_delta(rows: list[CaseRun], ctx: MetricContext) -> MetricResult:
    """Refusal rate candidate − current."""
    cur, cand = _arm_values(rows, lambda o: float(bool(o.blocked or REFUSAL.search(o.text))))
    if not cur or not cand:
        return na("needs both arms")
    delta = (mean(cand) or 0.0) - (mean(cur) or 0.0)
    return result(
        delta,
        abs(delta) <= float(ctx.p("max_refusal_delta", 0.05)),
        current=mean(cur),
        candidate=mean(cand),
    )
