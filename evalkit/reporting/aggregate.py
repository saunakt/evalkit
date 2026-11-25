"""Per-suite / per-metric / per-slice aggregation (Plan.md §11.3, §9.19)."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

from evalkit.registry import METRICS
from evalkit.schema import ResultRow
from evalkit.stats.intervals import wilson
from evalkit.stats.reliability import flaky
from evalkit.stats.sample_size import cases_needed, slice_table
from evalkit.util import mean

NA_PREFIX = "n/a"


@dataclass
class MetricSummary:
    metric: str
    scope: str = "case"
    kind: str = "deterministic"
    direction: str = "higher"
    unit: str = "ratio"
    n: int = 0
    n_skipped: int = 0
    n_na: int = 0
    value: float | None = None
    passes: int = 0
    n_scored: int = 0
    pass_rate: float | None = None
    lo: float | None = None
    hi: float | None = None
    passed: bool | None = None
    skipped_reasons: dict[str, int] = field(default_factory=dict)
    uncalibrated: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def not_tested(self) -> bool:
        return self.n == 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_metric(metric: str, rows: list[ResultRow]) -> MetricSummary:
    spec = METRICS.get(metric) if metric in METRICS else None
    s = MetricSummary(metric=metric)
    if spec is not None:
        s.scope, s.kind, s.direction, s.unit = spec.scope, spec.kind, spec.direction, spec.unit
    elif rows and all(r.case_id is None for r in rows):
        s.scope = "dataset"  # unregistered metric (e.g. a store from another version)
    computed = [r for r in rows if r.skipped is None]
    skipped = [r for r in rows if r.skipped is not None]
    s.n, s.n_skipped = len(computed), len(skipped)
    s.n_na = sum(1 for r in skipped if r.skipped and r.skipped.startswith(NA_PREFIX))
    s.skipped_reasons = dict(sorted(Counter(r.skipped or "" for r in skipped).items()))
    s.value = mean([r.value for r in computed])
    scored = [r for r in computed if r.passed is not None]
    s.n_scored, s.passes = len(scored), sum(1 for r in scored if r.passed)
    if scored:
        s.pass_rate = s.passes / len(scored)
        s.lo, s.hi = wilson(s.passes, len(scored))
        s.passed = s.passes == len(scored) if s.scope == "dataset" else None
    s.uncalibrated = any(r.details.get("uncalibrated") for r in computed)
    if s.scope == "dataset" and computed:
        s.details = computed[0].details
    return s


@dataclass
class SuiteSummary:
    suite: str
    metrics: dict[str, MetricSummary]
    n_cases: int
    overall: dict[str, Any]
    slices: dict[str, list[dict[str, Any]]]
    flaky_cases: list[str]
    failing: list[dict[str, Any]]
    cost_usd: float
    tokens: int


def _case_rows(rows: list[ResultRow]) -> list[ResultRow]:
    return [r for r in rows if r.case_id is not None and r.skipped is None]


def _failing_cases(rows: list[ResultRow]) -> tuple[list[dict[str, Any]], list[str]]:
    per_case: dict[str, list[ResultRow]] = defaultdict(list)
    for r in _case_rows(rows):
        if r.passed is not None:
            per_case[str(r.case_id)].append(r)
    failing, rates = [], {}
    for cid in sorted(per_case):
        crow = per_case[cid]
        by_repeat: dict[int, bool] = defaultdict(lambda: True)
        for r in crow:
            by_repeat[r.repeat or 0] &= bool(r.passed)
        rates[cid] = sum(by_repeat.values()) / len(by_repeat)
        bad = [r for r in crow if not r.passed]
        if bad:
            failing.append(
                {
                    "case_id": cid,
                    "pass_rate": rates[cid],
                    "must_pass": any(r.must_pass for r in crow),
                    "failed": sorted({r.metric for r in bad}),
                    "reasons": [_reason(r) for r in bad[:6]],
                    "trace_id": next((r.trace_id for r in crow if r.trace_id), None),
                }
            )
    return failing, flaky(rates)


def _reason(r: ResultRow) -> str:
    d = r.details
    why = d.get("reasoning") or d.get("reason") or d.get("missing") or d.get("violations")
    return f"{r.metric} (repeat {r.repeat}): {why if why is not None else 'failed'}"[:300]


def summarize_suite(suite: str, rows: list[ResultRow], metric_ids: list[str]) -> SuiteSummary:
    by_metric: dict[str, list[ResultRow]] = defaultdict(list)
    for r in rows:
        by_metric[r.metric].append(r)
    metrics = {m: summarize_metric(m, by_metric.get(m, [])) for m in metric_ids}
    scored = [r for r in _case_rows(rows) if r.passed is not None]
    passes = sum(1 for r in scored if r.passed)
    lo, hi = wilson(passes, len(scored))
    rate = passes / len(scored) if scored else None
    failing, flaky_ids = _failing_cases(rows)
    case_ids = {r.case_id for r in rows if r.case_id is not None}
    overall = {
        "pass_rate": rate,
        "lo": lo,
        "hi": hi,
        "n": len(scored),
        "flaky_rate": len(flaky_ids) / len(case_ids) if case_ids else 0.0,
        "cases_needed_5pct": cases_needed(rate, 0.05) if rate is not None else None,
    }
    slices = {
        "tag": slice_table(scored, "tags"),
        "language": slice_table(scored, "language"),
        "split": slice_table(scored, "split"),
        "must_pass": slice_table(scored, "must_pass"),
    }
    obs_rows = {(r.case_id, r.repeat): r for r in rows if r.case_id is not None}
    cost = sum(r.cost_usd or 0.0 for r in obs_rows.values())
    tokens = sum(r.tokens or 0 for r in obs_rows.values())
    return SuiteSummary(
        suite, metrics, len(case_ids), overall, slices, flaky_ids, failing, cost, tokens
    )
