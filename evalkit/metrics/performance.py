"""Performance and resilience metrics (Plan.md §9.13)."""

from __future__ import annotations

import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from evalkit.metrics.common import all_obs, na, result
from evalkit.registry import metric
from evalkit.schema import Case, CaseRun, MetricResult, Observation
from evalkit.suites.context import MetricContext
from evalkit.targets.base import has_hook
from evalkit.util import mean, percentile

PERF = ("performance",)
TRACEBACK = re.compile(r"Traceback|Exception|at .*\(.*\)")
OVERLOAD = re.compile(r"(?i)try again in a (minute|moment)|lot of requests")


def run_load(rows: list[CaseRun], ctx: MetricContext) -> dict[str, Any]:
    """Thread-pool load test per params.concurrency level (cached per suite run)."""
    memo = ctx.suite.memo
    if "load" in memo:
        return dict(memo["load"])
    cases = [cr.case for cr in rows] or []
    levels: dict[str, Any] = {}
    if ctx.p("engine") == "locust":
        from evalkit.engines.locust_engine import run_locust

        for level in [int(c) for c in ctx.p("concurrency", [1, 5])]:
            levels[str(level)] = run_locust(
                ctx.target, level, int(ctx.p("duration_s", 30)), ctx.run_dir / "locust" / str(level)
            )
        memo["load"] = levels
        return levels
    n_req = int(ctx.p("requests", 20))
    for level in [int(c) for c in ctx.p("concurrency", [1, 5])]:
        jobs = [cases[i % len(cases)] for i in range(n_req)] if cases else []
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=level) as pool:
            obs = list(pool.map(lambda c: ctx.suite.call(c), jobs))
        wall = time.perf_counter() - start
        lats = [o.latency_s for o in obs]
        levels[str(level)] = {
            "requests": len(obs),
            "wall_s": wall,
            "throughput": len(obs) / wall if wall else 0.0,
            "p50": percentile(lats, 50),
            "p95": percentile(lats, 95),
            "p99": percentile(lats, 99),
            "errors": sum(1 for o in obs if o.error),
        }
    memo["load"] = levels
    return levels


def _latency(rows: list[CaseRun], ctx: MetricContext, q: int) -> MetricResult:
    if ctx.p("mode", "sequential") == "load":
        levels = run_load(rows, ctx)
        vals = {k: v[f"p{q}"] for k, v in levels.items()}
        value = max(v for v in vals.values() if v is not None) if vals else None
        return result(value, _within(value, ctx, f"max_latency_p{q}"), per_concurrency=vals)
    lats = [o.latency_s for o in all_obs(rows)]
    if not lats:
        return na("no observations")
    value = percentile(lats, q)
    return result(value, _within(value, ctx, f"max_latency_p{q}"), n=len(lats))


def _within(value: float | None, ctx: MetricContext, key: str) -> bool | None:
    limit = ctx.p(key)
    return None if limit is None or value is None else value <= float(limit)


def _register_latency(q: int) -> None:
    def fn(rows: list[CaseRun], ctx: MetricContext) -> MetricResult:
        return _latency(rows, ctx, q)

    fn.__doc__ = f"p{q} end-to-end latency (per concurrency level in load mode)."
    metric(
        f"latency_p{q}",
        suites=PERF,
        kind="statistical",
        scope="dataset",
        direction="lower",
        unit="seconds",
    )(fn)


for _q in (50, 95, 99):
    _register_latency(_q)


def measure_ttft(case: Case, ctx: MetricContext) -> tuple[float | None, float, int]:
    """(ttft, total, pieces) by consuming target.stream()."""
    start = time.perf_counter()
    first: float | None = None
    pieces = 0
    for _ in ctx.target.stream(case, ctx.suite.target_ctx(case)):
        if first is None:
            first = time.perf_counter() - start
        pieces += 1
    return first, time.perf_counter() - start, pieces


def _ttfts(rows: list[CaseRun], ctx: MetricContext) -> list[tuple[float, float, int]]:
    memo = ctx.suite.memo
    if "ttft" not in memo:
        out = []
        for cr in rows[: int(ctx.p("ttft_cases", 10))]:
            got = [o.ttft_s for o in cr.observations if o.ttft_s is not None]
            if got:
                out.append((got[0], cr.observations[0].latency_s, cr.observations[0].tokens_out))
            elif has_hook(ctx.target, "stream"):
                ctx.suite.reset(ctx.seed)
                t, total, n = measure_ttft(cr.case, ctx)
                if t is not None:
                    out.append((t, total, n))
        memo["ttft"] = out
    data: list[tuple[float, float, int]] = memo["ttft"]
    return data


@metric(
    "ttft",
    suites=PERF + ("multimodal",),
    kind="statistical",
    scope="dataset",
    direction="lower",
    unit="seconds",
)
def ttft(rows: list[CaseRun], ctx: MetricContext) -> MetricResult:
    """Time to first token (stream hook or obs.ttft_s): p95 (p50 in details)."""
    data = _ttfts(rows, ctx)
    if not data:
        return na("needs a stream hook or obs.ttft_s")
    vals = [d[0] for d in data]
    value = percentile(vals, 95)
    return result(value, _within(value, ctx, "max_ttft"), p50=percentile(vals, 50), n=len(vals))


@metric("tokens_per_second", suites=PERF, kind="statistical", scope="dataset", unit="tok/s")
def tokens_per_second(rows: list[CaseRun], ctx: MetricContext) -> MetricResult:
    """Output tokens / (latency − ttft)."""
    data = _ttfts(rows, ctx)
    rates = [n / (total - t) for t, total, n in data if total > t]
    if not rates:
        return na("needs streaming timings")
    return result(mean(rates), None, n=len(rates))


@metric(
    "component_latency",
    suites=PERF,
    kind="statistical",
    scope="dataset",
    direction="lower",
    unit="seconds",
)
def component_latency(rows: list[CaseRun], ctx: MetricContext) -> MetricResult:
    """p95 per obs.spans[].name; value = the slowest component's p95."""
    spans: dict[str, list[float]] = {}
    for o in all_obs(rows):
        for s in o.spans:
            spans.setdefault(s.name, []).append(s.latency_s)
    if not spans:
        return na("no spans reported")
    table = {k: percentile(v, 95) for k, v in sorted(spans.items())}
    return result(max(v for v in table.values() if v is not None), None, p95=table)


@metric("throughput", suites=PERF, kind="statistical", scope="dataset", unit="req/s")
def throughput(rows: list[CaseRun], ctx: MetricContext) -> MetricResult:
    """Completed requests / wall time (per concurrency level in load mode)."""
    if ctx.p("mode", "sequential") == "load":
        levels = run_load(rows, ctx)
        per = {k: v["throughput"] for k, v in levels.items()}
        return result(max(per.values()) if per else None, None, per_concurrency=per)
    lats = [o.latency_s for o in all_obs(rows)]
    total = sum(lats)
    return result(len(lats) / total if total else None, None, n=len(lats))


def error_class(err: str) -> str:
    low = err.lower()
    for key, words in (
        ("timeout", ("timeout", "timed out")),
        ("429", ("429", "rate")),
        ("5xx", ("500", "502", "503", "504", "5xx")),
        ("malformed", ("malformed", "json")),
    ):
        if any(w in low for w in words):
            return key
    return "other"


@metric("error_rate", suites=PERF, kind="statistical", scope="dataset", direction="lower")
def error_rate(rows: list[CaseRun], ctx: MetricContext) -> MetricResult:
    """Share of calls with an error, by class (timeout, 429, 5xx, malformed output)."""
    obs = all_obs(rows)
    if not obs:
        return na("no observations")
    errs = [error_class(o.error) for o in obs if o.error]
    value = len(errs) / len(obs)
    return result(
        value, value <= float(ctx.p("max_error_rate", 0.01)), by_class=dict(Counter(errs))
    )


@metric(
    "cost_per_request",
    suites=PERF,
    kind="statistical",
    scope="dataset",
    direction="lower",
    unit="usd",
)
def cost_per_request(rows: list[CaseRun], ctx: MetricContext) -> MetricResult:
    """Mean cost per call (obs.cost_usd)."""
    costs = [o.cost_usd for o in all_obs(rows) if o.cost_usd is not None]
    if not costs:
        return na("targets report no cost")
    value = mean(costs) or 0.0
    return result(value, _within(value, ctx, "max_cost_per_request"), n=len(costs))


@metric("cache_hit_rate", suites=PERF, kind="statistical", scope="dataset")
def cache_hit_rate(rows: list[CaseRun], ctx: MetricContext) -> MetricResult:
    """Share of obs.cache_hit == true."""
    flags = [o.cache_hit for o in all_obs(rows) if o.cache_hit is not None]
    if not flags:
        return na("targets report no cache_hit")
    return result(sum(flags) / len(flags), None, n=len(flags))


def _probe(rows: list[CaseRun], ctx: MetricContext, fault: dict[str, Any]) -> list[Observation]:
    out = []
    for cr in rows[: int(ctx.p("resilience_cases", 2))]:
        out.append(ctx.suite.recall(cr.case.model_copy(update={"faults": [fault]})))
    return out


@metric("retry_backoff", suites=PERF, kind="deterministic", scope="dataset", unit="pass")
def retry_backoff(rows: list[CaseRun], ctx: MetricContext) -> MetricResult:
    """Fault http_429 × n: the request still succeeds with retries ≤ params.max_retries."""
    n = int(ctx.p("retry_faults", 2))
    obs = _probe(rows, ctx, {"provider": "llm", "error": "http_429", "times": n})
    if not obs:
        return na("no cases")
    retries = [sum(1 for e in o.events if e.get("type") == "retry") for o in obs]
    ok = all(not o.error and not OVERLOAD.search(o.text) for o in obs) and max(retries) <= int(
        ctx.p("max_retries", 3)
    )
    return result(1.0 if ok else 0.0, ok, retries=retries)


@metric("timeout_handling", suites=PERF, kind="deterministic", scope="dataset", unit="pass")
def timeout_handling(rows: list[CaseRun], ctx: MetricContext) -> MetricResult:
    """Fault `slow` beyond the timeout: graceful message within params.max_wait_s."""
    obs = _probe(
        rows, ctx, {"provider": "llm", "error": "slow", "delay_s": float(ctx.p("slow_delay_s", 30))}
    )
    if not obs:
        return na("no cases")
    wait = float(ctx.p("max_wait_s", 10))
    ok = all(o.text.strip() and not TRACEBACK.search(o.text) and o.latency_s <= wait for o in obs)
    return result(1.0 if ok else 0.0, ok, replies=[o.text[:120] for o in obs])


@metric("graceful_degradation", suites=PERF, kind="deterministic", scope="dataset", unit="pass")
def graceful_degradation(rows: list[CaseRun], ctx: MetricContext) -> MetricResult:
    """Fault http_500: friendly text and no stack trace."""
    obs = _probe(rows, ctx, {"provider": "llm", "error": "http_500", "times": 1})
    if not obs:
        return na("no cases")
    bad = [o.text[:120] for o in obs if not o.text.strip() or TRACEBACK.search(o.text)]
    return result(0.0 if bad else 1.0, not bad, bad_replies=bad)


@metric(
    "stream_interruption",
    suites=PERF,
    kind="deterministic",
    scope="dataset",
    unit="pass",
    needs=("hook:stream",),
)
def stream_interruption(rows: list[CaseRun], ctx: MetricContext) -> MetricResult:
    """Fault stream_drop: partial output is handled (no crash; apology or retry)."""
    if not rows:
        return na("no cases")
    case = rows[0].case.model_copy(update={"faults": [{"provider": "llm", "error": "stream_drop"}]})
    ctx.suite.reset(ctx.seed)
    try:
        text = "".join(ctx.target.stream(case, ctx.suite.target_ctx(case)))
    except Exception as exc:  # a crash here is exactly what this metric detects
        return result(0.0, False, error=f"{type(exc).__name__}: {exc}")
    ok = bool(re.search(r"(?i)sorry|again|lost|retry", text))
    return result(1.0 if ok else 0.0, ok, tail=text[-120:])


@metric("fallback_quality", suites=PERF, kind="statistical", scope="dataset")
def fallback_quality(rows: list[CaseRun], ctx: MetricContext) -> MetricResult:
    """Re-run params.base_metrics with ctx.model = models.fallback; compare pass rates."""
    fb = ctx.suite.project.config.models.get("fallback")
    if fb is None:
        return na("models.fallback is not configured")
    from evalkit.execution import run_case_metric
    from evalkit.registry import METRICS

    base = [str(m) for m in ctx.p("base_metrics", ["contains"])]
    primary, fallback = [], []
    for cr in rows:
        obs = ctx.suite.recall(cr.case, model=fb.model)
        for m in base:
            r_fb = run_case_metric(METRICS.get(m), "native", cr.case, obs, ctx)
            r_pr = [r for r in cr.by_metric(m) if r.passed is not None]
            if r_fb.passed is not None and r_pr:
                fallback.append(float(r_fb.passed))
                primary.append(float(bool(r_pr[0].passed)))
    if not primary:
        return na(f"no {base} results to compare")
    delta = (mean(fallback) or 0.0) - (mean(primary) or 0.0)
    return result(
        mean(fallback),
        delta >= -float(ctx.p("max_fallback_drop", 0.1)),
        primary=mean(primary),
        delta=delta,
        model=fb.model,
    )
