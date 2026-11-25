"""Helpers shared by metric implementations (no registrations here)."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from evalkit.judges.runner import Verdict
from evalkit.schema import Case, CaseRun, MetricResult, Observation
from evalkit.util import mean, normalize_text, percentile


def result(
    value: float | None = None,
    passed: bool | None = None,
    threshold: float | None = None,
    **details: Any,
) -> MetricResult:
    """Metric id is filled in by the runner."""
    return MetricResult(metric="", value=value, passed=passed, threshold=threshold, details=details)


def scored(
    value: float, threshold: float, direction: str = "higher", **details: Any
) -> MetricResult:
    ok = value >= threshold if direction == "higher" else value <= threshold
    return result(value, ok, threshold, **details)


def boolean(ok: bool, **details: Any) -> MetricResult:
    return result(1.0 if ok else 0.0, ok, **details)


def na(reason: str) -> MetricResult:
    return MetricResult(metric="", skipped=f"n/a: {reason}")


def skip(reason: str) -> MetricResult:
    return MetricResult(metric="", skipped=reason)


def from_verdict(v: Verdict, **details: Any) -> MetricResult:
    d = {"reasoning": v.reasoning, **v.details, **details}
    return result(v.score, v.passed, **d)


def contains_all(text: str, needles: list[str]) -> list[str]:
    """Needles missing from text (case-insensitive, whitespace-normalised)."""
    low = normalize_text(text)
    return [n for n in needles if normalize_text(n) not in low]


def contains_any(text: str, needles: list[str]) -> list[str]:
    low = normalize_text(text)
    return [n for n in needles if normalize_text(n) and normalize_text(n) in low]


def context_text(obs: Observation, extra: list[str] | None = None) -> str:
    parts = [f"[{c.id}] {c.text}" for c in obs.retrieved]
    parts += [f"tool {tc.name} → {tc.result}" for tc in obs.tool_calls if tc.result is not None]
    return "\n".join(parts + (extra or []))


def transcript(obs: Observation) -> str:
    if not obs.messages:
        return f"assistant: {obs.text}"
    return "\n".join(f"{m.get('role')}: {m.get('content')}" for m in obs.messages)


def trajectory(obs: Observation) -> list[dict[str, Any]]:
    return [
        {
            "tool": tc.name,
            "args": tc.args,
            "error": tc.error,
            **({"agent": tc.agent} if tc.agent else {}),
        }
        for tc in obs.tool_calls
    ]


def case_sections(case: Case, obs: Observation, **more: Any) -> dict[str, Any]:
    """Standard judge evidence: input + output (+ optional extras)."""
    sec: dict[str, Any] = {"User input": case.text_input(), "Output": obs.text}
    sec.update({k: v for k, v in more.items() if v})
    return sec


def extract_decision(text: str, spec: Any) -> str | None:
    """Decision from text: regex capture (str) or a label list (first label found)."""
    if spec is None:
        return normalize_text(text)
    if isinstance(spec, str):
        m = re.search(spec, text, re.I | re.S)
        if not m:
            return None
        return normalize_text(m.group(1) if m.groups() else m.group(0))
    if isinstance(spec, list | tuple):
        low = normalize_text(text)
        for label in spec:
            if normalize_text(str(label)) in low:
                return str(label)
        return None
    if isinstance(spec, dict) and "labels" in spec:
        return extract_decision(text, list(spec["labels"]))
    return normalize_text(text)


def passes_by_case(rows: list[CaseRun], metric: str) -> dict[str, list[bool]]:
    out: dict[str, list[bool]] = {}
    for cr in rows:
        vals = cr.passed_per_repeat(metric)
        if vals:
            out[cr.case.id] = vals
    return out


def values_of(rows: list[CaseRun], metric: str) -> list[float]:
    return [
        r.value
        for cr in rows
        for r in cr.by_metric(metric)
        if r.skipped is None and r.value is not None
    ]


def all_obs(rows: list[CaseRun]) -> list[Observation]:
    return [o for cr in rows for o in cr.observations]


def latency_stats(obs: list[Observation]) -> dict[str, float | None]:
    lats = [o.latency_s for o in obs if o.latency_s]
    return {
        "p50": percentile(lats, 50),
        "p95": percentile(lats, 95),
        "p99": percentile(lats, 99),
        "mean": mean(lats),
        "n": float(len(lats)),
    }


def group_by(rows: list[CaseRun], key: Callable[[CaseRun], str | None]) -> dict[str, list[CaseRun]]:
    out: dict[str, list[CaseRun]] = defaultdict(list)
    for cr in rows:
        k = key(cr)
        if k is not None:
            out[k].append(cr)
    return dict(sorted(out.items()))


def rate(vals: list[bool]) -> float | None:
    return sum(vals) / len(vals) if vals else None


def case_ok(cr: CaseRun, metric: str) -> bool | None:
    """All repeats passed `metric` (None when it never ran)."""
    vals = cr.passed_per_repeat(metric)
    return all(vals) if vals else None
