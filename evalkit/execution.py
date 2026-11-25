"""Running one metric safely: needs checks, engine choice, error capture."""

from __future__ import annotations

import traceback
from typing import Any

from evalkit.env.cache import CacheMiss
from evalkit.judges.runner import JudgeBlocked, JudgeParseError
from evalkit.llm.client import CostCapReached
from evalkit.registry import MetricSpec, resolve_impl
from evalkit.schema import Case, CaseRun, MetricResult, Observation
from evalkit.suites.context import MetricContext
from evalkit.targets.base import has_hook
from evalkit.util import MissingDependency, get_path, is_installed


def _field_present(obj: Any, name: str) -> bool:
    value = get_path(obj, name)
    return value not in (None, "", [], {})


def unmet_need(
    spec: MetricSpec, case: Case | None, obs: Observation | None, ctx: MetricContext
) -> str | None:
    """Skip reason for the first unmet `needs` entry, else None.

    Entries: `field` (case), `a|b` (any of), `obs.x`, `hook:x`, `pkg:module`,
    `extra.x`, `params.x`, `tag:x`.
    """
    for need in spec.needs:
        if need.startswith("pkg:"):
            mod = need[4:]
            if not is_installed(mod.split("=")[0]):
                return f"{mod.split('=')[-1]} not installed"
        elif need.startswith("hook:"):
            if not has_hook(ctx.target, need[5:]):
                return f"target has no {need[5:]} hook"
        elif need.startswith("params."):
            if ctx.p(need[7:]) is None:
                return f"n/a: needs params.{need[7:]}"
        elif case is None:
            continue
        elif need.startswith("tag:"):
            if need[4:] not in case.tags:
                return f"n/a: case not tagged {need[4:]}"
        elif need.startswith("obs."):
            if obs is not None and not _field_present(obs, need[4:]):
                return f"n/a: observation has no {need[4:]}"
        elif not any(_field_present(case, alt) for alt in need.split("|")):
            return f"n/a: needs {need.replace('|', ' or ')}"
    return None


def _guard(spec: MetricSpec, fn: Any, *args: Any) -> MetricResult | list[MetricResult]:
    try:
        out: MetricResult | list[MetricResult] = fn(*args)
        return out
    except MissingDependency as exc:
        return MetricResult(metric=spec.id, skipped=str(exc))
    except CostCapReached:
        return MetricResult(metric=spec.id, skipped="cost cap reached")
    except JudgeBlocked as exc:
        return MetricResult(metric=spec.id, skipped=str(exc))
    except JudgeParseError:
        return MetricResult(metric=spec.id, skipped="judge_parse_error")
    except CacheMiss:
        raise
    except Exception as exc:  # a broken metric must not kill the run
        return MetricResult(
            metric=spec.id,
            skipped=f"error: {type(exc).__name__}: {exc}"[:300],
            details={"traceback": traceback.format_exc(limit=4)},
        )


def run_case_metric(
    spec: MetricSpec, engine: str, case: Case, obs: Observation, ctx: MetricContext
) -> MetricResult:
    reason = unmet_need(spec, case, obs, ctx)
    if reason:
        return MetricResult(metric=spec.id, skipped=reason)
    fn = resolve_impl(spec.id, engine)
    res = _guard(spec, fn, case, obs, ctx)
    out = res[0] if isinstance(res, list) else res
    out.metric = spec.id
    if engine != "native":
        out.details.setdefault("engine", engine)
    return out


def run_dataset_metric(
    spec: MetricSpec, engine: str, rows: list[CaseRun], ctx: MetricContext
) -> list[MetricResult]:
    reason = unmet_need(spec, None, None, ctx)
    if reason:
        return [MetricResult(metric=spec.id, skipped=reason)]
    res = _guard(spec, resolve_impl(spec.id, engine), rows, ctx)
    outs = res if isinstance(res, list) else [res]
    for r in outs:
        r.metric = r.metric or spec.id
        if engine != "native":
            r.details.setdefault("engine", engine)
    return outs
