"""Classic software-test metrics (Plan.md §9.18) + a shared pytest/JUnit runner."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import jsonschema

from evalkit.env.cache import CacheMiss, ResponseCache
from evalkit.metrics.common import na, result
from evalkit.registry import metric
from evalkit.schema import Case, CaseRun, MetricResult, Observation
from evalkit.suites.context import MetricContext
from evalkit.util import import_string

CLASSIC = ("classic",)


def run_pytest(
    target: str | list[str], root: Path, env: dict[str, str] | None = None, timeout_s: float = 600
) -> dict[str, Any]:
    """Run pytest in a subprocess and import its JUnit XML."""
    targets = [target] if isinstance(target, str) else list(target)
    with tempfile.TemporaryDirectory(prefix="evalkit-junit-") as tmp:
        xml_path = Path(tmp) / "junit.xml"
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            f"--junitxml={xml_path}",
            *targets,
        ]
        proc = subprocess.run(
            cmd,
            cwd=root,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env={**os.environ, **(env or {})},
        )
        if not xml_path.exists():
            return {
                "tests": 0,
                "failures": 1,
                "n_passed": 0,
                "failed": ["(pytest did not run)"],
                "output": (proc.stdout + proc.stderr)[-2000:],
                "returncode": proc.returncode,
            }
        return parse_junit(xml_path) | {"returncode": proc.returncode}


def parse_junit(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    cases = list(root.iter("testcase"))
    failed = [
        f"{c.get('classname')}::{c.get('name')}"
        for c in cases
        if c.find("failure") is not None or c.find("error") is not None
    ]
    skipped = sum(1 for c in cases if c.find("skipped") is not None)
    tests = len(cases) - skipped
    return {
        "tests": tests,
        "failures": len(failed),
        "n_passed": tests - len(failed),
        "skipped": skipped,
        "failed": failed,
    }


def pytest_result(
    target: str | None, ctx: MetricContext, env: dict[str, str] | None = None
) -> MetricResult:
    if not target:
        return na("no tests configured")
    root = ctx.suite.project.root
    if not (root / target.split("::")[0]).exists():
        return na(f"{target} not found")
    rep = run_pytest(target, root, env)
    if rep["tests"] == 0:
        return result(0.0, False, **rep)
    return result(rep["n_passed"] / rep["tests"], rep["failures"] == 0, **rep)


@metric("unit_fake_llm", suites=CLASSIC, kind="deterministic", scope="dataset")
def unit_fake_llm(rows: list[CaseRun], ctx: MetricContext) -> MetricResult:
    """Run `pytest <params.unit_tests>` with EVALKIT_FAKE=1 and import the JUnit result."""
    return pytest_result(ctx.p("unit_tests"), ctx, {"EVALKIT_FAKE": "1"})


@metric(
    "ui_streaming",
    requires_extra="ui",
    suites=CLASSIC,
    kind="deterministic",
    scope="dataset",
    needs=("pkg:playwright",),
)
def ui_streaming(rows: list[CaseRun], ctx: MetricContext) -> MetricResult:
    """Run the user's Playwright tests (params.ui_tests) via pytest; import JUnit."""
    return pytest_result(ctx.p("ui_tests"), ctx)


def _replay_pass(
    cases: list[Case], ctx: MetricContext, mode: str, path: Path
) -> tuple[list[str], int]:
    """Call the target for each case with the runtime cache in `mode`; (texts, live calls)."""
    rt = ctx.suite.runtime
    old_cache, old_mode = rt._cache, rt.cache_mode
    rt._cache, rt.cache_mode = ResponseCache(path, mode), mode  # type: ignore[arg-type]
    before = sum(t.get("calls", 0) - t.get("cached_calls", 0) for t in rt.ledger.summary().values())
    try:
        texts = []
        for case in cases:
            ctx.suite.reset(ctx.seed)
            obs = ctx.suite.call(case)
            if obs.error and "CacheMiss" in obs.error:
                raise CacheMiss(obs.error)
            texts.append(obs.text)
    finally:
        rt._cache.close()
        rt._cache, rt.cache_mode = old_cache, old_mode
    after = sum(t.get("calls", 0) - t.get("cached_calls", 0) for t in rt.ledger.summary().values())
    return texts, after - before


@metric("record_replay", suites=CLASSIC, kind="deterministic", scope="dataset")
def record_replay(rows: list[CaseRun], ctx: MetricContext) -> MetricResult:
    """Record a pass, replay it in replay_only mode: identical outputs and no live calls."""
    cases = [cr.case for cr in rows][: int(ctx.p("record_replay_cases", 5))]
    if not cases:
        return na("no cases")
    with tempfile.TemporaryDirectory(prefix="evalkit-rr-") as tmp:
        path = Path(tmp) / "cache.sqlite"
        recorded, _ = _replay_pass(cases, ctx, "read_write", path)
        try:
            replayed, live = _replay_pass(cases, ctx, "replay_only", path)
        except CacheMiss as exc:
            return result(0.0, False, reason=f"replay miss: {exc}")
    diffs = [c.id for c, a, b in zip(cases, recorded, replayed, strict=True) if a != b]
    ok = not diffs and live == 0
    return result(1.0 if ok else 0.0, ok, cases=len(cases), differing=diffs, live_calls=live)


def _tool_schemas(spec: Any, root: Path) -> dict[str, Any]:
    if isinstance(spec, dict):
        return spec
    data: dict[str, Any] = json.loads((root / str(spec)).read_text(encoding="utf-8"))
    return data


@metric("contract", suites=CLASSIC, kind="deterministic", scope="dataset")
def contract(rows: list[CaseRun], ctx: MetricContext) -> MetricResult:
    """Responses validate against params.response_schema; tool schemas are valid JSON Schema
    and match what the SUT advertises (params.advertised: "module:function")."""
    problems: list[str] = []
    checked = 0
    schema = ctx.p("response_schema")
    if schema:
        for cr in rows:
            for obs in cr.observations:
                checked += 1
                problems += _validate_response(obs, schema, cr.case.id)
    tools = ctx.p("tool_schemas")
    if tools:
        declared = _tool_schemas(tools, ctx.suite.project.root)
        for name, sch in sorted(declared.items()):
            checked += 1
            try:
                jsonschema.Draft202012Validator.check_schema(sch)
            except jsonschema.SchemaError as exc:
                problems.append(f"tool {name}: invalid schema: {exc.message}")
        if ctx.p("advertised"):
            advertised = import_string(str(ctx.p("advertised")))()
            if sorted(advertised) != sorted(declared):
                problems.append(
                    f"advertised tools {sorted(advertised)} != declared {sorted(declared)}"
                )
    if not checked:
        return na("needs params.response_schema or params.tool_schemas")
    return result(
        1.0 - len(problems) / checked if checked else 0.0,
        not problems,
        checked=checked,
        problems=problems[:20],
    )


def _validate_response(obs: Observation, schema: dict[str, Any], cid: str) -> list[str]:
    try:
        jsonschema.validate(obs.model_dump(mode="json"), schema)
    except jsonschema.ValidationError as exc:
        return [f"{cid}: {exc.message}"]
    return []


@metric("api", suites=CLASSIC, kind="deterministic", scope="dataset")
def api(rows: list[CaseRun], ctx: MetricContext) -> MetricResult:
    """HTTP targets: empty / oversize / wrong content-type / missing auth → expected statuses."""
    import httpx

    target = ctx.target
    if not hasattr(target, "url"):
        return na("target is not an http target")
    expected = {
        "empty": 400,
        "oversize": 413,
        "content_type": 415,
        "no_auth": 401,
        **dict(ctx.p("expected_status") or {}),
    }
    big = "x" * (int(ctx.p("max_chars", 4000)) + 1)
    probes = {
        "empty": {"json": {"input": ""}, "headers": target.headers},
        "oversize": {"json": {"input": big}, "headers": target.headers},
        "content_type": {
            "content": "plain",
            "headers": {**target.headers, "Content-Type": "text/plain"},
        },
        "no_auth": {
            "json": {"input": "hi"},
            "headers": {k: v for k, v in target.headers.items() if k.lower() != "authorization"},
        },
    }
    got: dict[str, int] = {}
    errors: dict[str, str] = {}
    for name, kw in probes.items():
        try:
            got[name] = httpx.request(target.method, target.url, timeout=30, **kw).status_code
        except httpx.HTTPError as exc:
            got[name], errors[name] = -1, str(exc)[:200]
    bad = {k: (got.get(k), v) for k, v in expected.items() if got.get(k) != v}
    return result(
        1 - len(bad) / len(expected), not bad, statuses=got, mismatches=bad, errors=errors
    )
