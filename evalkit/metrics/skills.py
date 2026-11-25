"""Agent-skill metrics (Plan.md §9.8)."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from evalkit.metrics.agent import is_subsequence
from evalkit.metrics.common import boolean, contains_all, na, result
from evalkit.registry import metric
from evalkit.schema import Case, CaseRun, MetricResult, Observation
from evalkit.suites.context import MetricContext
from evalkit.util import is_installed, mean

SKILLS = ("skills",)
RISKY = {
    "network": r"\b(requests\.|httpx\.|urllib|socket\.|curl |wget )",
    "exfiltration": r"(?i)(upload|exfil|pastebin|webhook\.site)",
    "env_access": r"os\.environ|getenv\(|\.env\b",
    "shell": r"subprocess|os\.system|eval\(|exec\(",
}


def skills_loaded(obs: Observation) -> list[str]:
    names = [str(e.get("skill")) for e in obs.events if e.get("type") == "skill_loaded"]
    names += [
        str(tc.args.get("name", tc.args.get("skill", "")))
        for tc in obs.tool_calls
        if tc.name == "Skill"
    ]
    return names


def triggered(case: Case, obs: Observation) -> bool:
    return str(case.extra.get("skill")) in skills_loaded(obs)


def _rate(rows: list[CaseRun], want: bool) -> tuple[float | None, list[str]]:
    sel = [cr for cr in rows if cr.case.extra.get("should_trigger") is want]
    flags = [(cr.case.id, triggered(cr.case, o)) for cr in sel for o in cr.observations]
    if not flags:
        return None, []
    hits = [cid for cid, t in flags if t]
    return len(hits) / len(flags), hits


@metric("trigger_rate", suites=SKILLS, kind="deterministic", scope="dataset")
def trigger_rate(rows: list[CaseRun], ctx: MetricContext) -> MetricResult:
    """should_trigger cases: share where skill_loaded (or a Skill tool call) names extra.skill."""
    value, hits = _rate(rows, True)
    if value is None:
        return na("no should_trigger: true cases")
    return result(value, value >= float(ctx.p("min_trigger_rate", 0.9)), triggered=hits)


@metric(
    "false_trigger_rate", suites=SKILLS, kind="deterministic", scope="dataset", direction="lower"
)
def false_trigger_rate(rows: list[CaseRun], ctx: MetricContext) -> MetricResult:
    """should_trigger: false cases that triggered the skill."""
    value, hits = _rate(rows, False)
    if value is None:
        return na("no should_trigger: false cases")
    return result(value, value <= float(ctx.p("max_false_trigger_rate", 0.1)), false_triggers=hits)


@metric("execution_steps", suites=SKILLS, kind="deterministic", needs=("extra.expected_steps",))
def execution_steps(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """trajectory_in_order over extra.expected_steps (e.g. Read template → run script)."""
    expected = [str(s) for s in case.extra["expected_steps"]]
    actual = [tc.name for tc in obs.tool_calls]
    return boolean(is_subsequence(expected, actual), expected=expected, actual=actual)


def file_checks(case: Case, ctx: MetricContext) -> dict[str, Any]:
    """Best-effort checks on extra.output_file via optional readers (docx/xlsx/pptx/text)."""
    path = Path(str(case.extra.get("output_file")))
    if not path.is_absolute():
        path = ctx.suite.project.root / path
    if not path.exists():
        return {"exists": False}
    text = (
        path.read_text(encoding="utf-8", errors="ignore")
        if path.suffix in (".txt", ".md", ".csv", ".json")
        else ""
    )
    missing = contains_all(text, list(case.extra.get("file_checks", {}).get("contains", [])))
    return {"exists": True, "missing": missing}


@metric("output_quality", suites=SKILLS, kind="judge")
def output_quality(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """Deterministic checks + judge_binary on criteria (+ file checks when output_file set)."""
    parts: dict[str, bool] = {}
    if case.must_include:
        parts["contains"] = not contains_all(obs.text, case.must_include)
    if case.criteria:
        from evalkit.metrics.judged import per_criterion

        parts["criteria"] = bool(
            per_criterion(ctx.judge, "generic_binary", case.criteria, case, obs).passed
        )
    files = file_checks(case, ctx) if case.extra.get("output_file") else None
    if files is not None:
        parts["file"] = files["exists"] and not files.get("missing")
    if not parts:
        return na("no checks declared")
    return boolean(all(parts.values()), parts=parts, file=files)


@metric("with_without_delta", suites=SKILLS, kind="statistical", scope="dataset")
def with_without_delta(rows: list[CaseRun], ctx: MetricContext) -> MetricResult:
    """Re-run with skill_enabled=false: paired quality delta plus cost/time deltas."""
    from evalkit.execution import run_case_metric
    from evalkit.registry import METRICS

    quality = str(ctx.p("quality_metric", "output_quality"))
    pairs, dt, dc = [], [], []
    for cr in [cr for cr in rows if cr.case.extra.get("should_trigger")][:10]:
        off = ctx.suite.recall(cr.case, params={"skill_enabled": False})
        on_res = [r for r in cr.by_metric(quality) if r.passed is not None]
        off_res = run_case_metric(METRICS.get(quality), "native", cr.case, off, ctx)
        if on_res and off_res.passed is not None:
            pairs.append((float(bool(on_res[0].passed)), float(off_res.passed)))
            dt.append(cr.observations[0].latency_s - off.latency_s)
            dc.append((cr.observations[0].cost_usd or 0) - (off.cost_usd or 0))
    if not pairs:
        return na(f"no {quality} results to compare")
    delta = mean([a - b for a, b in pairs]) or 0.0
    return result(
        delta,
        delta >= 0,
        quality_with=mean([a for a, _ in pairs]),
        quality_without=mean([b for _, b in pairs]),
        latency_delta=mean(dt),
        cost_delta=mean(dc),
        n=len(pairs),
    )


@metric("skill_conflict", suites=SKILLS, kind="deterministic", needs=("extra.other_skill",))
def skill_conflict(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """Overlap-zone cases: the expected skill fired and the other one did not."""
    loaded = skills_loaded(obs)
    ok = str(case.extra.get("skill")) in loaded and str(case.extra["other_skill"]) not in loaded
    return boolean(ok, loaded=loaded)


@metric(
    "third_party_scan",
    suites=SKILLS,
    kind="deterministic",
    scope="dataset",
    needs=("params.skill_dirs",),
)
def third_party_scan(rows: list[CaseRun], ctx: MetricContext) -> MetricResult:
    """Static scan of skill folders: scripts, risky patterns, Bandit on Python (if installed)."""
    findings: list[dict[str, str]] = []
    scripts: list[str] = []
    for d in ctx.p("skill_dirs"):
        root = ctx.suite.project.path(str(d))
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            if path.suffix in (".py", ".sh", ".js", ".ps1"):
                scripts.append(path.relative_to(root).as_posix())
            text = path.read_text(encoding="utf-8", errors="ignore")
            for kind, rx in RISKY.items():
                if re.search(rx, text):
                    findings.append({"file": path.relative_to(root).as_posix(), "kind": kind})
            if path.suffix == ".py" and is_installed("bandit"):
                proc = subprocess.run(
                    [sys.executable, "-m", "bandit", "-q", str(path)],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if proc.returncode:
                    findings.append({"file": path.relative_to(root).as_posix(), "kind": "bandit"})
    return result(float(len(findings)), not findings, scripts=scripts, findings=findings)
