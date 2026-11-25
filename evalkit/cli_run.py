"""`evalctl plan | run | report`."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer
from rich.table import Table

from evalkit.cli import (
    ConfigOpt,
    MetricsOpt,
    ProfileOpt,
    ProviderOpt,
    RepeatsOpt,
    RoleOpt,
    SetOpt,
    SplitOpt,
    SuitesOpt,
    TagsOpt,
    _csv,
    console,
    project_from,
)
from evalkit.config import Project
from evalkit.planner import RunPlan


def print_plan(plan: RunPlan) -> None:
    roles = sorted(plan.calls)
    table = Table("suite", "metrics", "cases", "repeats", *[f"{r} calls" for r in roles])
    for sp in plan.suites:
        table.add_row(
            sp.name,
            str(len(sp.metric_ids)),
            str(len(sp.cases)),
            str(sp.repeats),
            *[str(sp.calls.get(r, 0)) for r in roles],
        )
    console.print(table)
    for sp in plan.suites:
        console.print(f"[bold]{sp.name}[/bold]: {', '.join(sp.metric_ids)}")
    paid = plan.paid_cost
    paid_txt = "unknown (no price)" if paid is None else f"${paid:.2f}"
    paid_roles = ", ".join(plan.paid_roles) or "none"
    console.print(
        f"paid: {plan.paid_calls} calls ({paid_roles}) ≈ {paid_txt} · free: {plan.free_calls} calls"
    )
    for note in plan.notes:
        console.print(f"[cyan]note:[/cyan] {note}")
    for w in plan.warnings:
        console.print(f"[yellow]warning:[/yellow] {w}")


def _project(
    config: str,
    profile: str | None,
    provider: str | None,
    suites: str | None,
    metrics: str | None,
    tags: str | None,
    split: str | None,
    repeats: int | None,
    role: list[str] | None,
    set_: list[str] | None,
) -> Project:
    return project_from(
        config, profile, provider, suites, metrics, tags, split, repeats, role, set_
    )


def plan(
    config: ConfigOpt = "evalkit.yaml",
    profile: ProfileOpt = None,
    provider: ProviderOpt = None,
    suites: SuitesOpt = None,
    metrics: MetricsOpt = None,
    tags: TagsOpt = None,
    split: SplitOpt = None,
    repeats: RepeatsOpt = None,
    role: RoleOpt = None,
    set_: SetOpt = None,
) -> None:
    """Validate and show what a run would do: suites × metrics × cases × calls × cost."""
    from evalkit.planner import build_plan

    project = _project(config, profile, provider, suites, metrics, tags, split, repeats, role, set_)
    print_plan(build_plan(project))


def run(
    config: ConfigOpt = "evalkit.yaml",
    profile: ProfileOpt = None,
    provider: ProviderOpt = None,
    suites: SuitesOpt = None,
    metrics: MetricsOpt = None,
    tags: TagsOpt = None,
    split: SplitOpt = None,
    repeats: RepeatsOpt = None,
    role: RoleOpt = None,
    set_: SetOpt = None,
) -> None:
    """Plan, run, store results, render reports and apply the gate (exit 0/1/2/3)."""
    from evalkit.planner import build_plan
    from evalkit.runner import run as run_plan

    project = _project(config, profile, provider, suites, metrics, tags, split, repeats, role, set_)
    the_plan = build_plan(project)
    print_plan(the_plan)
    try:
        outcome = run_plan(project, the_plan)
    except typer.Exit:
        raise
    except Exception as exc:
        from evalkit.util import EvalkitError

        if isinstance(exc, EvalkitError):
            raise
        console.print_exception(max_frames=5)
        raise typer.Exit(3) from exc
    out_dir = render_outcome(project, outcome.run_dir)
    print_outcome(outcome, out_dir)
    raise typer.Exit(outcome.gate.exit_code)


def render_outcome(project: Project, run_dir: Path, **kw: Any) -> Path:
    from evalkit.reporting.render import render_run
    from evalkit.reporting.store import find_baseline, load_run

    cfg = project.config
    stored = load_run(run_dir)
    baseline = find_baseline(run_dir.parent, project.thresholds.regression.baseline, stored.run_id)
    out_dir = kw.pop("out_dir", None) or project.path(cfg.reports.dir) / stored.run_id
    render_run(
        stored,
        project.thresholds,
        out_dir,
        baseline=baseline,
        formats=kw.pop("formats", cfg.reports.formats),
        trace_url_template=cfg.observe.trace_url_template,
        **kw,
    )
    if cfg.reports.evidence_pack and not kw:
        from evalkit.reporting.evidence import build_evidence

        build_evidence(project, stored, out_dir)
    return Path(out_dir)


def print_outcome(outcome: Any, out_dir: Path) -> None:
    table = Table("suite", "status", "cases", "pass rate", "failures / warnings")
    for name, sg in outcome.gate.suites.items():
        s = outcome.summaries[name]
        rate = s.overall["pass_rate"]
        colour = {"PASS": "green", "FAIL": "red", "WARN": "yellow"}[sg.status]
        msgs = [f"[red]{f}[/red]" for f in sg.failures[:6]] + sg.warnings[:3]
        table.add_row(
            name,
            f"[{colour}]{sg.status}[/{colour}]",
            str(s.n_cases),
            "–" if rate is None else f"{rate:.1%}",
            "\n".join(msgs),
        )
    console.print(table)
    for f in outcome.gate.must_pass_failures[:20]:
        console.print(
            f"[red]must-pass failed:[/red] {f['suite']}/{f['case_id']} {f['metric']} "
            f"(repeat {f['repeat']})"
        )
    led = outcome.info
    console.print(
        f"run {outcome.run_id} · paid spend ${led.paid_cost_usd:.4f} · tokens {led.total_tokens}"
    )
    console.print(f"reports: {out_dir / 'summary.html'}")
    verdict = "[green]GATE PASSED[/green]" if outcome.gate.passed else "[red]GATE FAILED[/red]"
    console.print(verdict)


def report(
    run_id: Annotated[str, typer.Argument(help="Run id or 'last'")],
    config: ConfigOpt = "evalkit.yaml",
    suites: SuitesOpt = None,
    metrics: MetricsOpt = None,
    format_: Annotated[str | None, typer.Option("--format", help="html,md,json,junit,csv")] = None,
    out: Annotated[str | None, typer.Option("--out", help="Output directory")] = None,
) -> None:
    """Re-render any subset of a stored run without re-running it.

    Subsets go to reports/<run_id>/subset/ (or --out) so the full report is never replaced.
    """
    from evalkit.reporting.render import render_run
    from evalkit.reporting.store import find_baseline, load_run, resolve_run

    project = project_from(config, None)
    run_dir = resolve_run(project.path(project.config.results_dir), run_id)
    formats = _csv(format_) or list(project.config.reports.formats)
    base = project.path(project.config.reports.dir) / run_dir.name
    subset = bool(suites or metrics)
    out_dir = Path(out) if out else (base / "subset" if subset else base)
    stored = load_run(run_dir)
    baseline = find_baseline(run_dir.parent, project.thresholds.regression.baseline, stored.run_id)
    written = render_run(
        stored,
        project.thresholds,
        out_dir,
        baseline=baseline,
        suites=_csv(suites),
        metrics=_csv(metrics),
        formats=formats,
        trace_url_template=project.config.observe.trace_url_template,
    )
    for path in written:
        console.print(str(path))


def register(app: typer.Typer) -> None:
    app.command("plan")(plan)
    app.command("run")(run)
    app.command("report")(report)
