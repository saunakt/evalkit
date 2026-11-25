"""`evalctl` commands beyond validate/list: providers, plan, run, report, calibrate, baseline."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.table import Table

from evalkit.cli import ConfigOpt, ProfileOpt, ProviderOpt, RoleOpt, console, project_from

providers_app = typer.Typer(help="Provider checks")
baseline_app = typer.Typer(help="Baseline management")


@providers_app.command("check")
def providers_check(
    config: ConfigOpt = "evalkit.yaml",
    profile: ProfileOpt = None,
    provider: ProviderOpt = None,
    role: RoleOpt = None,
    all_: Annotated[bool, typer.Option("--all", help="Also check unused providers")] = False,
) -> None:
    """Probe each provider used by a role: reachable, latency, free/paid, JSON mode."""
    from evalkit.llm.check import check_providers
    from evalkit.llm.roles import LLMRuntime

    project = project_from(config, profile, provider=provider, role=role)
    statuses = check_providers(LLMRuntime(project.config, project.root), include_unused=all_)
    table = Table("provider", "type", "reachable", "latency_s", "cost", "json", "roles", "detail")
    for s in statuses:
        table.add_row(
            s.name,
            s.type,
            "yes" if s.reachable else "[red]no[/red]",
            f"{s.latency_s:.3f}" if s.latency_s is not None else "-",
            "free" if s.free else "paid",
            "yes" if s.supports_json else "no",
            ",".join(s.roles) or "-",
            s.detail,
        )
    console.print(table)
    if any(not s.reachable and s.roles for s in statuses):
        raise typer.Exit(2)


def calibrate(
    judge_id: Annotated[str, typer.Argument(help="Judge id (judges/<id>.yaml)")],
    config: ConfigOpt = "evalkit.yaml",
    profile: ProfileOpt = None,
    provider: ProviderOpt = None,
    role: RoleOpt = None,
) -> None:
    """Run a judge on human labels; write judges/calibration/<id>@v<version>.json."""
    from evalkit.judges.calibration import calibrate as run_calibration

    project = project_from(config, profile, provider=provider, role=role)
    rec = run_calibration(project, judge_id)
    status = "[green]PASS[/green]" if rec["passed"] else "[red]FAIL[/red]"
    console.print(
        f"{rec['judge_id']} v{rec['version']} on {rec['provider']}:{rec['model']}: {status} "
        f"kappa={rec['kappa']:.3f} fail_recall={rec['fail_recall']:.3f} (n={rec['n']})"
    )
    console.print(f"record: judges/calibration/{rec['judge_id']}@v{rec['version']}.json")


@baseline_app.command("set")
def baseline_set(
    run_id: Annotated[str, typer.Argument(help="Run id or 'last'")],
    config: ConfigOpt = "evalkit.yaml",
) -> None:
    """Mark a run as the regression baseline (thresholds.regression.baseline: auto)."""
    from evalkit.reporting.store import resolve_run, set_baseline

    project = project_from(config, None)
    results = project.path(project.config.results_dir)
    rid = resolve_run(results, run_id).name
    set_baseline(results, rid)
    console.print(f"baseline set to {rid}")


def register(app: typer.Typer) -> None:
    """Attach command groups to the main app (idempotent)."""
    if getattr(app, "_evalkit_registered", False):
        return
    app._evalkit_registered = True  # type: ignore[attr-defined]
    app.add_typer(providers_app, name="providers")
    app.add_typer(baseline_app, name="baseline")
    app.command("calibrate")(calibrate)
    from evalkit import cli_run

    cli_run.register(app)
    from evalkit import cli_tools

    cli_tools.register(app)
