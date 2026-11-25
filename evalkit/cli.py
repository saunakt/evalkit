"""`evalctl` command-line interface (Typer)."""

from __future__ import annotations

import sys
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from evalkit.config import Overrides, Project, load_project
from evalkit.util import EvalkitError

app = typer.Typer(
    add_completion=False, no_args_is_help=True, help="evalkit: evaluate GenAI systems"
)
console = Console()

ConfigOpt = Annotated[str, typer.Option("--config", "-c", help="Path to evalkit.yaml")]
ProfileOpt = Annotated[str | None, typer.Option("--profile", "-p", help="Profile overlay name")]
ProviderOpt = Annotated[str | None, typer.Option("--provider", help="Map every role to a provider")]
SuitesOpt = Annotated[str | None, typer.Option("--suites", help="Comma-separated suites")]
MetricsOpt = Annotated[str | None, typer.Option("--metrics", help="Comma-separated metric ids")]
TagsOpt = Annotated[str | None, typer.Option("--tags", help="Comma-separated tag filter")]
SplitOpt = Annotated[str | None, typer.Option("--split", help="dev, heldout or dev,heldout")]
RepeatsOpt = Annotated[int | None, typer.Option("--repeats", help="Override run.repeats")]
RoleOpt = Annotated[list[str] | None, typer.Option("--role", help="role=provider:model")]
SetOpt = Annotated[list[str] | None, typer.Option("--set", help="key.path=value")]


def _csv(value: str | None) -> list[str] | None:
    return [v.strip() for v in value.split(",") if v.strip()] if value else None


def project_from(
    config: str,
    profile: str | None,
    provider: str | None = None,
    suites: str | None = None,
    metrics: str | None = None,
    tags: str | None = None,
    split: str | None = None,
    repeats: int | None = None,
    role: list[str] | None = None,
    set_: list[str] | None = None,
) -> Project:
    ov = Overrides(
        suites=_csv(suites),
        metrics=_csv(metrics),
        tags=_csv(tags),
        split=split,
        repeats=repeats,
        provider=provider,
        roles=role or [],
        sets=set_ or [],
    )
    return load_project(config, profile, ov)


# --------------------------------------------------------------------------- validate / list


@app.command()
def validate(config: ConfigOpt = "evalkit.yaml", profile: ProfileOpt = None) -> None:
    """Validate config, targets, datasets and judges without running anything."""
    from evalkit.planner import validate_project

    project = project_from(config, profile)
    errors = validate_project(project)
    if errors:
        for err in errors:
            console.print(f"[red]✗[/red] {err}")
        raise typer.Exit(2)
    console.print(f"[green]✓[/green] {config} is valid ({len(project.enabled_suites())} suites on)")
    install_file = project.root / "install.yaml"
    if install_file.exists():
        from evalkit.install.config import load_install
        from evalkit.install.resolve import validate_warnings

        for w in validate_warnings(project, load_install(install_file)):
            console.print(f"[yellow]WARNING:[/yellow] {w}")


@app.command("list")
def list_cmd(what: Annotated[str, typer.Argument(help="suites | metrics | engines")]) -> None:
    """List registered suites, metrics or engines."""
    from evalkit.registry import ENGINES, METRICS, SUITES, load_all

    load_all()
    if what == "suites":
        for name in SUITES.names():
            doc = (SUITES.get(name).__doc__ or "").strip().split("\n")[0]
            console.print(f"{name:16} {doc}")
    elif what == "metrics":
        table = Table("metric", "suites", "kind", "scope", "dir", "unit")
        for mid, spec in METRICS.items():
            table.add_row(
                mid, ",".join(spec.suites), spec.kind, spec.scope, spec.direction, spec.unit
            )
        console.print(table)
        console.print(f"{len(METRICS)} metrics")
    elif what == "engines":
        for name in ENGINES.names():
            console.print(name)
    else:
        console.print(f"[red]unknown list target {what!r}: use suites, metrics or engines[/red]")
        raise typer.Exit(2)


def _register_subcommands() -> None:
    """Attach command groups defined in other modules (kept small per file)."""
    from evalkit import cli_commands

    cli_commands.register(app)


def _utf8_console() -> None:
    """Windows consoles default to cp1252; reports contain ₹ and Indic scripts."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def main(argv: list[str] | None = None) -> Any:
    _utf8_console()
    _register_subcommands()
    try:
        return app(args=argv, standalone_mode=True)
    except EvalkitError as exc:
        console.print(f"[red]error:[/red] {exc}")
        sys.exit(exc.exit_code)


if __name__ == "__main__":
    main()
