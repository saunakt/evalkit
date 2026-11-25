"""`evalctl evidence | catalog | sample-size | generate | push | pull-traces | label`."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from evalkit.cli import ConfigOpt, ProfileOpt, ProviderOpt, RoleOpt, console, project_from
from evalkit.util import read_jsonl

label_app = typer.Typer(help="Label Studio export / import")


def evidence(
    run_id: Annotated[str, typer.Argument(help="Run id or 'last'")],
    config: ConfigOpt = "evalkit.yaml",
    release: Annotated[str | None, typer.Option(help="Release name (folder)")] = None,
) -> None:
    """Write the evidence pack (Plan.md §11.4) to evidence/<release>/."""
    from evalkit.reporting.evidence import build_evidence
    from evalkit.reporting.store import load_run, resolve_run

    project = project_from(config, None, set_=[f"reports.release={release}"] if release else [])
    run_dir = resolve_run(project.path(project.config.results_dir), run_id)
    stored = load_run(run_dir)
    report_dir = project.path(project.config.reports.dir) / stored.run_id
    if not (report_dir / "summary.md").exists():
        from evalkit.cli_run import render_outcome

        render_outcome(project, run_dir)
    out = build_evidence(project, stored, report_dir)
    console.print(f"evidence pack: {out}")


def catalog(
    config: ConfigOpt = "evalkit.yaml",
    check: Annotated[bool, typer.Option("--check", help="Fail if files are stale")] = False,
) -> None:
    """Generate metrics/catalog.yaml and docs/suites.md from the registry."""
    from evalkit.catalog import stale_files, write_catalog

    root = Path(config).resolve().parent
    if check:
        stale = stale_files(root)
        for p in stale:
            console.print(f"[red]stale:[/red] {p} (run `evalctl catalog`)")
        if stale:
            raise typer.Exit(1)
        console.print("[green]✓[/green] catalog and docs are up to date")
        return
    for p in write_catalog(root):
        console.print(f"wrote {p}")


def sample_size(
    pass_rate: Annotated[float, typer.Option("--pass-rate", "-p", help="Expected pass rate")],
    margin: Annotated[float, typer.Option("--margin", "-m", help="± margin")] = 0.05,
    z: Annotated[float, typer.Option(help="z for the confidence level")] = 1.96,
) -> None:
    """Cases needed for ±margin at the expected pass rate."""
    from evalkit.stats.sample_size import cases_needed

    n = cases_needed(pass_rate, margin, z)
    console.print(f"{n} cases for ±{margin:.1%} at pass rate {pass_rate:.1%} (z={z})")


def generate(
    suite: Annotated[str, typer.Option("--suite")] = "rag",
    docs: Annotated[str, typer.Option("--docs", help="Folder of .md/.txt documents")] = "docs",
    n: Annotated[int, typer.Option("--n", help="Number of drafts")] = 20,
    config: ConfigOpt = "evalkit.yaml",
    provider: ProviderOpt = None,
    role: RoleOpt = None,
) -> None:
    """Draft cases from documents into datasets/drafts/ (review before moving to golden)."""
    from evalkit.generate import generate_drafts

    project = project_from(config, None, provider=provider, role=role)
    out = generate_drafts(project, suite, project.path(docs), n)
    console.print(
        f"{len(read_jsonl(out))} draft cases written to {out} — review them before "
        "moving any into datasets/golden/"
    )


def push(run_id: Annotated[str, typer.Argument()], config: ConfigOpt = "evalkit.yaml") -> None:
    """Push a run to Langfuse: datasets per suite, items per case, scores per metric."""
    from evalkit.engines.langfuse_engine import push_run
    from evalkit.reporting.store import load_run, resolve_run

    project = project_from(config, None)
    stored = load_run(resolve_run(project.path(project.config.results_dir), run_id))
    console.print(push_run(project, stored))


def pull_traces(
    suite: Annotated[str, typer.Option("--suite")] = "quality",
    limit: Annotated[int, typer.Option("--limit")] = 50,
    filter_: Annotated[str | None, typer.Option("--filter", help="Trace tag")] = None,
    config: ConfigOpt = "evalkit.yaml",
) -> None:
    """Fetch flagged traces from Langfuse, scrub PII, write draft cases."""
    from evalkit.engines.langfuse_engine import pull_traces as pull

    project = project_from(config, None)
    console.print(f"drafts: {pull(project, suite, limit, filter_)}")


@label_app.command("export")
def label_export(
    judge: Annotated[str | None, typer.Option("--judge")] = None,
    metric: Annotated[str | None, typer.Option("--metric")] = None,
    run: Annotated[str, typer.Option("--run")] = "last",
    out: Annotated[str, typer.Option("--out")] = "labelstudio",
    config: ConfigOpt = "evalkit.yaml",
    profile: ProfileOpt = None,
) -> None:
    """Export outputs of a run for labelling (binary PASS/FAIL, or 1-5 MOS for --metric)."""
    from evalkit.engines.labelstudio import export_tasks
    from evalkit.reporting.store import load_run, resolve_run

    project = project_from(config, profile)
    stored = load_run(
        resolve_run(project.path(project.config.results_dir), run), with_observations=True
    )
    items = [
        {
            "id": f"{o['suite']}/{o['case_id']}",
            "output": o["observation"]["text"],
            "input": (
                c.text_input() if (c := stored.cases.get(f"{o['suite']}/{o['case_id']}")) else ""
            ),
        }
        for o in stored.observations
        if o.get("repeat", 0) == 0
    ]
    criterion = "Rate the output."
    if judge:
        from evalkit.config import load_judge

        criterion = load_judge(project.path(f"judges/{judge}.yaml")).criterion
    paths = export_tasks(
        items, project.path(out) / (judge or metric or "items"), criterion, mos=bool(metric)
    )
    console.print({k: str(v) for k, v in paths.items()})


@label_app.command("import")
def label_import(
    export_file: Annotated[str, typer.Argument()],
    judge: Annotated[str, typer.Option("--judge")],
    split: Annotated[str, typer.Option("--split")] = "dev",
    config: ConfigOpt = "evalkit.yaml",
) -> None:
    """Import a Label Studio export into datasets/human_labels/<judge>.jsonl."""
    from evalkit.engines.labelstudio import import_labels

    project = project_from(config, None)
    out = project.path(f"datasets/human_labels/{judge}.jsonl")
    console.print(import_labels(Path(export_file), out, split))


def setup(
    config: ConfigOpt = "evalkit.yaml",
    profile: ProfileOpt = None,
    install_file: Annotated[str, typer.Option("--install-file")] = "install.yaml",
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Print the plan; run nothing")] = False,
    check: Annotated[bool, typer.Option("--check", help="Report state; install nothing")] = False,
    no_external: Annotated[bool, typer.Option("--no-external")] = False,
    no_lock: Annotated[bool, typer.Option("--no-lock")] = False,
) -> None:
    """Install what install.yaml + evalkit.yaml need (extras via uv, external tools)."""
    from evalkit.install.apply import setup as run_setup
    from evalkit.install.config import load_install
    from evalkit.install.report import print_report, write_report

    project = project_from(config, profile)
    cfg = load_install(project.root / install_file)
    report = run_setup(
        project, cfg, dry_run=dry_run, check=check, no_external=no_external, no_lock=no_lock
    )
    print_report(report, console)
    if not dry_run:
        console.print(f"report: {write_report(report, project.root)}")
    raise typer.Exit(report.exit_code)


def register(app: typer.Typer) -> None:
    app.command("evidence")(evidence)
    app.command("catalog")(catalog)
    app.command("sample-size")(sample_size)
    app.command("generate")(generate)
    app.command("push")(push)
    app.command("pull-traces")(pull_traces)
    app.command("setup")(setup)
    app.add_typer(label_app, name="label")
