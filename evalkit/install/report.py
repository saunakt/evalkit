"""`evalctl setup` summary: CORE · EXTRAS · EXTERNAL table + .evalkit/setup_report.json."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from evalkit.install.external import ToolStatus

MARK = {
    "installed": "✓ installed",
    "found": "✓ found",
    "running": "✓ running",
    "missing": "✗ missing",
    "failed": "✗ failed",
    "planned": "→ planned",
    "off": "– off",
    "skipped": "– skipped",
    "already": "✓ already installed",
    "selected": "→ install",
    "not selected": "– skipped",
}


@dataclass
class SetupReport:
    mode: str  # apply | check | dry-run
    profile: str | None
    core: list[str]
    extras: dict[str, str]  # extra -> already | installed | selected | missing | not selected
    extra_reasons: dict[str, list[str]]
    tools: list[ToolStatus]
    commands: list[list[str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    lock_file: str | None = None
    external_skipped: bool = False

    @property
    def required_missing(self) -> list[str]:
        out = [f"extra {e}" for e, s in self.extras.items() if s in ("missing", "failed")]
        out += [f"tool {t.name}" for t in self.tools if t.required_missing]
        return out

    @property
    def exit_code(self) -> int:
        return 1 if self.required_missing and self.mode != "dry-run" else 0

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["required_missing"] = self.required_missing
        data["exit_code"] = self.exit_code
        return data


def write_report(report: SetupReport, root: Path) -> Path:
    path = root / ".evalkit" / "setup_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.as_dict(), indent=2, default=str), encoding="utf-8")
    return path


def print_report(report: SetupReport, console: Console) -> None:
    table = Table("section", "item", "status", "details", title=f"evalctl setup ({report.mode})")
    table.add_row(
        "CORE",
        f"{len(report.core)} packages",
        "✓ always installed",
        ", ".join(report.core[:6]) + (" …" if len(report.core) > 6 else ""),
    )
    for extra, state in report.extras.items():
        reasons = report.extra_reasons.get(extra, [])
        table.add_row(
            "EXTRAS",
            extra,
            MARK.get(state, state),
            ("needed by " + ", ".join(reasons[:3])) if reasons else "",
        )
    if report.external_skipped:
        table.add_row("EXTERNAL", "(all)", "– skipped", "--no-external")
    for t in report.tools:
        detail = t.detail
        if t.state in ("missing", "failed"):
            detail = f"{detail} → {t.manual}".strip(" →")
        if t.needed_by:
            detail = f"[needed by {', '.join(t.needed_by)}] {detail}"
        elif t.state in ("missing", "failed"):
            detail = f"[not needed by the current config] {detail}"
        table.add_row("EXTERNAL", t.name, f"{MARK.get(t.state, t.state)} ({t.action})", detail)
    console.print(table)
    for cmd in report.commands:
        console.print(("would run: " if report.mode == "dry-run" else "ran: ") + " ".join(cmd))
    for note in report.notes:
        console.print(f"[cyan]note:[/cyan] {note}")
    for w in report.warnings:
        console.print(f"[yellow]WARNING:[/yellow] {w}")
    if report.lock_file:
        console.print(f"lock: {report.lock_file}")
    if report.required_missing:
        console.print(f"[red]required but missing:[/red] {', '.join(report.required_missing)}")
