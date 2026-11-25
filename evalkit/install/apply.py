"""`evalctl setup`: resolve install.yaml + evalkit.yaml, install, report (Plan.md §13.1).

Every subprocess goes through `run_cmd()` so tests can mock it.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable
from importlib import metadata
from pathlib import Path
from typing import Any

from evalkit.config import Project
from evalkit.install.config import (
    EXTRA_NAMES,
    TOOL_NAMES,
    InstallConfig,
    core_packages,
    extra_packages,
)
from evalkit.install.external import ToolStatus, handle
from evalkit.install.report import SetupReport
from evalkit.install.resolve import Resolution, resolve


def run_cmd(cmd: list[str], cwd: Path | None = None, timeout: float = 3600) -> Any:
    """The only place `evalctl setup` starts a process."""
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)


def is_dist_installed(name: str) -> bool:
    try:
        metadata.version(name)
    except metadata.PackageNotFoundError:
        return False
    return True


def installed_extras(packages: dict[str, list[str]]) -> set[str]:
    """Extras whose packages are all present (importlib.metadata)."""
    return {
        extra
        for extra, pkgs in packages.items()
        if pkgs and all(is_dist_installed(p) for p in pkgs)
    }


def sync_command(extras: list[str]) -> list[str]:
    """One uv call; --inexact keeps what is installed, --no-install-project leaves the
    running evalkit alone (Windows locks evalctl.exe while it runs)."""
    cmd = ["uv", "sync", "--inexact", "--no-install-project"]
    for extra in extras:
        cmd += ["--extra", extra]
    return cmd


def export_command(extras: list[str]) -> list[str]:
    cmd = ["uv", "export", "--frozen", "--no-emit-project", "--no-header"]
    for extra in extras:
        cmd += ["--extra", extra]
    return cmd


def clean_lock(text: str) -> str:
    """Drop the editable/project line(s): requirements.lock pins third-party packages only."""
    keep = [
        line
        for line in text.splitlines()
        if not line.startswith("-e ")
        and "evalkit @" not in line
        and not line.strip().startswith("file:")
    ]
    return "\n".join(keep).rstrip() + "\n"


def _extra_states(res: Resolution, have: set[str]) -> dict[str, str]:
    return {
        e: ("already" if e in have else "selected") if e in res.selected else "not selected"
        for e in EXTRA_NAMES
    }


def _local_base_url(project: Project) -> str | None:
    for prov in project.config.providers.values():
        if (
            prov.type == "openai_compatible"
            and prov.base_url
            and ("localhost" in prov.base_url or "127.0.0.1" in prov.base_url)
        ):
            return prov.base_url
    return None


def _tools(
    cfg: InstallConfig, res: Resolution, project: Project, run: Callable[..., Any], mode: str
) -> list[ToolStatus]:
    base = _local_base_url(project)
    return [
        handle(
            name,
            cfg.external.tool(name),
            res.tools_needed.get(name, []),
            project.root,
            run,
            base,
            mode,
        )
        for name in TOOL_NAMES
    ]


def _install_extras(
    report: SetupReport, res: Resolution, have: set[str], run: Callable[..., Any], root: Path
) -> None:
    todo = [e for e in res.selected if e not in have]
    if not todo:
        return
    cmd = sync_command(todo)
    report.commands.append(cmd)
    if shutil.which("uv") is None and run is run_cmd:
        for e in todo:
            report.extras[e] = "failed"
        report.warnings.append("`uv` not found on PATH: install it (winget install astral-sh.uv)")
        return
    ok = run(cmd, cwd=root).returncode == 0
    for e in todo:
        report.extras[e] = "installed" if ok else "failed"


def _write_lock(report: SetupReport, res: Resolution, run: Callable[..., Any], root: Path) -> None:
    cmd = export_command(res.selected)
    report.commands.append(cmd)
    out = run(cmd, cwd=root)
    if out.returncode != 0:
        report.warnings.append(f"requirements.lock not written: {(out.stderr or '')[-200:]}")
        return
    path = root / "requirements.lock"
    path.write_text(clean_lock(out.stdout), encoding="utf-8", newline="\n")
    report.lock_file = path.name


def setup(
    project: Project,
    cfg: InstallConfig,
    *,
    dry_run: bool = False,
    check: bool = False,
    no_external: bool = False,
    no_lock: bool = False,
    run: Callable[..., Any] = run_cmd,
) -> SetupReport:
    """Plan (dry-run), verify (check) or apply the install described by install.yaml."""
    root = project.root
    res = resolve(cfg, project)
    have = installed_extras(extra_packages(root))
    mode = "dry-run" if dry_run else ("check" if check else "apply")
    report = SetupReport(
        mode,
        project.profile,
        core_packages(root),
        _extra_states(res, have),
        res.needed,
        [],
        warnings=list(res.warnings),
        notes=list(res.notes),
        external_skipped=no_external,
    )
    if mode == "dry-run":
        todo = [e for e in res.selected if e not in have]
        if todo:
            report.commands.append(sync_command(todo))
        if not no_lock:
            report.commands.append(export_command(res.selected))
    elif mode == "check":
        for e in res.selected:
            report.extras[e] = "already" if e in have else "missing"
    else:
        _install_extras(report, res, have, run, root)
        if not no_lock:
            _write_lock(report, res, run, root)
    if not no_external:
        report.tools = _tools(cfg, res, project, run, mode)
        if mode == "dry-run":
            report.commands += [c for t in report.tools for c in t.commands]
    return report
