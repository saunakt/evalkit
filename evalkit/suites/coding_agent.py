"""coding_agent — coding agents in a sandbox: hidden tests, regressions, diffs (ch. 21–22)."""

from __future__ import annotations

import difflib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from evalkit.env.sandbox import Sandbox
from evalkit.metrics.classic import run_pytest
from evalkit.registry import suite
from evalkit.schema import Case, Observation
from evalkit.suites.base import BaseSuite
from evalkit.suites.context import SuiteContext
from evalkit.targets.cli_target import quote
from evalkit.util import is_installed


def load_tasks(tasks_dir: Path) -> list[Case]:
    cases = []
    for task_yaml in sorted(tasks_dir.glob("*/task.yaml")):
        task = yaml.safe_load(task_yaml.read_text(encoding="utf-8")) or {}
        cases.append(
            Case(
                id=task_yaml.parent.name,
                input=str(task["prompt"]),
                tags=list(task.get("tags", [])),
                extra={
                    "task": task,
                    "task_dir": str(task_yaml.parent),
                    "kind": task.get("kind", "fix"),
                },
            )
        )
    return cases


def changed_lines(fixture: Path, sandbox: Path, files: list[str]) -> int:
    total = 0
    for rel in files:
        old = (
            (fixture / rel).read_text(encoding="utf-8").splitlines()
            if (fixture / rel).exists()
            else []
        )
        new = (sandbox / rel).read_text(encoding="utf-8", errors="ignore").splitlines()
        total += sum(
            1
            for line in difflib.unified_diff(old, new, lineterm="")
            if line[:1] in "+-" and not line.startswith(("+++", "---"))
        )
    return total


def lint_count(sbx: Sandbox, files: list[str]) -> int | None:
    py = [f for f in files if f.endswith(".py")]
    if not py or (not is_installed("ruff") and shutil.which("ruff") is None):
        return 0 if not py else None
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--output-format",
            "json",
            "--select",
            "E,F,W,B",
            *py,
        ],
        cwd=sbx.path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    try:
        return len(json.loads(proc.stdout or "[]"))
    except json.JSONDecodeError:
        return None


def security_scan(sbx: Sandbox, files: list[str]) -> dict[str, Any]:
    """Bandit (if installed) on changed Python files; semgrep / gitleaks when on PATH."""
    out: dict[str, Any] = {"tools": [], "findings": 0}
    py = [f for f in files if f.endswith(".py")]
    if py and is_installed("bandit"):
        proc = subprocess.run(
            [sys.executable, "-m", "bandit", "-q", "-f", "json", *py],
            cwd=sbx.path,
            capture_output=True,
            text=True,
            timeout=300,
        )
        out["tools"].append("bandit")
        out["findings"] += len(json.loads(proc.stdout or "{}").get("results", []))
    for tool in ("semgrep", "gitleaks"):
        if shutil.which(tool):
            out["tools"].append(tool)
    return out


def score_sandbox(
    sbx: Sandbox, task: dict[str, Any], task_dir: Path, fixture: Path, obs: Observation
) -> dict[str, Any]:
    """Run hidden tests, existing tests and rule checks; collect diff and lint facts."""
    changed = [f for f in sbx.changed_files() if not f.startswith("_hidden")]
    hidden_src = task_dir / str(task.get("hidden_tests", "hidden_tests"))
    hidden = {"tests": 0, "failures": 0}
    if hidden_src.exists():
        shutil.copytree(hidden_src, sbx.path / "_hidden_tests", dirs_exist_ok=True)
        hidden = run_pytest("_hidden_tests", sbx.path, timeout_s=300)
    existing = (
        run_pytest(str(task["existing_tests"]), sbx.path, timeout_s=300)
        if task.get("existing_tests")
        else {"tests": 0, "failures": 0}
    )
    rules = []
    for cmd in task.get("rule_checks", []):
        res = sbx.run(str(cmd).replace("{python}", quote(sys.executable)))
        rules.append({"command": cmd, "ok": res.returncode == 0})
    raw = obs.raw if isinstance(obs.raw, dict) else {}
    reference = list(task.get("reference_files", []))
    return {
        "resolved": hidden["tests"] > 0 and hidden["failures"] == 0,
        "hidden": hidden,
        "regressed": existing["failures"] > 0,
        "existing": existing,
        "rules": rules,
        "changed_files": changed,
        "changed_lines": changed_lines(fixture, sbx.path, changed),
        "unrelated_files": [f for f in changed if reference and f not in reference],
        "lint": lint_count(sbx, changed),
        "security": security_scan(sbx, changed),
        "commands": list(raw.get("commands", [])) + [c.command for c in sbx.log],
        "review": raw.get("review"),
        "tests_written": raw.get("tests_written"),
    }


@suite("coding_agent")
class CodingAgentSuite(BaseSuite):
    """Coding agents in a sandbox: hidden tests, regressions, diff discipline (ch. 21–22)."""

    name = "coding_agent"
    default_metrics = (
        "resolution_rate",
        "pass_hat_k",
        "regression_rate",
        "code_quality",
        "diff_discipline",
        "rules_adherence",
        "command_safety",
    )

    def prepare(self, cases: list[Case], sctx: SuiteContext) -> list[Case]:
        tasks_dir = sctx.project.path(str(sctx.params.get("tasks_dir", "datasets/coding_tasks")))
        return cases + load_tasks(tasks_dir)

    def execute(self, case: Case, repeat: int, sctx: SuiteContext) -> Observation:
        task: dict[str, Any] = dict(case.extra["task"])
        task_dir = Path(str(case.extra["task_dir"]))
        fixture = task_dir / str(task.get("repo", "repo"))
        prompt = f"{task['prompt']}\n\n[task:{case.id}]"
        with Sandbox(fixture, docker_image=sctx.params.get("docker_image")) as sbx:
            obs = sctx.call(case, params={"workdir": str(sbx.path), "prompt": prompt})
            obs.raw = {
                **(obs.raw if isinstance(obs.raw, dict) else {}),
                "coding": score_sandbox(sbx, task, task_dir, fixture, obs),
            }
        return obs
