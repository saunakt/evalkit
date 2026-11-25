"""`type: cli` target: runs a command template in a working dir (Plan.md §7.1)."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from evalkit.registry import target
from evalkit.schema import Case, Observation
from evalkit.targets.base import TargetContext, coerce_observation
from evalkit.util import extract_json


class CliTarget:
    """`command` may use {prompt_file}, {prompt}, {workdir}, {python}.

    The working dir is `ctx.params["workdir"]` (set by suites that sandbox a repo)
    or the configured `workdir`.
    """

    def __init__(self, cfg: dict[str, Any], root: Path) -> None:
        self.name = str(cfg.get("name", "cli"))
        self.command = str(cfg["command"])
        self.workdir = cfg.get("workdir")
        self.timeout_s = float(cfg.get("timeout_s", 600))
        self.parse = str(cfg.get("parse", "text"))
        self.root = root
        self.thread_safe = True
        self.supports_media = False
        self.uses_roles: list[str] = list(cfg.get("uses_roles", []))

    def call(self, case: Case, ctx: TargetContext) -> Observation:
        prompt = str(ctx.params.get("prompt") or case.text_input())
        workdir = Path(ctx.params.get("workdir") or self.workdir or self.root)
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as fh:
            fh.write(prompt)
            prompt_file = fh.name
        cmd = self.command.format(
            prompt_file=quote(prompt_file),
            prompt=quote(prompt),
            workdir=quote(str(workdir)),
            python=quote(sys.executable),
            root=quote(str(self.root)),
        )
        start = time.perf_counter()
        try:
            proc = subprocess.run(
                cmd,
                shell=True,
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )
        except subprocess.TimeoutExpired:
            return Observation(error=f"timeout after {self.timeout_s}s", latency_s=self.timeout_s)
        finally:
            Path(prompt_file).unlink(missing_ok=True)
        latency = time.perf_counter() - start
        obs = self._parse(proc.stdout)
        obs.latency_s = obs.latency_s or latency
        if proc.returncode != 0:
            obs.error = obs.error or f"exit {proc.returncode}: {proc.stderr[-500:]}"
        raw = obs.raw if isinstance(obs.raw, dict) else {}
        obs.raw = {**raw, "returncode": proc.returncode, "stderr": proc.stderr[-2000:]}
        return obs

    def _parse(self, stdout: str) -> Observation:
        if self.parse != "json":
            return Observation(text=stdout.strip())
        data = extract_json(stdout)
        if not isinstance(data, dict):
            return Observation(text=stdout.strip(), error="malformed JSON output")
        known = {k: v for k, v in data.items() if k in Observation.model_fields}
        obs = coerce_observation({"text": str(data.get("result", data.get("text", ""))), **known})
        obs.raw = data
        return obs


def quote(value: str) -> str:
    """Shell-quote for the platform shell (cmd.exe on Windows, sh elsewhere)."""
    return subprocess.list2cmdline([value]) if os.name == "nt" else shlex.quote(value)


@target("cli")
def make_cli_target(cfg: dict[str, Any], root: Path) -> CliTarget:
    return CliTarget(cfg, root)
