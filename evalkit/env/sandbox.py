"""Temp-dir / git-worktree / docker sandbox for CLI targets (Plan.md §7.3)."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType


@dataclass
class CommandResult:
    command: str
    returncode: int
    stdout: str
    stderr: str


@dataclass
class Sandbox:
    """Copy a repo fixture to a fresh temp dir; always cleaned up on exit.

    With `use_worktree=True` and a git repo fixture, uses `git worktree add`.
    With `docker_image`, `run()` executes inside `docker run` with the dir mounted.
    """

    fixture: Path
    use_worktree: bool = False
    docker_image: str | None = None
    path: Path = field(init=False)
    log: list[CommandResult] = field(default_factory=list)
    _tmp: str = field(init=False, default="")

    def __enter__(self) -> Sandbox:
        self._tmp = tempfile.mkdtemp(prefix="evalkit-sbx-")
        self.path = Path(self._tmp) / "repo"
        if self.use_worktree and (self.fixture / ".git").exists():
            subprocess.run(
                ["git", "-C", str(self.fixture), "worktree", "add", "--detach", str(self.path)],
                check=True,
                capture_output=True,
            )
        else:
            shutil.copytree(self.fixture, self.path)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self.use_worktree and (self.fixture / ".git").exists():
            subprocess.run(
                ["git", "-C", str(self.fixture), "worktree", "remove", "--force", str(self.path)],
                capture_output=True,
            )
        shutil.rmtree(self._tmp, ignore_errors=True)

    def run(self, command: str, timeout_s: float = 300.0) -> CommandResult:
        """Run a shell command in the sandbox and record it in the command log."""
        cmd = command
        if self.docker_image:
            mount = f'-v "{self.path}:/work" -w /work'
            cmd = f"docker run --rm {mount} {self.docker_image} sh -c {command!r}"
        try:
            proc = subprocess.run(
                cmd, shell=True, cwd=self.path, capture_output=True, text=True, timeout=timeout_s
            )
            result = CommandResult(command, proc.returncode, proc.stdout, proc.stderr)
        except subprocess.TimeoutExpired as exc:
            result = CommandResult(
                command, -1, str(exc.stdout or ""), f"timeout after {timeout_s}s"
            )
        self.log.append(result)
        return result

    def changed_files(self) -> list[str]:
        """Files that differ from the fixture (added or modified), relative paths, sorted."""
        out = []
        for p in sorted(self.path.rglob("*")):
            if not p.is_file() or "__pycache__" in p.parts or ".pytest_cache" in p.parts:
                continue
            rel = p.relative_to(self.path)
            orig = self.fixture / rel
            if not orig.exists() or orig.read_bytes() != p.read_bytes():
                out.append(rel.as_posix())
        return out
