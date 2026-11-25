"""External tools for `evalctl setup`: install / start / check (Plan.md §13.1)."""

from __future__ import annotations

import os
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from evalkit.install.config import Tool
from evalkit.util import is_installed

DEFAULT_URLS = {
    "wiremock": "http://localhost:8089/__admin/mappings",
    "label_studio": "http://localhost:8090",
    "langfuse": "http://localhost:3000",
}
SERVICES = {"wiremock": "wiremock", "label_studio": "label-studio"}
MANUAL = {
    "local_model": "Start a local OpenAI-compatible server, e.g. `ollama serve` (then "
    "`ollama pull <model>`) or `vllm serve <model> --port 8000`.",
    "promptfoo": "Install Node.js (https://nodejs.org), then run `npm install` in the repo "
    "(installs the promptfoo version pinned in package.json).",
    "gitleaks": "Install gitleaks: `winget install gitleaks.gitleaks` (Windows) or "
    "`brew install gitleaks` (macOS), or download it from GitHub releases.",
    "docker": "Install Docker Desktop (https://docs.docker.com/get-docker/) and start it.",
    "wiremock": "Run `docker compose -f sandbox/docker-compose.yml up -d wiremock`.",
    "label_studio": "Run `docker compose -f sandbox/docker-compose.yml up -d label-studio`.",
    "langfuse": "Use Langfuse Cloud or self-host it, then set observe.langfuse_host / "
    "LANGFUSE_HOST and the LANGFUSE_* keys.",
    "playwright_browsers": "Enable the `ui` extra, then run `uv run playwright install chromium`.",
}

RunCmd = Callable[..., Any]


@dataclass
class ToolStatus:
    name: str
    action: str
    state: str  # off | found | missing | running | installed | failed | planned | skipped
    needed_by: list[str] = field(default_factory=list)
    detail: str = ""
    manual: str = ""
    commands: list[list[str]] = field(default_factory=list)

    @property
    def required_missing(self) -> bool:
        """Missing and some enabled feature needs it."""
        return bool(self.needed_by) and self.state in ("missing", "failed")


def http_ok(url: str, timeout: float = 3.0) -> bool:
    try:
        return httpx.get(url, timeout=timeout).status_code < 500
    except httpx.HTTPError:
        return False


def tool_url(name: str, tool: Tool, local_base_url: str | None) -> str | None:
    if tool.url:
        return tool.url
    if name == "local_model" and local_base_url:
        return local_base_url.rstrip("/") + "/models"
    return DEFAULT_URLS.get(name)


def browsers_present() -> bool:
    base = os.environ.get("PLAYWRIGHT_BROWSERS_PATH") or (
        str(Path(os.environ.get("LOCALAPPDATA", "")) / "ms-playwright")
        if os.name == "nt"
        else str(Path.home() / ".cache" / "ms-playwright")
    )
    return is_installed("playwright") and any(Path(base).glob("chromium*"))


def present(name: str, tool: Tool, root: Path, local_base_url: str | None) -> tuple[bool, str]:
    """Is the tool there? (which / HTTP GET / file check) — never runs a command."""
    if name in ("local_model", "langfuse", "wiremock", "label_studio"):
        url = tool_url(name, tool, local_base_url)
        return (bool(url) and http_ok(str(url))), f"GET {url}"
    if name == "promptfoo":
        local = root / "node_modules" / ".bin"
        found = shutil.which("promptfoo") is not None or any(local.glob("promptfoo*"))
        return found, "node_modules/.bin/promptfoo or promptfoo on PATH"
    if name == "playwright_browsers":
        return browsers_present(), "playwright chromium browser"
    exe = {"docker": "docker", "gitleaks": "gitleaks"}[name]
    return shutil.which(exe) is not None, f"`{exe}` on PATH"


def planned_commands(name: str, action: str) -> list[list[str]]:
    if action == "install" and name == "promptfoo":
        return [["npm", "install"]]
    if action == "install" and name == "playwright_browsers":
        return [[sys.executable, "-m", "playwright", "install", "chromium"]]
    if action == "start":
        return [
            ["docker", "compose", "-f", "sandbox/docker-compose.yml", "up", "-d", SERVICES[name]]
        ]
    return []


def handle(
    name: str,
    tool: Tool,
    needed_by: list[str],
    root: Path,
    run_cmd: RunCmd,
    local_base_url: str | None,
    mode: str,
) -> ToolStatus:
    """mode: apply | check | dry-run."""
    st = ToolStatus(
        name,
        tool.action,
        "off",
        needed_by,
        manual=MANUAL[name],
        commands=planned_commands(name, tool.action),
    )
    if tool.action == "off":
        return st
    if mode == "dry-run":
        st.state = "planned"
        return st
    ok, how = present(name, tool, root, local_base_url)
    st.detail = how
    if ok or tool.action == "check" or mode == "check":
        st.state = ("running" if tool.action == "start" else "found") if ok else "missing"
        return st
    if tool.action == "install" and name == "promptfoo" and shutil.which("npm") is None:
        st.state, st.detail = "failed", "npm not found on PATH (needs Node.js)"
        return st
    for cmd in st.commands:
        res = run_cmd(cmd, cwd=root)
        if res.returncode != 0:
            st.state, st.detail = "failed", (res.stderr or res.stdout or "")[-300:]
            return st
    st.state = "running" if tool.action == "start" else "installed"
    return st
