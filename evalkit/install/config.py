"""install.yaml models and loading (Plan.md §13.1).

install.yaml never names packages: pyproject.toml extras are the single source of package
lists. It only switches extras on/off/auto and manages tools that live outside Python.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator

from evalkit.config import parse_model, read_yaml
from evalkit.util import ConfigError

EXTRA_NAMES = (
    "anthropic",
    "deepeval",
    "ragas",
    "textmetrics",
    "rai",
    "security",
    "multilingual",
    "voice",
    "perf",
    "coding",
    "agent",
    "ui",
    "observe",
)
TOOL_NAMES = (
    "local_model",
    "promptfoo",
    "gitleaks",
    "docker",
    "wiremock",
    "label_studio",
    "langfuse",
    "playwright_browsers",
)
INSTALLABLE = ("promptfoo", "playwright_browsers")
STARTABLE = ("wiremock", "label_studio")
# Extras that cannot share one environment (declared in [tool.uv] conflicts).
CONFLICTS = (("security", "anthropic"),)

Switch = bool | Literal["auto"]
Action = Literal["install", "start", "check", "off"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Extras(_Strict):
    mode: Literal["choose", "auto", "all"] = "choose"
    anthropic: Switch = "auto"
    deepeval: Switch = "auto"
    ragas: Switch = "auto"
    textmetrics: Switch = False
    rai: Switch = "auto"
    security: Switch = False
    multilingual: Switch = "auto"
    voice: Switch = "auto"
    perf: Switch = "auto"
    coding: Switch = False
    agent: Switch = "auto"
    ui: Switch = False
    observe: Switch = "auto"

    def switch(self, extra: str) -> Switch:
        value: Switch = getattr(self, extra)
        return value


class Tool(_Strict):
    action: Action = "off"
    url: str | None = None

    @field_validator("action", mode="before")
    @classmethod
    def _yaml_off(cls, value: Any) -> Any:
        """YAML 1.1 reads a bare `off` as false; treat it as the 'off' action."""
        return "off" if value is False else value


class External(_Strict):
    local_model: Tool = Tool(action="check")
    promptfoo: Tool = Tool()
    gitleaks: Tool = Tool()
    docker: Tool = Tool(action="check")
    wiremock: Tool = Tool()
    label_studio: Tool = Tool()
    langfuse: Tool = Tool()
    playwright_browsers: Tool = Tool()

    def tool(self, name: str) -> Tool:
        value: Tool = getattr(self, name)
        return value


class InstallConfig(_Strict):
    core: list[str] = []  # informational: generated from pyproject core dependencies
    extras: Extras = Extras()
    external: External = External()


KNOWN_KEYS = sorted(
    {*Extras.model_fields, *Tool.model_fields, *External.model_fields, *InstallConfig.model_fields}
)


def check_actions(cfg: InstallConfig, source: str) -> None:
    """`install` only for promptfoo/playwright_browsers; `start` only for docker services."""
    errors = []
    for name in TOOL_NAMES:
        action = cfg.external.tool(name).action
        if action == "install" and name not in INSTALLABLE:
            errors.append(
                f"{source}: external.{name}.action: 'install' is not supported for "
                f"{name} (check-only tool; use 'check' and install it yourself)"
            )
        if action == "start" and name not in STARTABLE:
            errors.append(
                f"{source}: external.{name}.action: 'start' is only supported for "
                f"{', '.join(STARTABLE)}"
            )
    if errors:
        raise ConfigError("\n".join(errors))


def load_install(path: str | Path) -> InstallConfig:
    """Load and validate install.yaml (missing file → defaults)."""
    p = Path(path)
    data: dict[str, Any] = read_yaml(p) if p.exists() else {}
    cfg: InstallConfig = parse_model(InstallConfig, data, p.name, KNOWN_KEYS)
    check_actions(cfg, p.name)
    return cfg


def read_pyproject(root: Path) -> dict[str, Any]:
    with open(root / "pyproject.toml", "rb") as fh:
        data: dict[str, Any] = tomllib.load(fh)
    return data


_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*")


def requirement_name(req: str) -> str:
    """'deepeval==4.3.5' → 'deepeval'; 'evalkit[a,b]' → 'evalkit'."""
    m = _NAME.match(req.strip())
    return m.group(0).lower().replace("_", "-") if m else req.strip().lower()


def core_packages(root: Path) -> list[str]:
    deps = read_pyproject(root)["project"]["dependencies"]
    return [requirement_name(d) for d in deps]


def extra_packages(root: Path) -> dict[str, list[str]]:
    """Extra → package names, for the extras install.yaml can switch (not dev/all)."""
    extras = read_pyproject(root)["project"]["optional-dependencies"]
    return {
        name: [requirement_name(r) for r in extras[name]] for name in EXTRA_NAMES if name in extras
    }
