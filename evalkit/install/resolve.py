"""Which extras and external tools the current evalkit.yaml (+ profile) needs (Plan.md §13.1)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from evalkit.config import Project
from evalkit.install.config import CONFLICTS, EXTRA_NAMES, TOOL_NAMES, InstallConfig
from evalkit.planner import selected_metric_ids
from evalkit.registry import METRICS, SUITES, load_all, requires_extra
from evalkit.util import ConfigError

LOCAL_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1"}


@dataclass
class Resolution:
    mode: str
    needed: dict[str, list[str]]  # extra -> what needs it ("rag.faithfulness (engine ragas)")
    selected: list[str]  # extras to install
    tools_needed: dict[str, list[str]]  # tool -> what needs it
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _add(out: dict[str, list[str]], key: str | None, reason: str) -> None:
    if key and reason not in out[key]:
        out[key].append(reason)


def needed_extras(project: Project) -> dict[str, list[str]]:
    """Union of requires_extra over enabled suites' selected metrics, chosen engines, the
    providers used by any role, the target type, langfuse and locust."""
    load_all()
    cfg = project.config
    out: dict[str, list[str]] = defaultdict(list)
    for name in project.enabled_suites():
        if name not in SUITES:
            continue
        sc = cfg.suites[name]
        for mid in selected_metric_ids(project, name, sc):
            spec = METRICS.get(mid)
            _add(out, spec.requires_extra, f"{name}.{mid}")
            eng = cfg.engines.get(mid, "native")
            if eng != "native":
                _add(out, requires_extra("engine", eng), f"{name}.{mid} (engine {eng})")
        if name == "performance" and sc.params.get("engine") == "locust":
            _add(out, requires_extra("engine", "locust"), "performance (params.engine: locust)")
    for role, rm in sorted(cfg.models.items()):
        if rm is not None and rm.provider in cfg.providers:
            ptype = cfg.providers[rm.provider].type
            _add(out, requires_extra("provider", ptype), f"role {role} (provider {rm.provider})")
    target_type = _target_type(project)
    if target_type:
        _add(out, requires_extra("target", target_type), f"target ({target_type})")
    if cfg.observe.langfuse:
        _add(out, requires_extra("engine", "langfuse"), "observe.langfuse")
    return {k: sorted(v) for k, v in sorted(out.items())}


def _target_type(project: Project) -> str | None:
    from evalkit.config import read_yaml

    path = project.path(project.config.target)
    try:
        return str(read_yaml(path).get("type")) if path.exists() else None
    except ConfigError:
        return None


def _used_roles(project: Project) -> set[str]:
    """Roles the planned run would call (falls back to every configured role)."""
    from evalkit.planner import build_plan

    try:
        return set(build_plan(project, check_providers=False).calls)
    except Exception:  # an invalid config is reported by `validate`; be generous here
        return {r for r, m in project.config.models.items() if m is not None}


def _is_local(project: Project, provider: str) -> bool:
    cfg = project.config.providers.get(provider)
    if cfg is None or cfg.type != "openai_compatible":
        return False
    return (urlparse(cfg.base_url or "").hostname or "") in LOCAL_HOSTS


def needed_tools(project: Project) -> dict[str, list[str]]:
    """External tools an enabled feature needs."""
    cfg = project.config
    out: dict[str, list[str]] = defaultdict(list)
    for role in sorted(_used_roles(project)):
        rm = cfg.models.get(role)
        if rm is not None and _is_local(project, rm.provider):
            _add(out, "local_model", f"role {role} (provider {rm.provider})")
    enabled = set(project.enabled_suites())
    sec = cfg.suites.get("security")
    if "security" in enabled and sec and sec.params.get("generate") == "promptfoo":
        _add(out, "promptfoo", "security (params.generate: promptfoo)")
    if cfg.observe.langfuse:
        _add(out, "langfuse", "observe.langfuse")
    cls = cfg.suites.get("classic")
    if (
        "classic" in enabled
        and cls
        and cls.params.get("ui_tests")
        and "ui_streaming" in selected_metric_ids(project, "classic", cls)
    ):
        _add(out, "playwright_browsers", "classic.ui_streaming")
    code = cfg.suites.get("coding_agent")
    if "coding_agent" in enabled and code:
        if "security_findings" in selected_metric_ids(project, "coding_agent", code):
            _add(out, "gitleaks", "coding_agent.security_findings")
        if code.params.get("docker_image"):
            _add(out, "docker", "coding_agent (params.docker_image)")
    return {k: sorted(v) for k, v in sorted(out.items())}


def _choose(cfg: InstallConfig, needed: dict[str, list[str]], res: Resolution) -> list[str]:
    selected = []
    for extra in EXTRA_NAMES:
        switch = cfg.extras.switch(extra)
        if switch is True or (switch == "auto" and extra in needed):
            selected.append(extra)
        elif switch is False and extra in needed:
            res.warnings.append(
                f"extra '{extra}' is set to false but needed by {', '.join(needed[extra])}: "
                "these will show as Not tested"
            )
    return selected


def _tool_warnings(cfg: InstallConfig, tools: dict[str, list[str]]) -> list[str]:
    out = []
    for tool, reasons in tools.items():
        if tool in TOOL_NAMES and cfg.external.tool(tool).action == "off":
            out.append(
                f"external tool '{tool}' is off but needed by {', '.join(reasons)}: "
                "that feature will be skipped"
            )
    return out


def resolve(cfg: InstallConfig, project: Project) -> Resolution:
    """Apply mode choose | auto | all to the needed extras; collect warnings."""
    needed = needed_extras(project)
    tools = needed_tools(project)
    mode = cfg.extras.mode
    res = Resolution(mode, needed, [], tools)
    if mode == "all":
        res.selected = list(EXTRA_NAMES)
    elif mode == "auto":
        res.selected = [e for e in EXTRA_NAMES if e in needed]
        explicit = [e for e in EXTRA_NAMES if cfg.extras.switch(e) != "auto"]
        if explicit:
            res.notes.append(
                f"mode: auto ignores the extra switches ({', '.join(explicit)}); "
                "only the needed extras are installed"
            )
    else:
        res.selected = _choose(cfg, needed, res)
    _apply_conflicts(res)
    res.warnings += _tool_warnings(cfg, tools)
    return res


def _apply_conflicts(res: Resolution) -> None:
    for a, b in CONFLICTS:
        if a in res.selected and b in res.selected:
            if res.mode == "all":
                res.selected.remove(a)
                res.notes.append(
                    f"mode: all leaves out '{a}': it cannot share an environment "
                    f"with '{b}' (see docs/DECISIONS.md)"
                )
            else:
                raise ConfigError(
                    f"install.yaml: extras '{a}' and '{b}' cannot be installed "
                    "together (garak pins anthropic<1.0); switch one off"
                )


def validate_warnings(project: Project, cfg: InstallConfig) -> list[str]:
    """Warnings `evalctl validate` prints for install.yaml vs evalkit.yaml."""
    return resolve(cfg, project).warnings


def describe(res: Resolution) -> dict[str, Any]:
    return {
        "mode": res.mode,
        "needed": res.needed,
        "selected": res.selected,
        "tools_needed": res.tools_needed,
        "warnings": res.warnings,
        "notes": res.notes,
    }
