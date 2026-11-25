"""Promptfoo red-team generation (optional; Node CLI pinned in package.json).

`promptfoo redteam generate` writes test cases; we point Promptfoo's provider at the
`generator` role's OpenAI-compatible endpoint so generation stays free, then import the
generated tests as evalkit cases. Promptfoo was not run in the build environment: re-check
flags with `npx promptfoo redteam generate --help` (see docs/DECISIONS.md).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml

from evalkit.registry import engine
from evalkit.schema import Case

PLUGIN_OWASP = {
    "prompt-extraction": "LLM07",
    "pii": "LLM02",
    "excessive-agency": "LLM06",
    "hijacking": "LLM01",
    "harmful": "LLM09",
    "rbac": "LLM08",
    "sql-injection": "LLM05",
}


@engine("promptfoo")
class PromptfooEngine:
    """Engine marker: red-team case generation for the security suite."""

    package = "promptfoo (npm)"


def redteam_config(
    purpose: str, plugins: list[str], strategies: list[str], base_url: str, model: str
) -> dict[str, Any]:
    return {
        "targets": [{"id": "echo"}],
        "redteam": {
            "purpose": purpose,
            "plugins": plugins,
            "strategies": strategies,
            "provider": {
                "id": f"openai:chat:{model}",
                "config": {"apiBaseUrl": base_url, "apiKey": "none"},
            },
        },
    }


def import_tests(path: Path) -> list[Case]:
    """Generated Promptfoo tests → evalkit attack cases."""
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cases = []
    for i, test in enumerate(data.get("tests", [])):
        vars_ = test.get("vars", {})
        text = next(iter(vars_.values()), "") if vars_ else ""
        plugin = str((test.get("metadata") or {}).get("pluginId", "promptfoo")).split(":")[-1]
        cases.append(
            Case(
                id=f"pf-{i:03d}",
                input=str(text),
                tags=["promptfoo"],
                attack={
                    "owasp": PLUGIN_OWASP.get(plugin, "LLM01"),
                    "vector": "direct",
                    "goal": f"promptfoo {plugin} attack",
                },
            )
        )
    return cases


def generate_cases(sctx: Any) -> list[Case]:
    """Run `npx promptfoo redteam generate`; returns [] (with a note) when Node is missing."""
    npx = shutil.which("npx")
    if npx is None:
        sctx.memo.setdefault("notes", []).append("promptfoo skipped: npx not found")
        return []
    rm = sctx.project.config.models.get("generator")
    prov = sctx.project.config.providers.get(rm.provider) if rm else None
    if rm is None or prov is None or not prov.base_url:
        sctx.memo.setdefault("notes", []).append(
            "promptfoo skipped: generator role has no OpenAI-compatible base_url"
        )
        return []
    out_dir = sctx.run_dir / "promptfoo"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = redteam_config(
        str(sctx.params.get("purpose", "customer support assistant")),
        list(sctx.params.get("plugins", ["prompt-extraction", "pii"])),
        list(sctx.params.get("promptfoo_strategies", ["jailbreak"])),
        prov.base_url,
        rm.model,
    )
    cfg_path, out_path = out_dir / "promptfooconfig.yaml", out_dir / "redteam.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    subprocess.run(
        [npx, "promptfoo", "redteam", "generate", "-c", str(cfg_path), "-o", str(out_path)],
        capture_output=True,
        text=True,
        timeout=3600,
        cwd=sctx.project.root,
    )
    return import_tests(out_path) if out_path.exists() else []
