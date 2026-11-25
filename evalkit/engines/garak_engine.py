"""garak wrapper (extra `[security]`): run probes against an http target, parse its report.

garak is not installed in the build environment, so the CLI flags below follow garak's
documented REST generator usage and must be re-checked with `python -m garak --help`
(see docs/DECISIONS.md). The report parser is unit-tested with a fixture.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from evalkit.registry import engine
from evalkit.util import is_installed


@engine("garak", requires_extra="security")
class GarakEngine:
    """Engine marker: provides `garak_scan`."""

    package = "garak"


def rest_generator_config(target: Any) -> dict[str, Any]:
    """garak REST generator options built from an evalkit http target."""
    return {
        "rest": {
            "RestGenerator": {
                "uri": target.url,
                "method": target.method.lower(),
                "headers": target.headers,
                "req_template_json_object": {"input": "$INPUT"},
                "response_json": True,
                "response_json_field": next(iter(target.mapping.values())),
            }
        }
    }


def parse_report(path: Path) -> dict[str, Any]:
    """ASR per probe from garak `*.report.jsonl` eval entries (passed/total per detector)."""
    per_probe: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry.get("entry_type") != "eval":
            continue
        fails = int(entry.get("total", 0)) - int(entry.get("passed", 0))
        per_probe[str(entry.get("probe"))][0] += fails
        per_probe[str(entry.get("probe"))][1] += int(entry.get("total", 0))
    total = sum(t for _, t in per_probe.values())
    hits = sum(f for f, _ in per_probe.values())
    return {
        "asr": hits / total if total else 0.0,
        "attempts": total,
        "per_probe": {p: (f / t if t else 0.0) for p, (f, t) in sorted(per_probe.items())},
    }


def run_garak(target: Any, probes: list[str], run_dir: Path) -> dict[str, Any]:
    if not is_installed("garak"):
        return {"skipped": "garak not installed"}
    if not hasattr(target, "url"):
        return {"skipped": "n/a: garak needs an http target"}
    out = run_dir / "garak"
    out.mkdir(parents=True, exist_ok=True)
    opts = out / "rest.json"
    opts.write_text(json.dumps(rest_generator_config(target)), encoding="utf-8")
    prefix = out / "evalkit"
    cmd = [
        sys.executable,
        "-m",
        "garak",
        "--model_type",
        "rest",
        "--generator_option_file",
        str(opts),
        "--probes",
        ",".join(probes),
        "--report_prefix",
        str(prefix),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    reports = sorted(out.glob("*.report.jsonl"))
    if not reports:
        return {"skipped": f"garak produced no report (exit {proc.returncode})"}
    return parse_report(reports[-1])
