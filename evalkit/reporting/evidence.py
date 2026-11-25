"""Evidence pack (Plan.md §11.4): `evalctl evidence <run_id>` → evidence/<release>/."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml

from evalkit.config import Project
from evalkit.reporting.store import StoredRun

EVIDENCE_ITEMS = (
    "system_card.md",
    "risk_register.md",
    "test_plan.md",
    "dataset_cards",
    "evaluation_report",
    "red_team_report.md",
    "bias_report.md",
    "judge_validation",
    "monitoring_plan.md",
    "change_log.md",
)


def latest_pack(evidence_dir: Path) -> Path | None:
    if not evidence_dir.exists():
        return None
    packs = [p for p in evidence_dir.iterdir() if p.is_dir()]
    return max(packs, key=lambda p: (p.stat().st_mtime, p.name)) if packs else None


def _md_kv(title: str, data: dict[str, Any]) -> str:
    lines = [f"# {title}", ""]
    for key, val in data.items():
        if isinstance(val, list):
            lines += [f"**{key}**", ""] + [f"- {v}" for v in val] + [""]
        else:
            lines.append(f"- **{key}**: {val}")
    return "\n".join(lines) + "\n"


def system_card(project: Project, stored: StoredRun) -> str:
    card = dict(project.config.system)
    card["evaluated in run"] = stored.run_id
    card["model roles"] = [
        f"{r}: {v.get('provider')}:{v.get('model')}" for r, v in stored.info.roles.items()
    ]
    return _md_kv(f"System card — {card.get('name', project.config.project)}", card)


def risk_register(project: Project, stored: StoredRun) -> str:
    case_status: dict[str, bool] = {}
    for r in stored.rows:
        if r.case_id and r.passed is not None and r.skipped is None:
            case_status[r.case_id] = case_status.get(r.case_id, True) and bool(r.passed)
    lines = [
        "# Risk register",
        "",
        "| id | risk | severity | owner | linked cases | latest |",
        "|---|---|---|---|---|---|",
    ]
    for risk in project.config.risks:
        cases = list(risk.get("cases", []))
        seen = [case_status[c] for c in cases if c in case_status]
        latest = "not tested" if not seen else ("pass" if all(seen) else "FAIL")
        lines.append(
            f"| {risk.get('id')} | {risk.get('risk')} | {risk.get('severity', '')} | "
            f"{risk.get('owner', '')} | {', '.join(cases)} | {latest} |"
        )
    return "\n".join(lines) + "\n"


def test_plan(project: Project, stored: StoredRun) -> str:
    lines = [
        "# Test plan",
        "",
        f"Run `{stored.run_id}` · profile {stored.info.profile} · "
        f"repeats {stored.info.repeats} · seed {stored.info.seed}",
        "",
        "## Selection",
        "",
    ]
    for suite, metrics in stored.info.suites.items():
        lines.append(f"- **{suite}**: {', '.join(metrics)}")
    lines += [
        "",
        "## Thresholds",
        "",
        "```yaml",
        yaml.safe_dump(project.thresholds.model_dump(mode="json"), sort_keys=True).strip(),
        "```",
    ]
    return "\n".join(lines) + "\n"


def _suite_report(report_dir: Path, suite: str, title: str) -> str:
    md = report_dir / suite / "report.md"
    if md.exists():
        return md.read_text(encoding="utf-8")
    return f"# {title}\n\nThe {suite} suite was not part of this run.\n"


def monitoring_plan(project: Project) -> str:
    obs = project.config.observe
    return _md_kv(
        "Monitoring plan",
        {
            "Langfuse tracing": "enabled" if obs.langfuse else "disabled",
            "Trace links in reports": obs.trace_url_template or "not configured",
            "Nightly evaluation": "GitHub workflow nightly.yml (nightly profile)",
            "Feedback loop": "evalctl pull-traces → datasets/drafts/ → human review → golden",
            "Gate": "release.yml fails the release when the gate fails",
        },
    )


def change_log(project: Project) -> str:
    paths = ["prompts", "datasets", "judges", "evalkit.yaml"]
    try:
        tag = subprocess.run(
            ["git", "describe", "--tags", "--abbrev=0"],
            cwd=project.root,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        rng = [f"{tag}..HEAD"] if tag else []
        log = subprocess.run(
            ["git", "log", "--oneline", *rng, "--", *paths],
            cwd=project.root,
            capture_output=True,
            text=True,
            timeout=30,
        )
        body = log.stdout.strip() if log.returncode == 0 else "(not a git repository)"
    except (OSError, subprocess.SubprocessError):
        body = "(git not available)"
    return (
        f"# Change log\n\nChanges to {', '.join(paths)} since the last release tag:\n\n"
        f"```\n{body or '(no changes)'}\n```\n"
    )


def build_evidence(project: Project, stored: StoredRun, report_dir: Path) -> Path:
    """Write every §11.4 item into evidence/<release>/ and return the folder."""
    release = project.config.reports.release or stored.run_id
    out = project.path("evidence") / release
    out.mkdir(parents=True, exist_ok=True)
    files = {
        "system_card.md": system_card(project, stored),
        "risk_register.md": risk_register(project, stored),
        "test_plan.md": test_plan(project, stored),
        "red_team_report.md": _suite_report(report_dir, "security", "Red-team report"),
        "bias_report.md": _suite_report(report_dir, "responsible_ai", "Bias report"),
        "monitoring_plan.md": monitoring_plan(project),
        "change_log.md": change_log(project),
    }
    for name, text in files.items():
        (out / name).write_text(text, encoding="utf-8")
    _copy_dir(project.path("datasets/cards"), out / "dataset_cards")
    _copy_dir(project.path(project.config.judges_dir) / "calibration", out / "judge_validation")
    eval_dir = out / "evaluation_report"
    _copy_dir(report_dir, eval_dir)
    (eval_dir / "run_info.json").write_text(stored.info.model_dump_json(indent=2), encoding="utf-8")
    (out / "manifest.json").write_text(
        json.dumps(
            {"run_id": stored.run_id, "release": release, "items": list(EVIDENCE_ITEMS)}, indent=2
        ),
        encoding="utf-8",
    )
    return out


def _copy_dir(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    if src.exists():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    if not any(dst.iterdir()):
        (dst / "README.md").write_text(f"No files found in {src.name}.\n", encoding="utf-8")
