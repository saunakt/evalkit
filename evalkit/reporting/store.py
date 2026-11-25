"""Results store: `results/<run_id>/` read/write (Plan.md §11.2)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evalkit.schema import Case, ResultRow, RunInfo
from evalkit.util import DataError, read_jsonl, write_jsonl

BASELINE_FILE = "baseline.json"


@dataclass
class StoredRun:
    run_dir: Path
    info: RunInfo
    rows: list[ResultRow]
    panels: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    cases: dict[str, Case] = field(default_factory=dict)
    observations: list[dict[str, Any]] = field(default_factory=list)

    @property
    def run_id(self) -> str:
        return self.info.run_id

    def suites(self) -> list[str]:
        return list(self.info.suites)

    def suite_rows(self, suite: str) -> list[ResultRow]:
        return [r for r in self.rows if r.suite == suite]


class ResultStore:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        run_dir.mkdir(parents=True, exist_ok=True)

    def write_info(self, info: RunInfo) -> None:
        (self.run_dir / "run_info.json").write_text(
            info.model_dump_json(indent=2), encoding="utf-8"
        )

    def write_rows(self, rows: list[ResultRow]) -> None:
        write_jsonl(self.run_dir / "results.jsonl", rows)

    def write_observations(self, obs_rows: list[dict[str, Any]]) -> None:
        write_jsonl(self.run_dir / "observations.jsonl", obs_rows)

    def write_cases(self, cases: list[dict[str, Any]]) -> None:
        write_jsonl(self.run_dir / "cases_snapshot.jsonl", cases)

    def write_panels(self, panels: dict[str, list[dict[str, Any]]]) -> None:
        (self.run_dir / "panels.json").write_text(
            json.dumps(panels, indent=2, sort_keys=True, default=str), encoding="utf-8"
        )


def load_run(run_dir: Path, with_observations: bool = False) -> StoredRun:
    info_path = run_dir / "run_info.json"
    if not info_path.exists():
        raise DataError(f"not a run directory: {run_dir}")
    info = RunInfo.model_validate_json(info_path.read_text(encoding="utf-8"))
    rows = [ResultRow.model_validate(r) for r in read_jsonl(run_dir / "results.jsonl")]
    panels_path = run_dir / "panels.json"
    panels = json.loads(panels_path.read_text(encoding="utf-8")) if panels_path.exists() else {}
    cases: dict[str, Case] = {}
    snap = run_dir / "cases_snapshot.jsonl"
    if snap.exists():
        for row in read_jsonl(snap):
            cases[f"{row.get('_suite')}/{row['id']}"] = Case.model_validate(row)
    obs_path = run_dir / "observations.jsonl"
    obs = read_jsonl(obs_path) if with_observations and obs_path.exists() else []
    return StoredRun(run_dir, info, rows, panels, cases, obs)


def resolve_run(results_dir: Path, run_id: str) -> Path:
    """Run id → directory; `last` means the most recent run."""
    if run_id in ("last", "latest"):
        runs = list_runs(results_dir)
        if not runs:
            raise DataError(f"no runs in {results_dir}")
        return results_dir / runs[-1]
    path = results_dir / run_id
    if not (path / "run_info.json").exists():
        raise DataError(f"run {run_id!r} not found in {results_dir}")
    return path


def list_runs(results_dir: Path) -> list[str]:
    if not results_dir.exists():
        return []
    runs = [p for p in results_dir.iterdir() if (p / "run_info.json").exists()]
    return [p.name for p in sorted(runs, key=lambda p: (p.stat().st_mtime, p.name))]


def set_baseline(results_dir: Path, run_id: str) -> None:
    resolve_run(results_dir, run_id)
    (results_dir / BASELINE_FILE).write_text(json.dumps({"run_id": run_id}), encoding="utf-8")


def find_baseline(results_dir: Path, setting: str | None, current: str) -> StoredRun | None:
    """`auto` → the run marked baseline (if any); otherwise an explicit run id."""
    if not setting:
        return None
    if setting == "auto":
        marker = results_dir / BASELINE_FILE
        if not marker.exists():
            return None
        setting = str(json.loads(marker.read_text(encoding="utf-8")).get("run_id", ""))
    if not setting or setting == current:
        return None
    try:
        return load_run(resolve_run(results_dir, setting), with_observations=True)
    except DataError:
        return None
