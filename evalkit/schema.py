"""Core data model — public interfaces (Plan.md §5). Keep stable."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, NamedTuple

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from evalkit.util import DataError


class _Open(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)


# --------------------------------------------------------------------------- Case


class Turn(_Open):
    user: str
    expect: dict[str, Any] = {}


class ToolExpectation(_Open):
    name: str
    args: dict[str, Any] | None = None
    agent: str | None = None


class Case(_Open):
    id: str
    suite: str | None = None
    tags: list[str] = []
    split: Literal["dev", "heldout"] = "dev"
    must_pass: bool = False
    repeats: int | None = None
    language: str | None = None
    # inputs (one of)
    input: str | None = None
    turns: list[Turn] | None = None
    persona: str | None = None
    hidden_goal: str | None = None
    max_turns: int = 8
    media: str | None = None
    # expectations
    reference: str | None = None
    label: str | None = None
    criteria: list[str] = []
    must_include: list[str] = []
    must_not_include: list[str] = []
    regex: str | None = None
    schema_: dict[str, Any] | str | None = Field(None, alias="schema")
    relevant_ids: list[str] = []
    relevance: dict[str, int] = {}
    expect_source: str | None = None
    expect_tools: list[str] = []
    forbid_tools: list[str] = []
    expect_trajectory: list[ToolExpectation] = []
    trajectory_mode: Literal["exact", "in_order", "any_order", "subset"] = "in_order"
    max_steps: int | None = None
    expect_state: dict[str, Any] = {}
    policies: list[dict[str, Any]] = []
    faults: list[dict[str, Any]] = []
    variants: list[str] | dict[str, list[str]] | None = None
    relation: Literal["same_decision", "different_decision"] | None = None
    template: str | None = None
    attack: dict[str, Any] | None = None
    expected_transcript: str | None = None
    expected_fields: dict[str, Any] = {}
    entities: list[str] = []
    source: str | None = None
    extra: dict[str, Any] = {}

    @classmethod
    def from_jsonl(cls, path: str | Path) -> list[Case]:
        """Validate every line; report all `file:line: error`s; reject duplicate ids."""
        errors: list[str] = []
        cases: list[Case] = []
        seen: dict[str, int] = {}
        with open(path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                case = _parse_line(path, lineno, line, errors)
                if case is None:
                    continue
                if case.id in seen:
                    errors.append(
                        f"{path}:{lineno}: duplicate id {case.id!r} (first on line {seen[case.id]})"
                    )
                    continue
                seen[case.id] = lineno
                cases.append(case)
        if errors:
            raise DataError("invalid cases:\n" + "\n".join(errors))
        return cases

    def text_input(self) -> str:
        """Best single-string view of the input (input, else concatenated turns)."""
        if self.input is not None:
            return self.input
        if self.turns:
            return "\n".join(t.user for t in self.turns)
        return self.hidden_goal or ""


def _parse_line(path: str | Path, lineno: int, line: str, errors: list[str]) -> Case | None:
    try:
        data = json.loads(line)
    except json.JSONDecodeError as exc:
        errors.append(f"{path}:{lineno}: invalid JSON: {exc.msg}")
        return None
    if not isinstance(data, dict):
        errors.append(f"{path}:{lineno}: expected a JSON object")
        return None
    try:
        return Case.model_validate(data)
    except ValidationError as exc:
        for err in exc.errors():
            loc = ".".join(str(p) for p in err["loc"]) or "<row>"
            errors.append(f"{path}:{lineno}: {loc}: {err['msg']}")
        return None


# --------------------------------------------------------------------------- Observation


class ToolCall(_Open):
    name: str
    args: dict[str, Any] = {}
    result: Any = None
    error: str | None = None
    agent: str | None = None


class Chunk(_Open):
    id: str
    text: str
    score: float | None = None
    metadata: dict[str, Any] = {}


class Span(_Open):
    name: str
    latency_s: float


class Observation(_Open):
    text: str = ""
    messages: list[dict[str, str]] = []
    tool_calls: list[ToolCall] = []
    retrieved: list[Chunk] = []
    state: dict[str, Any] = {}
    blocked: bool | None = None
    events: list[dict[str, Any]] = []
    transcript: str | None = None
    spans: list[Span] = []
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float | None = None
    latency_s: float = 0.0
    ttft_s: float | None = None
    cache_hit: bool | None = None
    trace_id: str | None = None
    error: str | None = None
    raw: Any = None


# --------------------------------------------------------------------------- results


class MetricResult(_Open):
    metric: str
    value: float | None = None
    passed: bool | None = None
    threshold: float | None = None
    details: dict[str, Any] = {}
    skipped: str | None = None


class ResultRow(_Open):
    run_id: str
    suite: str
    case_id: str | None
    repeat: int | None
    metric: str
    value: float | None
    passed: bool | None
    threshold: float | None
    skipped: str | None
    details: dict[str, Any]
    tags: list[str]
    language: str | None
    split: str | None
    must_pass: bool
    latency_s: float | None
    tokens: int | None
    cost_usd: float | None
    trace_id: str | None


class CaseRun(NamedTuple):
    """A case with its observations and metric results across repeats.

    `results[i]` were computed on repeat `repeats[i]` (see docs/DECISIONS.md).
    """

    case: Case
    observations: list[Observation]
    results: list[MetricResult]
    repeats: list[int]

    def by_metric(self, metric: str) -> list[MetricResult]:
        return [r for r in self.results if r.metric == metric]

    def passed_per_repeat(self, metric: str) -> list[bool]:
        return [bool(r.passed) for r in self.by_metric(metric) if r.skipped is None]


class RunInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    started: str
    finished: str | None = None
    git_commit: str | None = None
    git_dirty: bool | None = None
    profile: str | None = None
    cli_overrides: dict[str, Any] = {}
    target: str | None = None
    target_config_hash: str | None = None
    roles: dict[str, dict[str, Any]] = {}
    ledger: dict[str, dict[str, Any]] = {}
    prompt_hashes: dict[str, str] = {}
    datasets: dict[str, dict[str, Any]] = {}
    judges: dict[str, dict[str, Any]] = {}
    suites: dict[str, list[str]] = {}
    repeats: int = 1
    seed: int = 42
    evalkit_version: str = ""
    engines: dict[str, str] = {}
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    paid_cost_usd: float = 0.0
    baseline: str | None = None
    gate: dict[str, Any] = {}
