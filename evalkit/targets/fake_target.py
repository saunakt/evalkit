"""`type: fake` target: scripted observations for framework tests (Plan.md §7.1)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from evalkit.registry import target
from evalkit.schema import Case, Chunk, Observation
from evalkit.targets.base import TargetContext


class FakeTarget:
    """`responses: {case_id: Observation-dict}`; unknown ids echo the input.

    `param_responses: {"<param>=<value>": {case_id: Observation-dict}}` overrides a response
    when `ctx.params[<param>]` equals the value (e.g. skill_enabled=false).

    Optional: `states: {case_id: dict}` returned by `state()` after that case,
    `chunks: [Chunk]` for `ingest()`.
    """

    def __init__(self, cfg: dict[str, Any], root: Path) -> None:
        data = dict(cfg)
        if cfg.get("fixture"):
            data.update(yaml.safe_load((root / cfg["fixture"]).read_text(encoding="utf-8")) or {})
        self.name = str(data.get("name", "fake"))
        self.responses: dict[str, dict[str, Any]] = dict(data.get("responses") or {})
        self.states: dict[str, dict[str, Any]] = dict(data.get("states") or {})
        self.chunks = [Chunk.model_validate(c) for c in data.get("chunks") or []]
        self.param_responses: dict[str, dict[str, dict[str, Any]]] = dict(
            data.get("param_responses") or {}
        )
        self.thread_safe = False  # state() depends on the last case
        self.supports_media = bool(data.get("supports_media", False))
        self.uses_roles: list[str] = []
        self._last: str | None = None
        self.calls: list[tuple[str, TargetContext]] = []

    def call(self, case: Case, ctx: TargetContext) -> Observation:
        self._last = case.id
        self.calls.append((case.id, ctx))
        spec = self.responses.get(case.id)
        for key, mapping in self.param_responses.items():
            name, _, value = key.partition("=")
            if str(ctx.params.get(name)).lower() == value.lower() and case.id in mapping:
                spec = mapping[case.id]
        if spec is None:
            text = case.input if case.input is not None else case.text_input()
            return Observation(text=text)
        return Observation.model_validate(spec)

    def reset(self, seed: int) -> None:
        self._last = None

    def state(self) -> dict[str, Any]:
        return dict(self.states.get(self._last or "", {}))

    def ingest(self, docs_dir: str) -> list[Chunk]:
        return list(self.chunks)


@target("fake")
def make_fake_target(cfg: dict[str, Any], root: Path) -> FakeTarget:
    return FakeTarget(cfg, root)
