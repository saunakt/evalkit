"""`type: python` target: calls `module:function(input_or_messages, ctx)`."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

from evalkit.env.clock import frozen
from evalkit.registry import target
from evalkit.schema import Case, Observation
from evalkit.targets.base import TargetContext, coerce_observation
from evalkit.util import import_string

OPTIONAL = ("reset", "state", "ingest", "update_doc", "stream")


class PythonTarget:
    """Hooks are set as attributes only when configured (`has_hook` checks them)."""

    def __init__(self, cfg: dict[str, Any], root: Path) -> None:
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        self.name = str(cfg.get("name", cfg["entry"]))
        self.fn = import_string(cfg["entry"])
        self.freeze_time = bool(cfg.get("freeze_time", False))
        self.thread_safe = bool(cfg.get("thread_safe", False))
        self.supports_media = bool(cfg.get("supports_media", False))
        self.uses_roles = list(cfg.get("uses_roles", []))
        for hook in OPTIONAL:
            spec = cfg.get(hook)
            setattr(self, hook, import_string(spec) if spec else None)

    def call(self, case: Case, ctx: TargetContext) -> Observation:
        if ctx.messages:
            payload: Any = [*ctx.messages]
            if case.input and (not payload or payload[-1].get("content") != case.input):
                payload.append({"role": "user", "content": case.input})
        else:
            payload = case.input if case.input is not None else case.text_input()
        start = time.perf_counter()
        if self.freeze_time:
            with frozen(ctx.now):
                result = self.fn(payload, ctx)
        else:
            result = self.fn(payload, ctx)
        latency = time.perf_counter() - start
        obs = coerce_observation(result)
        if not obs.latency_s:
            obs.latency_s = latency
        return obs


@target("python")
def make_python_target(cfg: dict[str, Any], root: Path) -> PythonTarget:
    return PythonTarget(cfg, root)
