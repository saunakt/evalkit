"""Fault injection specs and helpers (Plan.md §7.3 "Faults")."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

FaultError = Literal[
    "timeout", "malformed", "empty", "http_500", "http_429", "slow", "down", "stream_drop"
]


class FaultSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    tool: str | None = None
    provider: str | None = None
    component: str | None = None
    error: FaultError
    times: int = 1
    delay_s: float = 0.0


class ToolFault(Exception):
    """Raised by a faulted tool; `kind` is the fault error name."""

    def __init__(self, kind: str, message: str = "") -> None:
        super().__init__(message or f"injected fault: {kind}")
        self.kind = kind


def parse_faults(faults: list[dict[str, Any]]) -> list[FaultSpec]:
    return [FaultSpec.model_validate(f) for f in faults]


def find_fault(
    faults: list[dict[str, Any]],
    *,
    tool: str | None = None,
    provider: str | None = None,
    component: str | None = None,
) -> FaultSpec | None:
    for spec in parse_faults(faults):
        if tool and spec.tool == tool:
            return spec
        if provider and spec.provider == provider:
            return spec
        if component and spec.component == component:
            return spec
    return None


class FaultCounter:
    """Counts how many times each fault fired so `times` is honoured per case."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._fired: dict[str, int] = {}

    def should_fire(self, spec: FaultSpec) -> bool:
        key = spec.model_dump_json()
        with self._lock:
            n = self._fired.get(key, 0)
            if n >= spec.times:
                return False
            self._fired[key] = n + 1
            return True


def apply_tool_fault(
    spec: FaultSpec | None,
    fn: Callable[..., Any],
    counter: FaultCounter | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> Callable[..., Any]:
    """Wrap a tool function so it misbehaves per `spec` (for python targets)."""
    if spec is None:
        return fn
    counter = counter or FaultCounter()

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        if not counter.should_fire(spec):
            return fn(*args, **kwargs)
        kind = spec.error
        if kind == "slow":
            sleep(spec.delay_s or 3.0)
            return fn(*args, **kwargs)
        if kind == "empty":
            return None
        if kind == "malformed":
            return "{not valid json"
        raise ToolFault(kind)

    return wrapped
