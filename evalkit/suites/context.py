"""SuiteContext and MetricContext: what suites and metrics can reach (Plan.md §6)."""

from __future__ import annotations

import threading
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evalkit.config import Project, SuiteConfig
from evalkit.judges.runner import JudgeRunner
from evalkit.llm.client import RoleClient
from evalkit.llm.roles import LLMRuntime
from evalkit.reporting.store import StoredRun
from evalkit.schema import Case, Observation, ResultRow
from evalkit.targets.base import TargetContext, has_hook


@dataclass
class SuiteContext:
    """Everything a suite needs to run its cases."""

    name: str
    project: Project
    sc: SuiteConfig
    params: dict[str, Any]
    target: Any
    runtime: LLMRuntime
    judge: JudgeRunner
    run_dir: Path
    seed: int
    repeats: int
    now: str | None
    baseline: StoredRun | None = None
    run_rows: Callable[[], list[ResultRow]] = field(default=lambda: [])
    targets: dict[str, Any] = field(default_factory=dict)
    memo: dict[str, Any] = field(default_factory=dict)
    model: str | None = (
        None  # SUT model override (migration, fallback)  # per-suite scratch (e.g. ingest)
    )
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def llm(self, role: str) -> RoleClient:
        return self.runtime.client(role)

    def target_ctx(self, case: Case, **over: Any) -> TargetContext:
        params = {**self.params.get("target_params", {}), **over.pop("params", {})}
        base: dict[str, Any] = {
            "now": self.now,
            "faults": list(case.faults),
            "max_steps": case.max_steps,
            "params": params,
        }
        if self.model:
            base["model"] = self.model
        base.update(over)
        return TargetContext(**base)

    def call(self, case: Case, target: Any = None, **over: Any) -> Observation:
        """Call the target; exceptions become `obs.error` (the run continues)."""
        tgt = target or self.target
        try:
            return tgt.call(case, self.target_ctx(case, **over))  # type: ignore[no-any-return]
        except Exception as exc:
            tb = traceback.format_exc(limit=3)
            return Observation(error=f"{type(exc).__name__}: {exc}", raw={"traceback": tb})

    def recall(self, case: Case, target: Any = None, **over: Any) -> Observation:
        """Re-call the target from a fresh sandbox (for metrics that call it again)."""
        self.reset(self.seed, target)
        return self.call(case, target, **over)

    def reset(self, seed: int, target: Any = None) -> None:
        tgt = target or self.target
        if has_hook(tgt, "reset"):
            tgt.reset(seed)

    def state(self, target: Any = None) -> dict[str, Any]:
        tgt = target or self.target
        return dict(tgt.state()) if has_hook(tgt, "state") else {}

    def resolve_target(self, ref: str | dict[str, Any]) -> Any:
        """Load (and cache) another target by yaml path, e.g. for finetune variants."""
        from evalkit.targets.base import load_target

        key = ref if isinstance(ref, str) else repr(sorted(ref.items()))
        with self._lock:
            if key not in self.targets:
                self.targets[key] = load_target(ref, self.project.root)
            return self.targets[key]


@dataclass
class MetricContext:
    """Passed to every metric (Plan.md §6)."""

    params: dict[str, Any]
    judge: JudgeRunner
    suite: SuiteContext
    repeat: int = 0
    metric_id: str = ""

    @property
    def llm(self) -> LLMRuntime:
        return self.suite.runtime

    @property
    def target(self) -> Any:
        return self.suite.target

    @property
    def baseline(self) -> StoredRun | None:
        return self.suite.baseline

    @property
    def run_dir(self) -> Path:
        return self.suite.run_dir

    @property
    def seed(self) -> int:
        return self.suite.seed

    def p(self, key: str, default: Any = None) -> Any:
        """Metric param: `params.<metric_id>.<key>` → `params.<key>` → default."""
        own = self.params.get(self.metric_id)
        if isinstance(own, dict) and key in own:
            return own[key]
        return self.params.get(key, default)
