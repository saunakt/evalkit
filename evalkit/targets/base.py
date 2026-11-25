"""Target protocol, TargetContext and target loading (Plan.md §7.1)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from evalkit.config import read_yaml
from evalkit.registry import TARGETS, load_all
from evalkit.schema import Case, Chunk, Observation
from evalkit.util import ConfigError, stable_hash

HOOKS = ("reset", "state", "ingest", "update_doc", "stream")


class TargetContext(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str | None = None
    messages: list[dict[str, Any]] = []
    session_id: str | None = None
    faults: list[dict[str, Any]] = []
    max_steps: int | None = None
    extra_context: list[Chunk] = []
    role: str | None = None
    params: dict[str, Any] = {}
    now: str | None = None


@runtime_checkable
class Target(Protocol):
    name: str

    def call(self, case: Case, ctx: TargetContext) -> Observation: ...


# Optional hook signatures, for reference (attributes may be None when absent):
#   reset(seed: int) -> None
#   state() -> dict[str, Any]
#   ingest(docs_dir: str) -> list[Chunk]
#   update_doc(doc_id: str, text: str | None) -> None
#   stream(case: Case, ctx: TargetContext) -> Iterator[str]
StreamFn = Iterator[str]


def has_hook(target: Any, name: str) -> bool:
    """True if the target provides the optional hook `name`."""
    return callable(getattr(target, name, None))


def coerce_observation(result: Any) -> Observation:
    """Accept an Observation, a dict (validated) or a str (→ text)."""
    if isinstance(result, Observation):
        return result
    if isinstance(result, dict):
        return Observation.model_validate(result)
    if isinstance(result, str):
        return Observation(text=result)
    if result is None:
        return Observation()
    raise TypeError(f"target returned unsupported type {type(result).__name__}")


def load_target(path_or_cfg: str | Path | dict[str, Any], root: Path) -> Any:
    """Build a target from a YAML file (relative to root) or an inline mapping."""
    load_all()
    if isinstance(path_or_cfg, dict):
        cfg = dict(path_or_cfg)
    else:
        p = Path(path_or_cfg)
        cfg = read_yaml(p if p.is_absolute() else root / p)
    ttype = cfg.get("type")
    if not ttype:
        raise ConfigError(f"target {path_or_cfg}: missing 'type'")
    factory = TARGETS.get(str(ttype))
    target = factory(cfg, root)
    target.config_hash = stable_hash(cfg)
    target.config = cfg
    return target
