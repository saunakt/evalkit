"""One decorator registry for metrics, suites, targets, engines and providers (Plan.md §6)."""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar

from evalkit.util import ConfigError, did_you_mean

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])

Kind = Literal["deterministic", "judge", "statistical", "human"]
Scope = Literal["case", "dataset"]
Direction = Literal["higher", "lower"]


@dataclass(frozen=True)
class MetricSpec:
    id: str
    fn: Callable[..., Any]
    suites: tuple[str, ...]
    kind: Kind
    scope: Scope = "case"
    needs: tuple[str, ...] = ()
    engines: tuple[str, ...] = ("native",)
    direction: Direction = "higher"
    unit: str = "ratio"
    description: str = ""
    requires_extra: str | None = None  # pyproject extra this metric needs (install.yaml)

    def applies_to(self, suite: str) -> bool:
        return suite in self.suites or "*" in self.suites


class Registry(Generic[T]):
    """Name → object mapping with helpful lookup errors."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, T] = {}

    def add(self, name: str, item: T) -> None:
        if name in self._items and self._items[name] is not item:
            raise ValueError(f"duplicate {self.kind} {name!r}")
        self._items[name] = item

    def get(self, name: str) -> T:
        try:
            return self._items[name]
        except KeyError:
            hint = did_you_mean(name, sorted(self._items))
            raise ConfigError(f"unknown {self.kind} {name!r}{hint}") from None

    def __contains__(self, name: object) -> bool:
        return name in self._items

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._items))

    def __len__(self) -> int:
        return len(self._items)

    def names(self) -> list[str]:
        return sorted(self._items)

    def items(self) -> list[tuple[str, T]]:
        return sorted(self._items.items())


METRICS: Registry[MetricSpec] = Registry("metric")
SUITES: Registry[Any] = Registry("suite")
TARGETS: Registry[Any] = Registry("target")
ENGINES: Registry[Any] = Registry("engine")
PROVIDERS: Registry[Any] = Registry("provider")
# (metric_id, engine) -> alternative implementation
IMPLS: dict[tuple[str, str], Callable[..., Any]] = {}
# (registry kind, name) -> pyproject extra the registration needs (None = core only)
REQUIRES_EXTRA: dict[tuple[str, str], str | None] = {}


def metric(
    id: str,
    *,
    suites: tuple[str, ...] | list[str],
    kind: Kind,
    scope: Scope = "case",
    needs: tuple[str, ...] | list[str] = (),
    engines: tuple[str, ...] | list[str] = ("native",),
    direction: Direction = "higher",
    unit: str = "ratio",
    description: str = "",
    requires_extra: str | None = None,
) -> Callable[[F], F]:
    """Register a native metric implementation."""

    def deco(fn: F) -> F:
        spec = MetricSpec(
            id=id,
            fn=fn,
            suites=tuple(suites),
            kind=kind,
            scope=scope,
            needs=tuple(needs),
            engines=tuple(engines),
            direction=direction,
            unit=unit,
            description=description or (fn.__doc__ or "").strip().split("\n")[0],
            requires_extra=requires_extra,
        )
        METRICS.add(id, spec)
        IMPLS[(id, "native")] = fn
        return fn

    return deco


def metric_impl(id: str, *, engine: str) -> Callable[[F], F]:
    """Register an engine-specific implementation of an existing metric id."""

    def deco(fn: F) -> F:
        IMPLS[(id, engine)] = fn
        return fn

    return deco


def resolve_impl(metric_id: str, engine: str = "native") -> Callable[..., Any]:
    try:
        return IMPLS[(metric_id, engine)]
    except KeyError:
        available = sorted(e for (m, e) in IMPLS if m == metric_id)
        raise ConfigError(
            f"metric {metric_id!r} has no {engine!r} implementation (available: {available})"
        ) from None


def _registrar(reg: Registry[Any]) -> Callable[..., Callable[[F], F]]:
    def register(name: str, *, requires_extra: str | None = None) -> Callable[[F], F]:
        def deco(obj: F) -> F:
            reg.add(name, obj)
            REQUIRES_EXTRA[(reg.kind, name)] = requires_extra
            return obj

        return deco

    return register


suite = _registrar(SUITES)
target = _registrar(TARGETS)
engine = _registrar(ENGINES)
provider = _registrar(PROVIDERS)

_PACKAGES = (
    "evalkit.llm",
    "evalkit.metrics",
    "evalkit.suites",
    "evalkit.targets",
    "evalkit.engines",
)
_loaded = False


def load_all() -> None:
    """Import every built-in module so its decorators run (idempotent)."""
    global _loaded
    if _loaded:
        return
    for pkg_name in _PACKAGES:
        pkg = importlib.import_module(pkg_name)
        for info in sorted(pkgutil.iter_modules(pkg.__path__), key=lambda i: i.name):
            importlib.import_module(f"{pkg_name}.{info.name}")
    _loaded = True


def metrics_for_suite(suite_name: str, *, cross_cutting: bool = False) -> list[MetricSpec]:
    """Metrics registered for a suite; cross-cutting (`"*"`) ones only when asked."""
    return [
        spec
        for _, spec in METRICS.items()
        if suite_name in spec.suites or (cross_cutting and "*" in spec.suites)
    ]


def requires_extra(kind: str, name: str) -> str | None:
    """Extra needed by a suite/target/engine/provider registration (None = core only)."""
    return REQUIRES_EXTRA.get((kind, name))
