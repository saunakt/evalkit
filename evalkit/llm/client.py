"""LLMClient protocol, retry helper, response cache wiring and the cost ledger (Plan.md §7.2)."""

from __future__ import annotations

import logging
import math
import random
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

from evalkit.config import Price
from evalkit.env.cache import ResponseCache

log = logging.getLogger("evalkit.llm")
T = TypeVar("T")
ResponseFormat = Literal["text", "json"]


class LLMResponse(BaseModel):
    text: str
    tokens_in: int = 0
    tokens_out: int = 0
    tokens_estimated: bool = False
    latency_s: float = 0.0
    model: str = ""
    provider: str = ""
    role: str | None = None
    cost_usd: float | None = None
    cached: bool = False


@runtime_checkable
class LLMClient(Protocol):
    def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        response_format: ResponseFormat = "text",
        meta: dict[str, Any] | None = None,
    ) -> LLMResponse: ...


class CostCapReached(Exception):
    """Raised before a paid call once `run.max_cost_usd` is reached."""


class RetryableError(Exception):
    """A transient provider error (429, 5xx, connection, timeout)."""


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / 4) if text else 0


def with_retries(
    fn: Callable[[], T],
    *,
    max_tries: int = 5,
    base_s: float = 0.5,
    cap_s: float = 30.0,
    sleep: Callable[[float], None] = time.sleep,
    rng: random.Random | None = None,
) -> T:
    """Exponential backoff with jitter on RetryableError (max tries, capped delay)."""
    rng = rng or random.Random(0)
    for attempt in range(1, max_tries + 1):
        try:
            return fn()
        except RetryableError:
            if attempt == max_tries:
                raise
            delay = min(cap_s, base_s * 2 ** (attempt - 1))
            sleep(delay * (0.5 + rng.random() / 2))
    raise AssertionError("unreachable")


def price_call(price: Price | None, tokens_in: int, tokens_out: int) -> float | None:
    if price is None:
        return None
    return tokens_in / 1e6 * price.input_per_mtok + tokens_out / 1e6 * price.output_per_mtok


# --------------------------------------------------------------------------- ledger


@dataclass
class RoleTotals:
    calls: int = 0
    paid_calls: int = 0
    free_calls: int = 0
    cached_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    paid_cost_usd: float = 0.0
    unpriced_paid_calls: int = 0
    models: set[str] = field(default_factory=set)

    def as_dict(self) -> dict[str, Any]:
        out = dict(self.__dict__)
        out["models"] = sorted(self.models)
        out["paid_cost_usd"] = round(self.paid_cost_usd, 6)
        return out


class CostLedger:
    """Thread-safe record of every model call, per role, split paid vs free."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._roles: dict[str, RoleTotals] = {}
        self._warned: set[str] = set()

    def record(self, role: str, resp: LLMResponse, *, paid: bool) -> None:
        with self._lock:
            tot = self._roles.setdefault(role, RoleTotals())
            tot.calls += 1
            tot.models.add(f"{resp.provider}:{resp.model}")
            if resp.cached:
                tot.cached_calls += 1
                return
            tot.tokens_in += resp.tokens_in
            tot.tokens_out += resp.tokens_out
            if not paid:
                tot.free_calls += 1
                return
            tot.paid_calls += 1
            if resp.cost_usd is None:
                tot.unpriced_paid_calls += 1
                if resp.model not in self._warned:
                    self._warned.add(resp.model)
                    log.warning("no price configured for paid model %s; cost_usd=null", resp.model)
            else:
                tot.paid_cost_usd += resp.cost_usd

    def paid_spend(self) -> float:
        with self._lock:
            return sum(t.paid_cost_usd for t in self._roles.values())

    def paid_calls(self) -> int:
        with self._lock:
            return sum(t.paid_calls for t in self._roles.values())

    def summary(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {role: tot.as_dict() for role, tot in sorted(self._roles.items())}

    def total_tokens(self) -> int:
        with self._lock:
            return sum(t.tokens_in + t.tokens_out for t in self._roles.values())


# --------------------------------------------------------------------------- role client


class RoleClient:
    """Wraps a provider client for one role: cache, pricing, ledger, cost cap."""

    def __init__(
        self,
        *,
        role: str,
        provider_name: str,
        client: LLMClient,
        model: str,
        free: bool,
        price: Price | None,
        ledger: CostLedger,
        cache: ResponseCache | None,
        max_cost_usd: float | None,
    ) -> None:
        self.role = role
        self.provider_name = provider_name
        self.inner = client
        self.model = model
        self.free = free
        self.price = price
        self.ledger = ledger
        self.cache = cache
        self.max_cost_usd = max_cost_usd

    def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        response_format: ResponseFormat = "text",
        meta: dict[str, Any] | None = None,
    ) -> LLMResponse:
        model = model or self.model
        meta = {**(meta or {}), "role": self.role}
        key = None
        if self.cache is not None:
            key = self.cache.key(
                self.provider_name, model, system, messages, temperature, max_tokens
            )
            hit = self.cache.get(key)
            if hit is not None:
                resp = LLMResponse.model_validate({**hit, "cached": True, "role": self.role})
                self.ledger.record(self.role, resp, paid=not self.free)
                return resp
        capped = not self.free and self.max_cost_usd is not None
        if capped and self.ledger.paid_spend() >= (self.max_cost_usd or 0.0):
            raise CostCapReached(f"paid spend reached max_cost_usd={self.max_cost_usd}")
        resp = self.inner.complete(
            system=system,
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            meta=meta,
        )
        resp.role, resp.provider = self.role, self.provider_name
        resp.cost_usd = (
            0.0 if self.free else price_call(self.price, resp.tokens_in, resp.tokens_out)
        )
        self.ledger.record(self.role, resp, paid=not self.free)
        if self.cache is not None and key is not None:
            self.cache.put(key, resp.model_dump(exclude={"cached", "role"}))
        return resp

    def stream(
        self, *, system: str, messages: list[dict[str, Any]], model: str | None = None, **kw: Any
    ) -> Iterator[str]:
        fn = getattr(self.inner, "stream", None)
        if fn is None:
            raise NotImplementedError(f"provider {self.provider_name} does not support streaming")
        it: Iterator[str] = fn(system=system, messages=messages, model=model or self.model, **kw)
        return it
