"""`evalctl providers check`: reachability of configured providers (Plan.md §7.2)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from evalkit.llm.roles import LLMRuntime, is_free


@dataclass
class ProviderStatus:
    name: str
    type: str
    reachable: bool
    latency_s: float | None
    free: bool
    supports_json: bool
    roles: list[str] = field(default_factory=list)
    detail: str = ""


def roles_by_provider(rt: LLMRuntime) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for role, rm in rt.config.models.items():
        if rm is not None:
            out.setdefault(rm.provider, []).append(role)
    return out


def _probe(rt: LLMRuntime, name: str) -> str:
    """Cheapest possible check: list models when possible, else a tiny completion."""
    client: Any = rt.provider_client(name)
    if hasattr(client, "list_models"):
        models = client.list_models()
        return f"{len(models)} models listed"
    if hasattr(client, "_client") and hasattr(client._client, "models"):
        client._client.models.list()
        return "models listed"
    resp = client.complete(
        system="", messages=[{"role": "user", "content": "Reply OK"}], model="probe", max_tokens=5
    )
    return f"replied {resp.text[:20]!r}"


def check_provider(rt: LLMRuntime, name: str, roles: list[str]) -> ProviderStatus:
    cfg = rt.provider_config(name)
    start = time.perf_counter()
    try:
        detail, ok = _probe(rt, name), True
    except Exception as exc:  # report any failure as unreachable
        detail, ok = f"{type(exc).__name__}: {exc}"[:200], False
    return ProviderStatus(
        name=name,
        type=cfg.type,
        reachable=ok,
        latency_s=round(time.perf_counter() - start, 3) if ok else None,
        free=is_free(cfg),
        supports_json=cfg.supports_json or cfg.type == "fake",
        roles=sorted(roles),
        detail=detail,
    )


def check_providers(rt: LLMRuntime, include_unused: bool = False) -> list[ProviderStatus]:
    used = roles_by_provider(rt)
    names = sorted(rt.config.providers) if include_unused else sorted(used)
    return [check_provider(rt, n, used.get(n, [])) for n in names]
