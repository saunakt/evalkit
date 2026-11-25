"""Resolve a role → provider + model; holds the per-run LLM runtime (Plan.md §7.2)."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from evalkit.config import ROLES, EvalkitConfig, ProviderConfig
from evalkit.env.cache import ResponseCache
from evalkit.llm.client import CostLedger, LLMClient, RoleClient
from evalkit.registry import PROVIDERS, load_all
from evalkit.util import ConfigError

LOCAL_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1"}
NON_JUDGE_CACHE_ROLES = ("simulator", "perturb", "generator")


def is_free(cfg: ProviderConfig) -> bool:
    """Explicit `free`, else fake, else any localhost base_url."""
    if cfg.free is not None:
        return cfg.free
    if cfg.type == "fake":
        return True
    host = urlparse(cfg.base_url or "").hostname or ""
    return host in LOCAL_HOSTS


class LLMRuntime:
    """Provider clients, ledger and cache for one run (or one CLI command)."""

    def __init__(self, config: EvalkitConfig, root: Path, cache_mode: str | None = None) -> None:
        load_all()
        self.config = config
        self.root = root
        self.ledger = CostLedger()
        mode = cache_mode or config.run.cache
        self.cache_mode = mode
        self._cache: ResponseCache | None = None
        self._clients: dict[str, LLMClient] = {}
        self._lock = threading.Lock()

    # ----------------------------------------------------------------- resolution

    def role_model(self, role: str) -> tuple[str, str]:
        if role not in ROLES:
            raise ConfigError(f"unknown role {role!r}")
        rm = self.config.models.get(role)
        if rm is None:
            raise ConfigError(f"models.{role} is not configured")
        if rm.provider not in self.config.providers:
            raise ConfigError(f"models.{role}: unknown provider {rm.provider!r}")
        return rm.provider, rm.model

    def provider_config(self, name: str) -> ProviderConfig:
        try:
            return self.config.providers[name]
        except KeyError:
            raise ConfigError(f"unknown provider {name!r}") from None

    def provider_client(self, name: str) -> LLMClient:
        with self._lock:
            if name not in self._clients:
                cfg = self.provider_config(name)
                factory = PROVIDERS.get(cfg.type)
                self._clients[name] = factory(name, cfg, self.root)
            return self._clients[name]

    def cache(self) -> ResponseCache:
        with self._lock:
            if self._cache is None:
                mode = "read_write" if self.cache_mode == "off" else self.cache_mode
                path = self.root / self.config.run.cache_path
                self._cache = ResponseCache(path, mode)  # type: ignore[arg-type]
            return self._cache

    def _cache_for(self, role: str, cfg: ProviderConfig) -> ResponseCache | None:
        if cfg.type == "fake":
            return None
        if role == "judge" or role in NON_JUDGE_CACHE_ROLES:
            return self.cache()  # judges and LLM transforms are always cacheable
        return None if self.cache_mode == "off" else self.cache()

    def client(
        self, role: str, *, provider: str | None = None, model: str | None = None
    ) -> RoleClient:
        prov_name, role_model = self.role_model(role) if provider is None else (provider, "")
        cfg = self.provider_config(prov_name)
        model = model or role_model or cfg.default_model or ""
        if not model and cfg.type != "fake":
            raise ConfigError(f"models.{role}.model is empty (set it or its ${{ENV_VAR}})")
        return RoleClient(
            role=role,
            provider_name=prov_name,
            client=self.provider_client(prov_name),
            model=model or "fake",
            free=is_free(cfg),
            price=cfg.prices.get(model),
            ledger=self.ledger,
            cache=self._cache_for(role, cfg),
            max_cost_usd=self.config.run.max_cost_usd,
        )

    def roles_summary(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for role in ROLES:
            rm = self.config.models.get(role)
            if rm is None:
                continue
            cfg = self.config.providers.get(rm.provider)
            out[role] = {
                "provider": rm.provider,
                "model": rm.model,
                "free": is_free(cfg) if cfg else None,
            }
        return out

    def close(self) -> None:
        if self._cache is not None:
            self._cache.close()


_current: LLMRuntime | None = None


def set_runtime(rt: LLMRuntime | None) -> None:
    global _current
    _current = rt


def get_runtime() -> LLMRuntime:
    if _current is None:
        raise ConfigError("no LLM runtime is active (load a project first)")
    return _current


def client_for(role: str) -> tuple[RoleClient, str]:
    """(client, model) for a role in the active run."""
    client = get_runtime().client(role)
    return client, client.model
