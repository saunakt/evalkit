"""Optional Anthropic provider (extra `[anthropic]`); the SDK is imported lazily."""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from evalkit.config import ProviderConfig
from evalkit.llm.client import LLMResponse, ResponseFormat
from evalkit.registry import provider
from evalkit.util import optional_import


class AnthropicClient:
    """Thin wrapper over `anthropic.Anthropic().messages`.

    Current Claude models reject sampling parameters, so `temperature` is not sent
    (see docs/DECISIONS.md). Retries use the SDK's own `max_retries`.
    """

    def __init__(self, name: str, cfg: ProviderConfig, sdk: Any = None) -> None:
        sdk = sdk or optional_import("anthropic")
        kwargs: dict[str, Any] = {"timeout": cfg.timeout_s, "max_retries": cfg.max_retries}
        if cfg.api_key:
            kwargs["api_key"] = cfg.api_key
        if cfg.base_url:
            kwargs["base_url"] = cfg.base_url
        if cfg.headers:
            kwargs["default_headers"] = cfg.headers
        self.name = name
        self._client = sdk.Anthropic(**kwargs)

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
    ) -> LLMResponse:
        start = time.perf_counter()
        msg = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[m for m in messages if m.get("role") != "system"],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        return LLMResponse(
            text=text,
            tokens_in=int(msg.usage.input_tokens),
            tokens_out=int(msg.usage.output_tokens),
            latency_s=time.perf_counter() - start,
            model=str(msg.model),
            provider=self.name,
        )

    def stream(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        model: str,
        max_tokens: int = 1024,
        **_: Any,
    ) -> Iterator[str]:
        with self._client.messages.stream(
            model=model, max_tokens=max_tokens, system=system, messages=messages
        ) as stream:
            yield from stream.text_stream


@provider("anthropic", requires_extra="anthropic")
def make_anthropic(name: str, cfg: ProviderConfig, root: Path) -> AnthropicClient:
    return AnthropicClient(name, cfg)
