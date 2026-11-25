"""Built-in OpenAI-compatible chat client on httpx (OpenRouter, Ollama, vLLM, LM Studio…)."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx

from evalkit.config import ProviderConfig
from evalkit.llm.client import (
    LLMResponse,
    ResponseFormat,
    RetryableError,
    estimate_tokens,
    with_retries,
)
from evalkit.llm.roles import is_free
from evalkit.registry import provider
from evalkit.util import EvalkitError

RETRY_STATUS = {429, 500, 502, 503, 504}


class ProviderError(EvalkitError):
    exit_code = 3


class OpenAICompatibleClient:
    def __init__(
        self, name: str, cfg: ProviderConfig, transport: httpx.BaseTransport | None = None
    ):
        if not cfg.base_url:
            raise ProviderError(f"providers.{name}.base_url is required")
        self.name = name
        self.cfg = cfg
        headers = {"Content-Type": "application/json", **cfg.headers}
        if cfg.api_key and cfg.api_key.lower() != "none":
            headers["Authorization"] = f"Bearer {cfg.api_key}"
        self._http = httpx.Client(
            base_url=cfg.base_url.rstrip("/"),
            headers=headers,
            timeout=cfg.timeout_s,
            transport=transport,
        )
        limit = cfg.max_concurrency or (4 if is_free(cfg) else 8)
        self._sem = threading.BoundedSemaphore(limit)

    def _payload(
        self,
        system: str,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float,
        max_tokens: int,
        response_format: ResponseFormat,
        stream: bool = False,
    ) -> dict[str, Any]:
        msgs = ([{"role": "system", "content": system}] if system else []) + list(messages)
        body: dict[str, Any] = {
            "model": model,
            "messages": msgs,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format == "json" and self.cfg.supports_json:
            body["response_format"] = {"type": "json_object"}
        if stream:
            body["stream"] = True
        return body

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        try:
            resp = self._http.post("/chat/completions", json=body)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise RetryableError(f"{self.name}: {type(exc).__name__}: {exc}") from exc
        if resp.status_code in RETRY_STATUS:
            raise RetryableError(f"{self.name}: HTTP {resp.status_code}")
        if resp.status_code >= 400:
            raise ProviderError(f"{self.name}: HTTP {resp.status_code}: {resp.text[:300]}")
        data: dict[str, Any] = resp.json()
        return data

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
        body = self._payload(system, messages, model, temperature, max_tokens, response_format)
        start = time.perf_counter()
        with self._sem:
            data = with_retries(
                lambda: self._post(body),
                max_tries=self.cfg.max_retries,
                base_s=self.cfg.retry_base_s,
            )
        latency = time.perf_counter() - start
        text = str(((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
        usage = data.get("usage") or {}
        estimated = "prompt_tokens" not in usage
        prompt_text = system + "".join(str(m.get("content", "")) for m in messages)
        return LLMResponse(
            text=text,
            tokens_in=int(usage.get("prompt_tokens", estimate_tokens(prompt_text))),
            tokens_out=int(usage.get("completion_tokens", estimate_tokens(text))),
            tokens_estimated=estimated,
            latency_s=latency,
            model=str(data.get("model") or model),
            provider=self.name,
        )

    def stream(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        **_: Any,
    ) -> Iterator[str]:
        body = self._payload(system, messages, model, temperature, max_tokens, "text", True)
        with self._sem, self._http.stream("POST", "/chat/completions", json=body) as resp:
            if resp.status_code >= 400:
                raise ProviderError(f"{self.name}: HTTP {resp.status_code}")
            for line in resp.iter_lines():
                piece = parse_sse_line(line)
                if piece is None:
                    break
                if piece:
                    yield piece

    def list_models(self) -> list[str]:
        resp = self._http.get("/models")
        resp.raise_for_status()
        return sorted(str(m.get("id")) for m in resp.json().get("data", []))


def parse_sse_line(line: str) -> str | None:
    """Content delta of one SSE line; "" for non-content lines; None at [DONE]."""
    line = line.strip()
    if not line.startswith("data:"):
        return ""
    payload = line[5:].strip()
    if payload == "[DONE]":
        return None
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return ""
    delta = ((data.get("choices") or [{}])[0].get("delta") or {}).get("content")
    return str(delta) if delta else ""


@provider("openai_compatible")
def make_openai_compatible(name: str, cfg: ProviderConfig, root: Path) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(name, cfg)
