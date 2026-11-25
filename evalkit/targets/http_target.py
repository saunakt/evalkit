"""`type: http` target: templated request, dotted-path response mapping (Plan.md §7.1)."""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
from jinja2 import Environment, StrictUndefined

from evalkit.llm.client import RetryableError, with_retries
from evalkit.llm.openai_compatible import parse_sse_line
from evalkit.registry import target
from evalkit.schema import Case, Observation
from evalkit.targets.base import TargetContext, coerce_observation
from evalkit.util import ConfigError, get_path

_ENV = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")
_JINJA = Environment(undefined=StrictUndefined, autoescape=False)
_JINJA.filters["tojson"] = lambda v: json.dumps(v, ensure_ascii=False, default=str)


def _env(text: str) -> str:
    return _ENV.sub(lambda m: os.environ.get(m.group(1), ""), text)


class HttpTarget:
    def __init__(
        self, cfg: dict[str, Any], root: Path, transport: httpx.BaseTransport | None = None
    ) -> None:
        if "url" not in cfg:
            raise ConfigError("http target needs 'url'")
        self.name = str(cfg.get("name", cfg["url"]))
        self.url = _env(str(cfg["url"]))
        self.method = str(cfg.get("method", "POST")).upper()
        self.headers = {k: _env(str(v)) for k, v in (cfg.get("headers") or {}).items()}
        self.body = str(cfg.get("body", '{"input": {{ input | tojson }}}'))
        self.mapping: dict[str, str] = dict(cfg.get("response") or {"text": "text"})
        self.retries = int(cfg.get("retries", 3))
        self.supports_media = bool(cfg.get("supports_media", False))
        self.thread_safe = True
        self.uses_roles: list[str] = []
        self._http = httpx.Client(timeout=float(cfg.get("timeout_s", 60)), transport=transport)
        self._sse = cfg.get("stream") == "sse"
        self._reset_url = cfg.get("reset_url")
        self._state_url = cfg.get("state_url")
        self.reset = self._reset if self._reset_url else None
        self.state = self._state if self._state_url else None
        self.stream = self._stream if self._sse else None

    def render_body(self, case: Case, ctx: TargetContext) -> Any:
        rendered = _JINJA.from_string(self.body).render(
            case=case.model_dump(by_alias=True),
            ctx=ctx.model_dump(),
            input=case.input if case.input is not None else case.text_input(),
            messages=ctx.messages,
        )
        try:
            return json.loads(rendered)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"http target body did not render to JSON: {exc}") from exc

    def _send(self, body: Any) -> httpx.Response:
        try:
            resp = self._http.request(self.method, self.url, headers=self.headers, json=body)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise RetryableError(str(exc)) from exc
        if resp.status_code in (429, 502, 503, 504):
            raise RetryableError(f"HTTP {resp.status_code}")
        return resp

    def call(self, case: Case, ctx: TargetContext) -> Observation:
        body = self.render_body(case, ctx)
        start = time.perf_counter()
        try:
            resp = with_retries(lambda: self._send(body), max_tries=self.retries, base_s=0.2)
        except RetryableError as exc:
            return Observation(error=f"http error: {exc}", latency_s=time.perf_counter() - start)
        latency = time.perf_counter() - start
        if resp.status_code >= 400:
            return Observation(
                error=f"HTTP {resp.status_code}",
                latency_s=latency,
                raw={"status": resp.status_code},
            )
        try:
            data = resp.json()
        except json.JSONDecodeError:
            return Observation(text=resp.text, latency_s=latency, error="malformed response")
        obs = coerce_observation(self.map_response(data))
        obs.latency_s = obs.latency_s or latency
        obs.raw = {"status": resp.status_code, "body": data}
        return obs

    def map_response(self, data: Any) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for field_name, path in self.mapping.items():
            value = get_path(data, path)
            if value is not None:
                out[field_name] = value
        out.setdefault("text", "")
        if not isinstance(out["text"], str):
            out["text"] = json.dumps(out["text"])
        return out

    def _stream(self, case: Case, ctx: TargetContext) -> Iterator[str]:
        body = self.render_body(case, ctx)
        with self._http.stream(self.method, self.url, headers=self.headers, json=body) as resp:
            for line in resp.iter_lines():
                piece = parse_sse_line(line)
                if piece is None:
                    break
                if piece:
                    yield piece

    def _reset(self, seed: int) -> None:
        self._http.post(str(self._reset_url), json={"seed": seed}, headers=self.headers)

    def _state(self) -> dict[str, Any]:
        data: dict[str, Any] = self._http.get(str(self._state_url), headers=self.headers).json()
        return data


@target("http")
def make_http_target(cfg: dict[str, Any], root: Path) -> HttpTarget:
    return HttpTarget(cfg, root)
