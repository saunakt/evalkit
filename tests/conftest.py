from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Reply:
    status: int = 200
    body: Any = None
    delay_s: float = 0.0
    sse: list[str] | None = None
    headers: dict[str, str] = field(default_factory=dict)


@dataclass
class FakeServer:
    """In-process HTTP server. Queue replies; inspect `requests` afterwards."""

    url: str
    replies: list[Reply] = field(default_factory=list)
    default: Reply = field(default_factory=Reply)
    requests: list[dict[str, Any]] = field(default_factory=list)

    def queue(self, *replies: Reply) -> None:
        self.replies.extend(replies)


def _handler(server: FakeServer) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            pass

        def _reply(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            try:
                body = json.loads(raw) if raw else None
            except json.JSONDecodeError:
                body = raw.decode()
            server.requests.append(
                {
                    "method": self.command,
                    "path": self.path,
                    "body": body,
                    "headers": dict(self.headers),
                }
            )
            reply = server.replies.pop(0) if server.replies else server.default
            if reply.delay_s:
                time.sleep(reply.delay_s)
            self.send_response(reply.status)
            if reply.sse is not None:
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for line in reply.sse:
                    self.wfile.write(f"data: {line}\n\n".encode())
                    self.wfile.flush()
                return
            payload = json.dumps(reply.body if reply.body is not None else {}).encode()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            for k, v in reply.headers.items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(payload)

        do_GET = do_POST = _reply

    return Handler


@pytest.fixture
def http_server() -> Iterator[FakeServer]:
    holder = FakeServer(url="")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _handler(holder))
    holder.url = f"http://127.0.0.1:{httpd.server_address[1]}"
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield holder
    finally:
        httpd.shutdown()
        httpd.server_close()


def chat_reply(text: str, usage: dict[str, int] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"choices": [{"message": {"role": "assistant", "content": text}}]}
    if usage is not None:
        out["usage"] = usage
    return out
