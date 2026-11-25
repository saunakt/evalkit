"""Tiny python target used by framework tests."""

from __future__ import annotations

from typing import Any

STATE: dict[str, Any] = {}


def ask(payload: Any, ctx: Any) -> dict[str, Any]:
    text = payload if isinstance(payload, str) else payload[-1]["content"]
    STATE["last"] = text
    return {"text": f"echo: {text}", "tool_calls": [{"name": "lookup", "args": {"q": text}}]}


def ask_str(payload: Any, ctx: Any) -> str:
    return f"now={ctx.now}"


def reset(seed: int) -> None:
    STATE.clear()
    STATE["seed"] = seed


def state() -> dict[str, Any]:
    return dict(STATE)
