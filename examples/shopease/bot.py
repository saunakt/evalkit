"""ShopEase support agent: guardrail → agent loop (max 6 steps) → answer.

`ask(input_or_messages, ctx)` is the evalkit python target entry point; `reset`, `state`,
`ingest`, `update_doc` and `stream` are the optional hooks.
"""

from __future__ import annotations

import hashlib
import math
import threading
import time
from collections.abc import Iterator
from typing import Any

from evalkit.env.faults import FaultCounter, find_fault
from examples.shopease import guard, kb, tools
from examples.shopease.i18n import detect_language, localise
from examples.shopease.llm import Step, Turn, make_llm, system_prompt

MAX_STEPS = 6
AGENTS = {
    "search_policy": "policy_agent",
    "get_order": "orders_agent",
    "cancel_order": "orders_agent",
}
ROLE_ACCESS = {None: {"public"}, "customer": {"public"}, "agent": {"public", "internal"}}
_memory: dict[str, list[dict[str, str]]] = {}
_memory_lock = threading.Lock()
_local = threading.local()


def _messages(payload: Any) -> list[dict[str, str]]:
    if isinstance(payload, str):
        return [{"role": "user", "content": payload}]
    return [{"role": str(m["role"]), "content": str(m["content"])} for m in payload]


def _tokens(text: str) -> int:
    return math.ceil(len(text) / 4)


def _provider_fault(ctx: Any, events: list[dict[str, Any]]) -> str | None:
    """Simulate LLM provider faults: retries for 429, friendly errors otherwise."""
    spec = find_fault(ctx.faults, provider="llm")
    if spec is None:
        return None
    counter = getattr(_local, "provider_counter", None) or FaultCounter()
    _local.provider_counter = counter
    if spec.error == "http_429":
        retries = 0
        while counter.should_fire(spec):
            retries += 1
            events.append({"type": "retry", "attempt": retries, "error": "http_429"})
            if retries > int(ctx.params.get("max_retries", 3)):
                return "We're getting a lot of requests right now. Please try again in a minute."
        return None
    if spec.error == "slow" and spec.delay_s > float(ctx.params.get("timeout_s", 5)):
        events.append({"type": "timeout", "after_s": ctx.params.get("timeout_s", 5)})
        return "This is taking longer than usual. Please try again shortly."
    if spec.error in ("http_500", "timeout", "malformed", "down") and counter.should_fire(spec):
        return "Sorry, something went wrong on our side. Please try again in a moment."
    return None


def _filter(chunks: list[dict[str, Any]], ctx: Any) -> list[dict[str, Any]]:
    allowed = ROLE_ACCESS.get(ctx.role, {"public"})
    tenant = ctx.params.get("tenant", "shopease")
    out = [
        c
        for c in chunks
        if c["metadata"].get("access", "public") in allowed
        and c["metadata"].get("tenant", tenant) == tenant
    ]
    extra = [c.model_dump() if hasattr(c, "model_dump") else dict(c) for c in ctx.extra_context]
    return out + extra


def _run_tool(
    name: str,
    args: dict[str, Any],
    ctx: Any,
    events: list[dict[str, Any]],
    retrieved: list[dict[str, Any]],
    spans: list[dict[str, Any]],
) -> Step:
    if name in ctx.params.get("approval_tools", []):
        events.append({"type": "approval_requested", "tool": name})
        if ctx.params.get("approval") == "deny":
            events.append({"type": "approval_denied", "tool": name})
            return Step(name, args, None, "denied by approver")
        events.append({"type": "approval_granted", "tool": name})
    if name == "search_policy" and "retriever_k" in ctx.params:
        args = {**args, "k": int(ctx.params["retriever_k"])}
    start = time.perf_counter()
    result, error = tools.call_tool(name, args, ctx.faults)
    spans.append({"name": f"tool:{name}", "latency_s": time.perf_counter() - start})
    if name == "search_policy" and result is not None:
        result = _filter(result, ctx)
        retrieved.extend(result)
        spans.append({"name": "retrieval", "latency_s": time.perf_counter() - start})
    return Step(name, args, result, error)


def ask(payload: Any, ctx: Any) -> dict[str, Any]:
    t0 = time.perf_counter()
    messages = _messages(payload)
    if ctx.session_id:
        with _memory_lock:
            history = _memory.setdefault(ctx.params.get("customer_id", "anon"), [])
            messages = [*history, *messages]
    user = messages[-1]["content"] if messages else ""
    events: list[dict[str, Any]] = []
    spans: list[dict[str, Any]] = []
    g0 = time.perf_counter()
    blocked, reason = guard.check(user, ctx.params, ctx.faults)
    spans.append({"name": "guardrail", "latency_s": time.perf_counter() - g0})
    base = {
        "spans": spans,
        "events": events,
        "trace_id": _trace_id(messages),
        "tokens_in": _tokens(system_prompt() + user),
    }
    if blocked:
        events.append({"type": "guardrail_blocked", "reason": reason})
        return {
            **base,
            "text": localise("refusal", detect_language(user))
            or "Sorry, I can't help with that request.",
            "blocked": True,
            "tokens_out": 10,
        }
    failure = _provider_fault(ctx, events)
    if failure:
        return {**base, "text": failure, "blocked": False, "tokens_out": _tokens(failure)}
    llm = make_llm(ctx.params)
    turn = Turn(messages, params=ctx.params, model=ctx.model)
    tool_calls: list[dict[str, Any]] = []
    retrieved: list[dict[str, Any]] = []
    text = "I couldn't complete that request within the step limit."
    tokens_out = 0
    for i in range(int(ctx.max_steps or MAX_STEPS)):
        if ctx.params.get("interrupt_after_steps") == i and i > 0:
            events.append({"type": "interrupted", "after_steps": i})
            text = "Interrupted; resume to continue."
            break
        l0 = time.perf_counter()
        action = llm.next_action(turn)
        spans.append({"name": "llm", "latency_s": time.perf_counter() - l0})
        tokens_out += action.pop("_tokens", (0, 0))[1] if "_tokens" in action else 0
        if "answer" in action:
            text = str(action["answer"])
            break
        name, args = str(action.get("tool")), dict(action.get("args") or {})
        step = _run_tool(name, args, ctx, events, retrieved, spans)
        turn.steps.append(step)
        tool_calls.append(
            {
                "name": name,
                "args": args,
                "result": step.result,
                "error": step.error,
                "agent": AGENTS.get(name),
            }
        )
    if ctx.session_id:
        with _memory_lock:
            _memory[ctx.params.get("customer_id", "anon")] = [
                *messages,
                {"role": "assistant", "content": text},
            ][-20:]
    return {
        **base,
        "text": text,
        "blocked": False,
        "tool_calls": tool_calls,
        "retrieved": retrieved,
        "tokens_out": tokens_out or _tokens(text),
        "latency_s": time.perf_counter() - t0,
        "cost_usd": 0.0,
    }


def _trace_id(messages: list[dict[str, str]]) -> str:
    raw = "|".join(m["content"] for m in messages)
    return "tr-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


# --------------------------------------------------------------------------- hooks


def reset(seed: int) -> None:
    tools.reset(seed)
    _local.provider_counter = FaultCounter()
    with _memory_lock:
        _memory.clear()


def state() -> dict[str, Any]:
    return tools.state()


def ingest(docs_dir: str) -> list[dict[str, Any]]:
    return kb.ingest(docs_dir)


def update_doc(doc_id: str, text: str | None) -> None:
    kb.kb().update(doc_id, text)


def stream(case: Any, ctx: Any) -> Iterator[str]:
    """Stream the answer word by word (for TTFT); honours a `stream_drop` fault."""
    out = ask(case.input or case.text_input(), ctx)
    words = str(out["text"]).split(" ")
    drop = find_fault(ctx.faults, provider="llm")
    for i, word in enumerate(words):
        if drop is not None and drop.error == "stream_drop" and i == len(words) // 2:
            yield " [connection lost — sorry, please ask again]"
            return
        yield word + " "
