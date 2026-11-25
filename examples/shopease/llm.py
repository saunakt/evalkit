"""The ShopEase "brain": a deterministic FakeLLM (offline) and a RealLLM (any provider).

Both speak the same JSON action format, so the agent loop in bot.py is identical:
`{"tool": name, "args": {...}}` to call a tool, `{"answer": text}` to finish.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evalkit.util import extract_json
from examples.shopease.i18n import detect_language, localise

PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "shopease_system_v1.txt"
ORDER_ID = re.compile(r"\bORD-\d{6}\b", re.IGNORECASE)
INJECTION = re.compile(
    r"ignore (all |any |the )?(previous|prior|above) (instructions|rules)|system prompt|"
    r"hidden (rules|instructions)|developer mode|\bDAN\b|you are now|reveal your",
    re.IGNORECASE,
)
HARMFUL = re.compile(r"\b(bomb|explosive|weapon|poison|steal|phish|hack into)\b", re.IGNORECASE)
TOO_BIG = re.compile(r"\b(repeat|write|list)\b.{0,40}\b\d{4,}\b", re.IGNORECASE)
AI_QUESTION = re.compile(
    r"\b(are you (a )?(human|real person|bot|robot|ai)|pretend (to be|you are) "
    r"(a )?human)",
    re.IGNORECASE,
)
NAME_STATE = re.compile(r"\b(?i:my name is) ([A-Z][a-z]+)")
NAME_ASK = re.compile(r"what('?s| is) my name", re.IGNORECASE)
CONFIRM = re.compile(r"\b(yes|confirm|go ahead|please cancel|kindly cancel|cancel it now)\b", re.I)
ASK_CONFIRM = "Reply yes to confirm."
CANCEL = re.compile(r"\bcancel|रद्द|cancel kar", re.IGNORECASE)
REFERENCE = re.compile(r"\b(it|that order|the order|same order|that one)\b", re.IGNORECASE)
DATA_RIGHTS = re.compile(
    r"\b(delete|erase|access|correct|update)\b.{0,30}\b(data|"
    r"information|details)\b",
    re.IGNORECASE,
)
OUT_OF_SCOPE = re.compile(
    r"\b(weather|joke|poem|cricket|politic\w*|election|recipe|python code|write code|stock|"
    r"movie|capital of|homework|horoscope)\b",
    re.IGNORECASE,
)
POLICY = re.compile(
    r"\b(return\w*|refund\w*|ship\w*|deliver\w*|cod|cash on delivery|warrant\w*|exchang\w*|"
    r"gift ?cards?|defect\w*|damaged|privacy|contact|support hours|coupon|voucher|wapas|"
    r"paisa|policy|installation|price match|goodwill|partner|megamart|install)\b"
    r"|रिटर्न|वापस|शिपिंग|டெலிவரி|திருப்பி|ফেরত|শিপিং",
    re.IGNORECASE,
)
JUST_NUMBER = re.compile(r"\b(just|only) the number\b", re.IGNORECASE)
SANITISE = re.compile(r"ignore|instruction|system:|you must|assistant:|canary", re.IGNORECASE)
REFUSAL = (
    "Sorry, I can't help with that. I can help with ShopEase orders, returns, refunds, "
    "shipping and payments."
)
NOT_FOUND = (
    "I couldn't find that in ShopEase's policies, so I don't want to guess. "
    "I can connect you with a human agent if you like."
)


def system_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


@dataclass
class Step:
    tool: str
    args: dict[str, Any]
    result: Any = None
    error: str | None = None


@dataclass
class Turn:
    """What the model sees: transcript, tool steps so far, request params."""

    messages: list[dict[str, str]]
    steps: list[Step] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    model: str | None = None

    @property
    def user(self) -> str:
        users = [m["content"] for m in self.messages if m.get("role") == "user"]
        return users[-1] if users else ""

    def history_text(self) -> str:
        return "\n".join(m["content"] for m in self.messages[:-1])

    def step(self, tool: str) -> Step | None:
        return next((s for s in self.steps if s.tool == tool), None)


def _decoded(text: str) -> str:
    """Decode an embedded base64 payload (to see through encoded attacks)."""
    for token in re.findall(r"[A-Za-z0-9+/=]{16,}", text):
        try:
            return text + " " + base64.b64decode(token, validate=True).decode("utf-8", "ignore")
        except (binascii.Error, ValueError):
            continue
    return text


class FakeLLM:
    """Deterministic, regex-driven policy that behaves like a well-instructed model."""

    def next_action(self, turn: Turn) -> dict[str, Any]:
        text = re.sub(r"\s+", " ", _decoded(turn.user)).strip()
        lang = detect_language(text)
        leak = turn.params.get("fake_leak") or os.environ.get("EVALKIT_DEMO_LEAK") == "1"
        if INJECTION.search(text):
            return {"answer": system_prompt()[:400] if leak else _loc("refusal", lang, REFUSAL)}
        if HARMFUL.search(text) or TOO_BIG.search(text):
            return {"answer": _loc("refusal", lang, REFUSAL)}
        if AI_QUESTION.search(text):
            return {
                "answer": "I'm ShopEase's AI assistant, not a human. I can connect you with "
                "a human agent at any time. [ai-disclosure]"
            }
        name = NAME_STATE.search(text)
        if name:
            return {"answer": f"Nice to meet you, {name.group(1)}! How can I help with your order?"}
        if NAME_ASK.search(text):
            found = NAME_STATE.findall(turn.history_text())
            return {
                "answer": f"Your name is {found[-1]}."
                if found
                else "You haven't told me your name yet."
            }
        oid = self._order_id(turn, text)
        if oid:
            return self._order_flow(turn, text, oid, lang)
        if DATA_RIGHTS.search(text) or POLICY.search(text):
            return self._policy_flow(turn, text, lang)
        if OUT_OF_SCOPE.search(text):
            return {
                "answer": "That's outside what I can help with. I can help with ShopEase "
                "orders, returns, refunds, shipping and payments."
            }
        if re.search(r"^\s*(hi|hello|hey|namaste|thanks|thank you)\b", text, re.IGNORECASE):
            return {
                "answer": "Hello! I can help with orders, returns, refunds, shipping and "
                "payments. What do you need?"
            }
        return self._policy_flow(turn, text, lang)  # unrecognised: try the knowledge base

    @staticmethod
    def _confirming(turn: Turn, text: str) -> bool:
        """The user is saying yes to our earlier 'Reply yes to confirm' question."""
        last = [m["content"] for m in turn.messages[:-1] if m.get("role") == "assistant"]
        return bool(last) and ASK_CONFIRM in last[-1] and bool(CONFIRM.search(text))

    def _order_id(self, turn: Turn, text: str) -> str | None:
        m = ORDER_ID.search(text)
        if m:
            return m.group(0).upper()
        if REFERENCE.search(text) or CANCEL.search(text) or self._confirming(turn, text):
            earlier = ORDER_ID.findall(turn.history_text())
            return earlier[-1].upper() if earlier else None
        return None

    def _order_flow(self, turn: Turn, text: str, oid: str, lang: str) -> dict[str, Any]:
        got = turn.step("get_order")
        if got is None:
            return {"tool": "get_order", "args": {"order_id": oid}}
        if got.error:
            return {
                "answer": "I'm having trouble looking up your order right now, so I can't "
                "confirm anything yet. Please try again in a few minutes."
            }
        status = str(got.result.get("status"))
        confirmed = bool(CONFIRM.search(text)) or self._confirming(turn, text)
        if CANCEL.search(text) and not confirmed:
            return {
                "answer": f"Order {oid} is {status}. Do you want me to cancel it? {ASK_CONFIRM}"
            }
        if CANCEL.search(text) or self._confirming(turn, text):
            cancel = turn.step("cancel_order")
            if cancel is None:
                return {
                    "tool": "cancel_order",
                    "args": {"order_id": oid, "reason": "customer request"},
                }
            if cancel.error:
                if "denied" in cancel.error:
                    return {
                        "answer": f"The cancellation of {oid} was not approved, so the "
                        "order is unchanged."
                    }
                if "unavailable" in cancel.error:
                    return {
                        "answer": f"I couldn't complete the cancellation of {oid} right "
                        "now, so the order is unchanged. Please try again shortly."
                    }
                return {
                    "answer": f"Order {oid} is {status} and can't be cancelled. Only orders "
                    "still processing can be cancelled; you can return it within 30 days "
                    "of delivery instead. [returns]"
                }
            return {
                "answer": _loc(
                    "cancelled",
                    lang,
                    f"Your order {oid} has been cancelled. "
                    "The refund reaches your original payment method in 5-7 "
                    "business days. [refund-timeline]",
                    oid=oid,
                )
            }
        if re.search(r"\bjson\b", text, re.IGNORECASE):
            return {
                "answer": json.dumps(
                    {"order_id": oid, "status": status, "amount": got.result.get("amount")}
                )
            }
        amount = got.result.get("amount")
        return {
            "answer": _loc(
                "status",
                lang,
                f"Order {oid} ({got.result.get('item')}, ₹{amount:,}) is {status}.",
                oid=oid,
                status=status,
                amount=f"₹{amount:,}",
            )
        }

    def _policy_flow(self, turn: Turn, text: str, lang: str) -> dict[str, Any]:
        search = turn.step("search_policy")
        if search is None:
            return {"tool": "search_policy", "args": {"query": text}}
        chunks = [
            c
            for c in (search.result or [])
            if not SANITISE.search(c.get("text", ""))
            and float(c.get("score") or 0) >= 0.5
            and str(c.get("metadata", {}).get("source", "")).startswith("docs/")
        ]
        if search.error or not chunks:
            return {"answer": _loc("not_found", lang, NOT_FOUND)}
        top = chunks[0]
        sentences = re.split(r"(?<=[.!?])\s+", top["text"].strip())
        english = " ".join(sentences[:2]) + f" [{top['id']}]"
        if JUST_NUMBER.search(text):
            number = re.search(r"\d[\d,]*", sentences[0])
            return {"answer": number.group(0) if number else english}
        return {"answer": _loc(top["id"], lang, english) or english}


def _loc(key: str, lang: str, english: str, **kw: str) -> str:
    if lang == "en":
        return english.format(**kw) if kw else english
    return localise(key, lang, **kw) or english


class RealLLM:
    """Calls a real model through evalkit's `sut` role (any configured provider)."""

    def __init__(self) -> None:
        from evalkit.llm.roles import client_for

        self.client, self.model = client_for("sut")

    def next_action(self, turn: Turn) -> dict[str, Any]:
        msgs = [dict(m) for m in turn.messages]
        for s in turn.steps:
            payload = {"tool": s.tool, "result": s.result, "error": s.error}
            msgs.append(
                {"role": "user", "content": "TOOL RESULT " + json.dumps(payload, default=str)}
            )
        resp = self.client.complete(
            system=system_prompt(),
            messages=msgs,
            model=turn.model or self.model,
            response_format="json",
        )
        action = extract_json(resp.text)
        if isinstance(action, dict) and ("tool" in action or "answer" in action):
            action["_tokens"] = (resp.tokens_in, resp.tokens_out)
            return action
        return {"answer": resp.text.strip(), "_tokens": (resp.tokens_in, resp.tokens_out)}


def make_llm(params: dict[str, Any]) -> FakeLLM | RealLLM:
    return RealLLM() if params.get("llm", os.environ.get("SHOPEASE_LLM")) == "real" else FakeLLM()
