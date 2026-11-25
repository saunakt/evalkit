"""Tiny input guardrail: regex + keyword list. Fails closed when the guard is down."""

from __future__ import annotations

import base64
import binascii
import re
from typing import Any

from evalkit.env.faults import find_fault

PATTERNS = [
    r"ignore (all |any |the )?(previous|prior|above) (instructions|rules)",
    r"(reveal|print|show|repeat).{0,30}(system prompt|instructions|hidden rules)",
    r"\bjailbreak\b|\bDAN mode\b|developer mode",
    r"\b(bomb|explosive|weapon|poison)\b",
    r"\b(hack|hacking|steal|phish)\b.{0,40}\b(account|password|card|otp)\b",
    r"\b(credit card numbers?|aadhaar numbers?) of (other|all) (customers|users)\b",
    r"pichle (sabhi )?nirdesh (bhool|ignore)",
    r"पिछले (सभी )?निर्देश",
    r"बम|हथियार",
]
_COMPILED = [re.compile(p, re.IGNORECASE) for p in PATTERNS]
_UNLEET = str.maketrans({"4": "a", "3": "e", "1": "i", "0": "o", "5": "s", "7": "t", "@": "a"})


def views(text: str) -> list[str]:
    """The raw text plus normalised views that defeat simple encodings."""
    out = [text, text.translate(_UNLEET)]
    for token in re.findall(r"[A-Za-z0-9+/=]{16,}", text):
        try:
            out.append(base64.b64decode(token, validate=True).decode("utf-8", "ignore"))
        except (binascii.Error, ValueError):
            continue
    return out


def check(text: str, params: dict[str, Any], faults: list[dict[str, Any]]) -> tuple[bool, str]:
    """(blocked, reason). Disabled via params.guardrail_enabled=false."""
    if params.get("guardrail_enabled", True) is False:
        return False, "guardrail disabled"
    fault = find_fault(faults, component="guardrail")
    if fault is not None and fault.error == "down":
        return True, "guardrail unavailable (fail closed)"
    for view in views(text):
        for rx in _COMPILED:
            if rx.search(view):
                return True, f"matched {rx.pattern[:40]}"
    return False, "ok"
