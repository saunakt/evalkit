"""PII detection: Presidio when installed, else a regex fallback with Luhn/Verhoeff checks."""

from __future__ import annotations

import re
from typing import Any

from evalkit.util import is_installed

EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
PHONE_IN = re.compile(r"(?<!\d)(?:\+91[\s-]?|0)?[6-9]\d{4}[\s-]?\d{5}(?!\d)")
PAN = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")
AADHAAR = re.compile(r"(?<!\d)[2-9]\d{3}[\s-]?\d{4}[\s-]?\d{4}(?!\d)")
CARD = re.compile(r"(?<!\d)(?:\d[\s-]?){13,19}(?!\d)")

_VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]


def verhoeff_valid(number: str) -> bool:
    c = 0
    for i, ch in enumerate(reversed(re.sub(r"\D", "", number))):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][int(ch)]]
    return c == 0


def luhn_valid(number: str) -> bool:
    digits = [int(d) for d in re.sub(r"\D", "", number)]
    if len(digits) < 13:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def regex_entities(text: str) -> list[dict[str, str]]:
    found: list[dict[str, str]] = []
    found += [{"type": "EMAIL", "value": m} for m in EMAIL.findall(text)]
    found += [{"type": "IN_PAN", "value": m} for m in PAN.findall(text)]
    for m in AADHAAR.findall(text):
        if verhoeff_valid(m):
            found.append({"type": "IN_AADHAAR", "value": m})
    for m in CARD.findall(text):
        if luhn_valid(m):
            found.append({"type": "CREDIT_CARD", "value": m.strip()})
    aadhaar_digits = {re.sub(r"\D", "", f["value"]) for f in found}
    for m in PHONE_IN.findall(text):
        if re.sub(r"\D", "", m)[-10:] not in {d[-10:] for d in aadhaar_digits}:
            found.append({"type": "PHONE_NUMBER", "value": m})
    return found


_ANALYZER: list[Any] = []


def detect(text: str) -> tuple[list[dict[str, str]], str]:
    """(entities, engine). Presidio is used when installed (lazy), regex otherwise."""
    if is_installed("presidio_analyzer"):
        try:
            if not _ANALYZER:
                from presidio_analyzer import AnalyzerEngine

                _ANALYZER.append(AnalyzerEngine())
            res = _ANALYZER[0].analyze(text=text, language="en")
            ents = [
                {"type": r.entity_type, "value": text[r.start : r.end]}
                for r in res
                if r.score >= 0.5
            ]
            return ents + [
                e for e in regex_entities(text) if e["type"].startswith("IN_")
            ], "presidio"
        except Exception:  # model download or init failure → fall back to regex
            pass
    return regex_entities(text), "regex"
