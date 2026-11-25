"""Presidio PII detection (extra `[rai]`). The detector itself lives in evalkit.metrics.pii
(Presidio when installed, regex + Luhn/Verhoeff fallback otherwise)."""

from __future__ import annotations

from evalkit.registry import engine


@engine("presidio", requires_extra="rai")
class PresidioEngine:
    """Engine marker: Presidio entities for `pii_leakage_rate` and trace scrubbing."""

    package = "presidio-analyzer"
