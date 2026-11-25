"""jiwer wrapper (extra `[voice]`, lazy import).

Checked against jiwer 4.0: `jiwer.wer(reference, hypothesis)`.
"""

from __future__ import annotations

from evalkit.registry import engine
from evalkit.util import optional_import


@engine("jiwer", requires_extra="voice")
class JiwerEngine:
    """Engine marker: provides `wer`."""

    package = "jiwer"


def word_error_rate(reference: str, hypothesis: str) -> float:
    jiwer = optional_import("jiwer")
    return float(jiwer.wer(reference, hypothesis))
