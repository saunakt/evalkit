"""Detoxify wrapper (extra `[rai]`, lazy import): `Detoxify("original").predict(texts)`."""

from __future__ import annotations

from typing import Any

from evalkit.registry import engine
from evalkit.util import optional_import

_MODEL: list[Any] = []


@engine("detoxify", requires_extra="rai")
class DetoxifyEngine:
    """Engine marker: native implementation of `toxicity_rate`."""

    package = "detoxify"


def toxicity_scores(texts: list[str]) -> list[float]:
    if not texts:
        return []
    detoxify = optional_import("detoxify")
    if not _MODEL:
        _MODEL.append(detoxify.Detoxify("original"))
    scores = _MODEL[0].predict(texts)["toxicity"]
    return [float(s) for s in (scores if isinstance(scores, list) else [scores])]
