"""Sample-size advice and slice tables (Plan.md §10)."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import Any

from evalkit.stats.intervals import wilson


def cases_needed(p: float, margin: float, z: float = 1.96) -> int:
    """Cases needed for ±margin at observed pass rate p: ceil(z² p(1−p) / margin²)."""
    if margin <= 0:
        raise ValueError("margin must be > 0")
    return math.ceil(z * z * p * (1 - p) / (margin * margin))


def slice_table(
    rows: Iterable[Any], by: str | Callable[[Any], Iterable[str]]
) -> list[dict[str, Any]]:
    """Pass rate + Wilson CI + n per slice. Rows need `.passed`; `by` is an attribute
    name (list attributes such as `tags` fan out) or a function returning slice keys."""
    groups: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        if getattr(row, "passed", None) is None or getattr(row, "skipped", None):
            continue
        if callable(by):
            keys = list(by(row))
        else:
            val = getattr(row, by, None)
            keys = [str(v) for v in val] if isinstance(val, list) else [str(val)]
        for key in keys or ["(none)"]:
            groups[key].append(bool(row.passed))
    out = []
    for key in sorted(groups):
        vals = groups[key]
        lo, hi = wilson(sum(vals), len(vals))
        out.append(
            {"slice": key, "n": len(vals), "pass_rate": sum(vals) / len(vals), "lo": lo, "hi": hi}
        )
    return out
