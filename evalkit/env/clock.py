"""Frozen clock: `run.now` → TargetContext.now + EVALKIT_NOW (+ freezegun on request)."""

from __future__ import annotations

import contextlib
import os
import threading
from collections.abc import Iterator
from datetime import datetime

_freeze_lock = threading.Lock()


def parse_now(now: str | None) -> datetime | None:
    return datetime.fromisoformat(now) if now else None


@contextlib.contextmanager
def exported_now(now: str | None) -> Iterator[None]:
    """Export EVALKIT_NOW for the duration of a run."""
    old = os.environ.get("EVALKIT_NOW")
    if now:
        os.environ["EVALKIT_NOW"] = now
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("EVALKIT_NOW", None)
        else:
            os.environ["EVALKIT_NOW"] = old


@contextlib.contextmanager
def frozen(now: str | None) -> Iterator[None]:
    """Run a block under freezegun.freeze_time(now).

    freezegun patches process-global state, so frozen blocks are serialised with a
    lock (see docs/DECISIONS.md). No-op when `now` is empty.
    """
    if not now:
        yield
        return
    from freezegun import freeze_time

    with _freeze_lock, freeze_time(now):
        yield
