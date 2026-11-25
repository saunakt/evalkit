"""classic — unit tests with a fake LLM, record/replay, contracts, API checks, UI tests (ch. 15)."""

from __future__ import annotations

from evalkit.registry import suite
from evalkit.suites.base import BaseSuite


@suite("classic")
class ClassicSuite(BaseSuite):
    """Unit tests with a fake LLM, record/replay, contracts, API checks, UI tests (ch. 15)."""

    name = "classic"
    default_metrics = ("unit_fake_llm", "record_replay", "contract", "api", "ui_streaming")
