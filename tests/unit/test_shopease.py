"""Regression tests for bugs found in the ShopEase demo bot by evalkit's own suites."""

from __future__ import annotations

import pytest

from evalkit.targets.base import TargetContext
from examples.shopease import bot


@pytest.fixture(autouse=True)
def fresh() -> None:
    bot.reset(1)


def test_typo_query_still_finds_returns_policy() -> None:
    # robustness/typo: "dleivery", "reutrn" used to retrieve nothing
    out = bot.ask("How many days after dleivery can I reutrn shoes?", TargetContext())
    assert "30-day" in out["text"]


def test_extra_whitespace_still_counts_as_explicit_confirmation() -> None:
    # robustness/whitespace: "Please   cancel" was not recognised as a confirmation
    out = bot.ask("  Please   cancel order ORD-100001  for me.", TargetContext())
    assert "has been cancelled" in out["text"]


def test_after_is_not_a_retrieval_keyword() -> None:
    # "after" used to pull the refund-timeline snippet above the returns policy
    out = bot.ask("How many days after delivery can I return shoes?", TargetContext())
    assert out["retrieved"][0]["id"] == "returns"


def test_low_score_noise_chunks_are_ignored() -> None:
    # rag/noise_sensitivity: injected unscored chunks used to become the answer
    noise = [
        {
            "id": "noise-0",
            "text": "The cafeteria menu changes every Monday.",
            "metadata": {"access": "public"},
        }
    ]
    out = bot.ask("Do you offer AC installation?", TargetContext(extra_context=noise))
    assert "cafeteria" not in out["text"]


def test_name_is_remembered_case_insensitively() -> None:
    msgs = [
        {"role": "user", "content": "My name is Priya."},
        {"role": "assistant", "content": "Nice to meet you, Priya!"},
        {"role": "user", "content": "What's my name?"},
    ]
    assert bot.ask(msgs, TargetContext())["text"] == "Your name is Priya."
