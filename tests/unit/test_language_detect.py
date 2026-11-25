"""Regression: concurrent first calls to langdetect corrupted its global profiles, so the
nightly multilingual run labelled a Hindi reply `bn` / None and language_match failed."""

from __future__ import annotations

import subprocess
import sys

import pytest

from tests.conftest import ROOT

SCRIPT = """
from concurrent.futures import ThreadPoolExecutor
from evalkit.metrics.language import detect_language
t = "माफ़ कीजिए, मैं इसमें मदद नहीं कर सकता। मैं ऑर्डर, रिटर्न और शिपिंग में मदद कर सकता हूँ।"
with ThreadPoolExecutor(8) as ex:
    print(",".join(sorted(set(str(r) for r in ex.map(detect_language, [t] * 16)))))
"""


def test_detect_language_is_thread_safe_on_first_use() -> None:
    pytest.importorskip("langdetect")
    out = subprocess.run(
        [sys.executable, "-c", SCRIPT],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
        check=True,
    )
    assert out.stdout.strip() == "hi"


def test_detect_language_empty_text_is_none() -> None:
    pytest.importorskip("langdetect")
    from evalkit.metrics.language import detect_language

    assert detect_language("") is None and detect_language("123 !!") is None
