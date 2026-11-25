"""Deterministic case checks (Plan.md §9.1) — reused by many suites."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import jsonschema

from evalkit.metrics.common import boolean, contains_all, contains_any, na, result, scored
from evalkit.registry import metric
from evalkit.schema import Case, MetricResult, Observation
from evalkit.suites.context import MetricContext
from evalkit.util import extract_json, first_number, get_path, normalize_text, word_count

DET_SUITES = (
    "quality",
    "prompt",
    "robustness",
    "conversation",
    "rag",
    "agent",
    "skills",
    "security",
    "responsible_ai",
    "guardrails",
    "compliance",
    "performance",
    "migration",
    "multimodal",
    "finetune",
    "multilingual",
)
_VOWELS = re.compile(r"[aeiouy]+")


@metric("exact_match", suites=DET_SUITES, kind="deterministic", needs=("reference|label",))
def exact_match(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """Normalised output equals the reference (or label)."""
    ref = case.reference if case.reference is not None else case.label
    return boolean(normalize_text(obs.text) == normalize_text(ref), expected=ref)


@metric("contains", suites=DET_SUITES, kind="deterministic", needs=("must_include",))
def contains(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """All must_include strings appear (case-insensitive); details list missing ones."""
    missing = contains_all(obs.text, case.must_include)
    n = len(case.must_include)
    return result((n - len(missing)) / n, not missing, missing=missing)


@metric("not_contains", suites=DET_SUITES, kind="deterministic", needs=("must_not_include",))
def not_contains(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """None of must_not_include appear."""
    present = contains_any(obs.text, case.must_not_include)
    return boolean(not present, present=present)


@metric("regex", suites=DET_SUITES, kind="deterministic", needs=("regex",))
def regex(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """`re.search(case.regex, output)` matches."""
    return boolean(
        bool(re.search(str(case.regex), obs.text, re.IGNORECASE | re.S)), pattern=case.regex
    )


def load_schema(spec: dict[str, Any] | str, root: Path) -> dict[str, Any]:
    if isinstance(spec, dict):
        return spec
    path = Path(spec)
    data: dict[str, Any] = json.loads(
        (path if path.is_absolute() else root / path).read_text(encoding="utf-8")
    )
    return data


@metric("json_schema", suites=DET_SUITES, kind="deterministic", needs=("schema_",))
def json_schema(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """Extract JSON from the output (fences stripped) and validate against the schema."""
    data = extract_json(obs.text)
    if data is None:
        return boolean(False, reason="no JSON found in output")
    schema = load_schema(case.schema_ or {}, ctx.suite.project.root)
    try:
        jsonschema.validate(data, schema)
    except jsonschema.ValidationError as exc:
        return boolean(False, reason=exc.message, path=list(exc.path))
    return boolean(True)


@metric("numeric_tolerance", suites=DET_SUITES, kind="deterministic", needs=("reference",))
def numeric_tolerance(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """First number in output (or extra.number_path) within params.tolerance of reference."""
    ref = first_number(case.reference)
    if ref is None:
        return na("reference is not a number")
    path = case.extra.get("number_path")
    if path:
        val = get_path(extract_json(obs.text), path)
        got = float(val) if isinstance(val, int | float) else first_number(str(val))
    else:
        got = first_number(obs.text)
    tol = float(ctx.p("tolerance", 0.01))
    if got is None:
        return boolean(False, reason="no number in output", expected=ref)
    return result(
        1.0 if abs(got - ref) <= tol else 0.0, abs(got - ref) <= tol, tol, got=got, expected=ref
    )


@metric("length", suites=DET_SUITES, kind="deterministic")
def length(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """Word count ≤ params.max_words (or extra.max_words)."""
    limit = case.extra.get("max_words", ctx.p("max_words"))
    if limit is None:
        return na("needs params.max_words or extra.max_words")
    n = word_count(obs.text)
    return result(float(n), n <= int(limit), float(limit), words=n)


def syllables(word: str) -> int:
    word = word.lower()
    n = len(_VOWELS.findall(word))
    if word.endswith("e") and n > 1:
        n -= 1
    return max(1, n)


def flesch(text: str) -> float:
    sentences = max(1, len(re.findall(r"[.!?]+", text)) or 1)
    words = re.findall(r"[A-Za-z]+", text)
    if not words:
        return 0.0
    syl = sum(syllables(w) for w in words)
    return 206.835 - 1.015 * len(words) / sentences - 84.6 * syl / len(words)


@metric("readability", suites=("quality",), kind="deterministic", unit="score")
def readability(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """Flesch reading ease (textstat if installed, else a built-in syllable heuristic)."""
    try:
        import textstat

        score, engine = float(textstat.flesch_reading_ease(obs.text)), "textstat"
    except ImportError:
        score, engine = flesch(obs.text), "builtin"
    minimum = ctx.p("min_readability")
    if minimum is None:
        return result(score, None, implementation=engine)
    return scored(score, float(minimum), implementation=engine)
