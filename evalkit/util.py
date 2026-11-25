"""Small shared helpers: errors, hashing, text normalisation, JSON extraction, timing."""

from __future__ import annotations

import contextlib
import difflib
import hashlib
import importlib
import json
import math
import re
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any


class EvalkitError(Exception):
    """Base error; carries the CLI exit code."""

    exit_code = 3


class ConfigError(EvalkitError):
    exit_code = 2


class DataError(EvalkitError):
    exit_code = 2


class MissingDependency(EvalkitError):
    """Raised by lazy imports; metrics report it as `skipped: "<pkg> not installed"`."""

    def __init__(self, package: str) -> None:
        super().__init__(f"{package} not installed")
        self.package = package


# --------------------------------------------------------------------------- imports


def optional_import(module: str, package: str | None = None) -> ModuleType:
    """Import an optional library or raise MissingDependency naming the PyPI package."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise MissingDependency(package or module.split(".")[0]) from exc


def is_installed(module: str) -> bool:
    try:
        importlib.import_module(module)
    except ImportError:
        return False
    return True


def import_string(spec: str) -> Any:
    """Import `package.module:attr` (or `package.module.attr`)."""
    if ":" in spec:
        mod_name, _, attr = spec.partition(":")
    else:
        mod_name, _, attr = spec.rpartition(".")
    try:
        mod = importlib.import_module(mod_name)
    except ImportError as exc:
        raise ConfigError(f"cannot import {spec!r}: {exc}") from exc
    obj = mod
    for part in attr.split("."):
        if not hasattr(obj, part):
            raise ConfigError(f"{mod_name!r} has no attribute {attr!r}")
        obj = getattr(obj, part)
    return obj


# --------------------------------------------------------------------------- hashing


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def stable_hash(obj: Any) -> str:
    return sha256_text(json.dumps(obj, sort_keys=True, default=str, ensure_ascii=False))


# --------------------------------------------------------------------------- text

_SPACES = re.compile(r"\s+")
_TRAILING_PUNCT = re.compile(r"[\s.!?,;:]+$")
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?|-?\.\d+")
_WORD = re.compile(r"\w+", re.UNICODE)


def normalize_text(text: str | None) -> str:
    """strip, lower, collapse spaces, strip trailing punctuation."""
    if text is None:
        return ""
    out = _SPACES.sub(" ", str(text).strip().lower())
    return _TRAILING_PUNCT.sub("", out)


def tokens(text: str | None) -> list[str]:
    return _WORD.findall(normalize_text(text))


def word_count(text: str | None) -> int:
    return len((text or "").split())


def numbers_in(text: str | None) -> list[float]:
    out: list[float] = []
    for m in _NUMBER.findall(text or ""):
        with contextlib.suppress(ValueError):
            out.append(float(m.replace(",", "")))
    return out


def first_number(text: str | None) -> float | None:
    nums = numbers_in(text)
    return nums[0] if nums else None


def jaccard(a: set[Any], b: set[Any]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def ngrams(words: Sequence[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(words[i : i + n]) for i in range(max(0, len(words) - n + 1))}


# --------------------------------------------------------------------------- JSON

_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str | None) -> Any:
    """Return the first JSON object/array in text (fences stripped), or None."""
    if not text:
        return None
    candidates = [m.group(1) for m in _FENCE.finditer(text)] + [text]
    decoder = json.JSONDecoder()
    for cand in candidates:
        for i, ch in enumerate(cand):
            if ch not in "{[":
                continue
            try:
                obj, _ = decoder.raw_decode(cand[i:])
            except json.JSONDecodeError:
                continue
            return obj
    return None


def get_path(obj: Any, dotted: str, default: Any = None) -> Any:
    """Resolve `a.b.0.c` over dicts, lists and attributes."""
    cur = obj
    for part in dotted.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(part, default)
        elif isinstance(cur, list | tuple) and part.lstrip("-").isdigit():
            idx = int(part)
            cur = cur[idx] if -len(cur) <= idx < len(cur) else default
        else:
            cur = getattr(cur, part, default)
    return cur


def deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    """Recursive dict merge; lists and scalars in `over` replace those in `base`."""
    out = dict(base)
    for key, val in over.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def set_path(obj: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cur = obj
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


def did_you_mean(key: str, options: Sequence[str]) -> str:
    close = difflib.get_close_matches(key, list(options), n=1, cutoff=0.6)
    return f" (did you mean {close[0]!r}?)" if close else ""


# --------------------------------------------------------------------------- numbers


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolated percentile, q in [0, 100]."""
    vals = sorted(v for v in values if v is not None and not math.isnan(v))
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


def mean(values: Sequence[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


# --------------------------------------------------------------------------- timing


class Timer:
    """`with Timer() as t: ...; t.elapsed` in seconds (perf_counter)."""

    def __init__(self, clock: Callable[[], float] = time.perf_counter) -> None:
        self._clock = clock
        self.start = 0.0
        self.elapsed = 0.0

    def __enter__(self) -> Timer:
        self.start = self._clock()
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed = self._clock() - self.start


@contextlib.contextmanager
def chdir(path: str | Path) -> Iterator[None]:
    import os

    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Sequence[Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            data = row.model_dump(mode="json", by_alias=True) if hasattr(row, "model_dump") else row
            fh.write(json.dumps(data, ensure_ascii=False, sort_keys=True, default=str) + "\n")
