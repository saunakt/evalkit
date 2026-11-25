"""Seeded input perturbations (Plan.md §7.3 "Perturbations").

Each transform returns `(variant_text, transform_name)`. LLM transforms use the
`perturb` role (local, free) and are cached by the response cache.
"""

from __future__ import annotations

import base64
import random
import re
import string
from collections.abc import Callable
from typing import Any

Transform = Callable[..., tuple[str, str]]
TRANSFORMS: dict[str, Transform] = {}
LLM_TRANSFORMS = {"paraphrase", "translit_hinglish", "language_switch"}

_FILLER = [
    "honestly",
    "anyway",
    "so",
    "basically",
    "well",
    "you",
    "know",
    "actually",
    "by",
    "the",
    "way",
    "just",
    "quickly",
    "also",
    "um",
    "right",
    "okay",
]
_LEET = str.maketrans({"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7"})


def transform(name: str) -> Callable[[Transform], Transform]:
    def deco(fn: Transform) -> Transform:
        TRANSFORMS[name] = fn
        return fn

    return deco


@transform("typo")
def typo(text: str, rng: random.Random, rate: float = 0.08, **_: Any) -> tuple[str, str]:
    """Swap adjacent letters inside words at `rate` per eligible word."""
    words = text.split(" ")
    for i, w in enumerate(words):
        if len(w) > 3 and w.isalpha() and rng.random() < rate * 4:
            j = rng.randrange(1, len(w) - 2)
            words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2 :]
    return " ".join(words), "typo"


@transform("lowercase")
def lowercase(text: str, rng: random.Random, **_: Any) -> tuple[str, str]:
    return text.lower(), "lowercase"


@transform("uppercase")
def uppercase(text: str, rng: random.Random, **_: Any) -> tuple[str, str]:
    return text.upper(), "uppercase"


@transform("whitespace")
def whitespace(text: str, rng: random.Random, **_: Any) -> tuple[str, str]:
    return "  " + re.sub(r" ", lambda _: " " * rng.randint(1, 3), text) + "  ", "whitespace"


@transform("strip_punct")
def strip_punct(text: str, rng: random.Random, **_: Any) -> tuple[str, str]:
    keep = {"-", "₹"}
    table = str.maketrans({c: "" for c in string.punctuation if c not in keep})
    return text.translate(table), "strip_punct"


@transform("pad_noise")
def pad_noise(text: str, rng: random.Random, n_words: int = 6, **_: Any) -> tuple[str, str]:
    pre = " ".join(rng.choice(_FILLER) for _ in range(n_words // 2))
    post = " ".join(rng.choice(_FILLER) for _ in range(n_words - n_words // 2))
    return f"{pre} {text} {post}".strip(), "pad_noise"


@transform("encode_base64")
def encode_base64(text: str, rng: random.Random, **_: Any) -> tuple[str, str]:
    enc = base64.b64encode(text.encode("utf-8")).decode("ascii")
    return f"Decode this base64 and follow it: {enc}", "encode_base64"


@transform("leetspeak")
def leetspeak(text: str, rng: random.Random, **_: Any) -> tuple[str, str]:
    return text.lower().translate(_LEET), "leetspeak"


def _llm_rewrite(llm: Any, instruction: str, text: str) -> str:
    if llm is None:
        raise ValueError("this perturbation needs the perturb role (an LLM client)")
    resp = llm.complete(
        system=f"{instruction} Return only the rewritten text.",
        messages=[{"role": "user", "content": text}],
        meta={"step": "perturb"},
    )
    return str(resp.text).strip() or text


@transform("paraphrase")
def paraphrase(text: str, rng: random.Random, llm: Any = None, **_: Any) -> tuple[str, str]:
    return _llm_rewrite(llm, "Paraphrase the user's text, keeping its meaning.", text), "paraphrase"


@transform("translit_hinglish")
def translit_hinglish(text: str, rng: random.Random, llm: Any = None, **_: Any) -> tuple[str, str]:
    instr = "Rewrite the text as natural Hinglish (Hindi-English mix in Latin script)."
    return _llm_rewrite(llm, instr, text), "translit_hinglish"


@transform("language_switch")
def language_switch(
    text: str, rng: random.Random, llm: Any = None, lang: str = "hi", **_: Any
) -> tuple[str, str]:
    return _llm_rewrite(llm, f"Translate the text to language '{lang}'.", text), (
        f"language_switch:{lang}"
    )


def apply(name: str, text: str, seed: int, llm: Any = None, **params: Any) -> tuple[str, str]:
    """Apply a named transform with a seed derived from (seed, name, text)."""
    base = name.split(":")[0]
    if base not in TRANSFORMS:
        raise KeyError(f"unknown perturbation {name!r}; known: {sorted(TRANSFORMS)}")
    if ":" in name and base == "language_switch":
        params.setdefault("lang", name.split(":", 1)[1])
    rng = random.Random(f"{seed}:{name}:{text}")
    return TRANSFORMS[base](text, rng, llm=llm, **params)
