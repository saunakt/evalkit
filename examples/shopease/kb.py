"""ShopEase knowledge base: markdown policy snippets + a small BM25 retriever."""

from __future__ import annotations

import math
import os
import re
import threading
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DOCS_DIR = Path(__file__).parent / "docs"
STOPWORDS = set(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "could",
        "do",
        "does",
        "for",
        "from",
        "have",
        "how",
        "i",
        "if",
        "in",
        "is",
        "it",
        "its",
        "may",
        "me",
        "my",
        "of",
        "on",
        "or",
        "our",
        "please",
        "should",
        "so",
        "that",
        "the",
        "this",
        "to",
        "was",
        "what",
        "when",
        "where",
        "which",
        "will",
        "with",
        "would",
        "you",
        "your",
        "there",
        "their",
        "they",
        "them",
        "get",
        "got",
        "any",
        "long",
        "take",
        "takes",
        "many",
        "much",
        "want",
        "need",
        "tell",
        "know",
        "about",
        "after",
        "before",
        "kya",
        "hai",
        "hain",
        "mera",
        "meri",
        "mujhe",
        "karna",
        "kaise",
        "kab",
        "kitne",
        "ka",
        "ki",
        "ke",
    ]
)
# Split on whitespace/punctuation only: Indic vowel signs are not `\w`.
_TOKEN = re.compile(r"[^\s\-.,!?;:()\[\]\"'।|/]+")
_SUFFIXES = ("ing", "ed", "es", "s", "y")
SYNONYMS = {
    "wapas": "return",
    "vapas": "return",
    "वापस": "return",
    "रिटर्न": "return",
    "திருப்பி": "return",
    "ফেরত": "return",
    "paisa": "refund",
    "रिफ़ंड": "refund",
    "शिपिंग": "shipping",
    "ஷிப்பிங்": "shipping",
    "শিপিং": "shipping",
    "डिलीवरी": "delivery",
    "டெலிவரி": "delivery",
    "ডেলিভারি": "delivery",
    "din": "days",
    "दिन": "days",
    "நாட்கள்": "days",
    "দিন": "days",
}


def buggy() -> bool:
    return os.environ.get("EVALKIT_DEMO_BUGGY") == "1"


def stem(word: str) -> str:
    """Light suffix stripping ("refunds" / "refunded" → "refund")."""
    for suf in _SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            return word[: -len(suf)]
    return word


def tokenize(text: str) -> list[str]:
    words = [SYNONYMS.get(w, w) for w in _TOKEN.findall(text.lower())]
    words = [w for w in words if w not in STOPWORDS]
    return words if buggy() else [stem(w) for w in words]


@dataclass
class Doc:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def parse_doc(path: Path) -> Doc:
    raw = path.read_text(encoding="utf-8")
    meta: dict[str, Any] = {}
    body = raw
    if raw.startswith("---"):
        _, front, body = raw.split("---", 2)
        for line in front.strip().splitlines():
            key, _, val = line.partition(":")
            meta[key.strip()] = val.strip()
    doc_id = str(meta.pop("id", path.stem))
    meta["source"] = f"docs/{path.name}"
    return Doc(doc_id, body.strip(), meta)


class KnowledgeBase:
    """BM25 over whole-snippet chunks (k1=1.2, b=0.75)."""

    def __init__(self, docs: list[Doc]) -> None:
        self.docs = {d.id: d for d in docs}
        self._index()

    def _index(self) -> None:
        self.tf = {d.id: Counter(tokenize(d.text)) for d in self.docs.values()}
        n = len(self.tf) or 1
        df: Counter[str] = Counter()
        for counts in self.tf.values():
            df.update(counts.keys())
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.avg_len = sum(sum(c.values()) for c in self.tf.values()) / n

    def correct(self, token: str) -> str:
        """Fix swapped-letter typos: map an unknown token to a known anagram of it."""
        if token in self.idf or len(token) < 4 or buggy():
            return token
        key = "".join(sorted(token))
        for known in sorted(self.idf):
            if len(known) == len(token) and "".join(sorted(known)) == key:
                return known
        return token

    def search(self, query: str, k: int = 5, min_score: float = 0.5) -> list[tuple[Doc, float]]:
        q = [self.correct(t) for t in tokenize(query)]
        scored = []
        for doc_id, counts in self.tf.items():
            dl = sum(counts.values()) or 1
            score = 0.0
            for t in q:
                f = counts.get(t, 0)
                if f:
                    score += self.idf[t] * f * 2.2 / (f + 1.2 * (0.25 + 0.75 * dl / self.avg_len))
            if score >= min_score:
                scored.append((self.docs[doc_id], round(score, 4)))
        scored.sort(key=lambda x: (-x[1], x[0].id))
        return scored[:k]

    def update(self, doc_id: str, text: str | None) -> None:
        if text is None:
            self.docs.pop(doc_id, None)
        elif doc_id in self.docs:
            self.docs[doc_id].text = text
        else:
            self.docs[doc_id] = Doc(
                doc_id,
                text,
                {"source": f"docs/{doc_id}.md", "access": "public", "tenant": "shopease"},
            )
        self._index()


def load_docs(docs_dir: Path = DOCS_DIR) -> list[Doc]:
    return [parse_doc(p) for p in sorted(Path(docs_dir).glob("*.md"))]


_local = threading.local()


def kb() -> KnowledgeBase:
    """Per-thread knowledge base (reset() rebuilds it)."""
    if not hasattr(_local, "kb"):
        _local.kb = KnowledgeBase(load_docs())
    value: KnowledgeBase = _local.kb
    return value


def reset_kb() -> None:
    _local.kb = KnowledgeBase(load_docs())


def chunks_for(docs: list[tuple[Doc, float]]) -> list[dict[str, Any]]:
    return [{"id": d.id, "text": d.text, "score": s, "metadata": dict(d.metadata)} for d, s in docs]


def ingest(docs_dir: str) -> list[dict[str, Any]]:
    """Ingestion hook: parse every doc into chunks with metadata."""
    path = Path(docs_dir)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    return [{"id": d.id, "text": d.text, "metadata": dict(d.metadata)} for d in load_docs(path)]
