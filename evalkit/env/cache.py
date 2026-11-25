"""SQLite response cache with record/replay modes (Plan.md §7.2 "Cache")."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Literal

from evalkit.util import EvalkitError, stable_hash

CacheMode = Literal["off", "read_write", "replay_only"]


class CacheMiss(EvalkitError):
    """Raised in replay_only mode when a call is not in the cache."""

    exit_code = 3


class ResponseCache:
    """key = sha256(provider, model, system, messages, temperature, max_tokens)."""

    def __init__(self, path: str | Path, mode: CacheMode = "read_write") -> None:
        self.path = Path(path)
        self.mode = mode
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, value TEXT)")
        self._db.commit()

    @staticmethod
    def key(
        provider: str,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        return stable_hash([provider, model, system, messages, temperature, max_tokens])

    def get(self, key: str) -> dict[str, Any] | None:
        if self.mode == "off":
            return None
        with self._lock:
            row = self._db.execute("SELECT value FROM responses WHERE key = ?", (key,)).fetchone()
        if row is None:
            self.misses += 1
            if self.mode == "replay_only":
                raise CacheMiss(f"cache miss in replay_only mode (key {key[:12]})")
            return None
        self.hits += 1
        value: dict[str, Any] = json.loads(row[0])
        return value

    def put(self, key: str, value: dict[str, Any]) -> None:
        if self.mode != "read_write":
            return
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO responses (key, value) VALUES (?, ?)",
                (key, json.dumps(value, sort_keys=True, default=str)),
            )
            self._db.commit()

    def close(self) -> None:
        with self._lock:
            self._db.close()
