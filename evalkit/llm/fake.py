"""Deterministic fake provider for tests and the offline demo (Plan.md §7.2 "fake")."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

from evalkit.config import ProviderConfig
from evalkit.llm.client import LLMResponse, ResponseFormat, estimate_tokens
from evalkit.registry import provider

_SENT = re.compile(r"(?<=[.!?])\s+")


class FakeLLM:
    """Resolves responses from script rules, judge_script entries, or defaults.

    Script YAML::

        rules: [{match: <regex>, role: <optional role>, response: <str|json>}]
        judge_script: [{case: <id>, judge: <id>, match: <regex>, verdict: PASS}]
        default_judge: PASS

    Defaults: judges PASS (in the JSON shape their style expects), simulator `<DONE>`,
    everything else echoes the last user message.
    """

    def __init__(self, name: str = "fake", script: dict[str, Any] | None = None) -> None:
        self.name = name
        script = script or {}
        self.rules: list[dict[str, Any]] = list(script.get("rules") or [])
        self.judge_script: list[dict[str, Any]] = list(script.get("judge_script") or [])
        self.default_judge = str(script.get("default_judge", "PASS"))
        self.calls: list[dict[str, Any]] = []

    @classmethod
    def from_file(cls, name: str, path: Path | None) -> FakeLLM:
        if path is None or not path.exists():
            return cls(name)
        return cls(name, yaml.safe_load(path.read_text(encoding="utf-8")) or {})

    def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        response_format: ResponseFormat = "text",
        meta: dict[str, Any] | None = None,
    ) -> LLMResponse:
        meta = meta or {}
        prompt = system + "\n" + "\n".join(str(m.get("content", "")) for m in messages)
        self.calls.append({"meta": meta, "prompt": prompt, "model": model})
        role = meta.get("role")
        text = self._judge(prompt, meta) if role == "judge" else self._rule(prompt, role)
        if text is None:
            text = self._default(messages, role)
        return LLMResponse(
            text=text,
            tokens_in=estimate_tokens(prompt),
            tokens_out=estimate_tokens(text),
            tokens_estimated=True,
            model=model,
            provider=self.name,
        )

    def stream(self, *, system: str, messages: list[dict[str, Any]], model: str, **_: Any) -> Any:
        resp = self.complete(system=system, messages=messages, model=model)
        yield from (w + " " for w in resp.text.split())

    # ----------------------------------------------------------------- resolution

    def _rule(self, prompt: str, role: str | None) -> str | None:
        for rule in self.rules:
            if rule.get("role") and rule["role"] != role:
                continue
            if re.search(str(rule.get("match", "")), prompt, re.IGNORECASE | re.DOTALL):
                return _as_text(rule.get("response", ""))
        return None

    def _judge(self, prompt: str, meta: dict[str, Any]) -> str | None:
        for entry in self.judge_script:
            if entry.get("case") and entry["case"] != meta.get("case_id"):
                continue
            if entry.get("judge") and entry["judge"] != meta.get("judge_id"):
                continue
            if entry.get("step") and entry["step"] != meta.get("step"):
                continue
            if entry.get("match") and not re.search(entry["match"], prompt, re.I | re.S):
                continue
            if "response" in entry:
                return _as_text(entry["response"])
            return judge_payload(meta, str(entry.get("verdict", self.default_judge)), entry)
        return judge_payload(meta, self.default_judge, {})

    def _default(self, messages: list[dict[str, Any]], role: str | None) -> str:
        if role == "simulator":
            return "<DONE>"
        users = [m for m in messages if m.get("role") == "user"]
        return str(users[-1].get("content", "")) if users else ""


def judge_payload(meta: dict[str, Any], verdict: str, entry: dict[str, Any]) -> str:
    """Build the JSON a judge of `meta['style']` / `meta['step']` would return."""
    style, step = meta.get("style", "binary"), meta.get("step")
    ok = verdict.upper() == "PASS"
    if step == "steps":
        out: dict[str, Any] = {"steps": ["Read the criterion.", "Check the output against it."]}
    elif step == "extract":
        text = str(meta.get("text", ""))
        out = {"claims": entry.get("claims") or [s for s in _SENT.split(text) if s.strip()]}
    elif step == "verify":
        n = int(meta.get("n_claims", 1))
        out = {"supported": entry.get("supported") or [ok] * n}
    elif style in ("rubric", "geval"):
        out = {"reasoning": "scripted", "score": int(entry.get("score", 5 if ok else 1))}
    elif style == "pairwise":
        out = {"reasoning": "scripted", "winner": entry.get("winner", "tie")}
    else:
        out = {"reasoning": "scripted", "verdict": "PASS" if ok else "FAIL"}
    return json.dumps(out)


def _as_text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value)


@provider("fake")
def make_fake(name: str, cfg: ProviderConfig, root: Path) -> FakeLLM:
    path = (root / cfg.script) if cfg.script else None
    return FakeLLM.from_file(name, path)
