"""Judge prompt templates (Plan.md §8.2)."""

from __future__ import annotations

import json
from typing import Any

from evalkit.config import JudgeConfig

SYSTEM = (
    "You are a careful, strict evaluator. Judge ONLY the single criterion given. "
    "Length is not a quality signal: do not prefer longer or more detailed answers "
    "unless the criterion asks for it. Think step by step in `reasoning` first, then "
    "give the verdict. Respond with a single JSON object and nothing else."
)

FORMATS = {
    "binary": '{"reasoning": "<why>", "verdict": "PASS" | "FAIL"}',
    "reference": '{"reasoning": "<why>", "verdict": "PASS" | "FAIL"}',
    "rubric": '{"reasoning": "<why>", "score": <integer 1-5>}',
    "geval": '{"reasoning": "<why>", "score": <integer 1-5>}',
    "pairwise": '{"reasoning": "<why>", "winner": "A" | "B" | "tie"}',
    "extract": '{"claims": ["<atomic claim>", ...]}',
    "verify": '{"supported": [true | false, ...]}  (one entry per claim, in order)',
    "steps": '{"steps": ["<evaluation step>", ...]}',
}

REMINDER = "Return ONLY the JSON object in the required format, with no other text."


def _section(title: str, body: Any) -> str:
    if body is None or body == "" or body == []:
        return ""
    if not isinstance(body, str):
        body = json.dumps(body, ensure_ascii=False, indent=1, default=str)
    return f"## {title}\n{body}\n\n"


def few_shot_block(judge: JudgeConfig) -> str:
    if not judge.few_shot:
        return ""
    lines = ["## Labelled examples"]
    for ex in judge.few_shot:
        lines.append(
            f"- input: {ex.get('input')}\n  output: {ex.get('output')}\n"
            f"  verdict: {ex.get('verdict')}  ({ex.get('why', '')})"
        )
    return "\n".join(lines) + "\n\n"


def build_prompt(
    judge: JudgeConfig,
    fmt: str,
    *,
    criterion: str | None = None,
    sections: dict[str, Any] | None = None,
) -> str:
    """User message: criterion, few-shot, evidence sections, output format."""
    parts = [_section("Criterion", criterion or judge.criterion)]
    if judge.rubric and fmt in ("rubric", "geval"):
        scale = "\n".join(f"{k}: {v}" for k, v in sorted(judge.rubric.items()))
        parts.append(_section("Scoring rubric (1-5)", scale))
    parts.append(few_shot_block(judge))
    for title, body in (sections or {}).items():
        parts.append(_section(title, body))
    parts.append(f"## Respond with JSON\n{FORMATS[fmt]}\n")
    return "".join(parts)
