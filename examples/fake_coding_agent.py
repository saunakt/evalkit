"""Offline "coding agent" CLI for the demo: applies a known fix for each demo task.

Usage: python fake_coding_agent.py <prompt_file>   (run with cwd = the sandboxed repo)
Prints a JSON result like a real agent CLI (result text, tokens, cost, command log).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PATCHES = {
    "fix-refund-window": (
        "refunds.py",
        "    return (today - delivered_on).days < WINDOW_DAYS",
        "    return (today - delivered_on).days <= WINDOW_DAYS",
    ),
    "validate-order-id": (
        "orders.py",
        "    return text.strip().upper()",
        "    value = text.strip().upper()\n"
        '    if not re.fullmatch(r"ORD-\\d{6}", value):\n'
        '        raise ValueError(f"invalid order id: {text!r}")\n'
        "    return value",
    ),
}


def main(prompt_file: str) -> int:
    prompt = Path(prompt_file).read_text(encoding="utf-8")
    match = re.search(r"\[task:([\w-]+)\]", prompt)
    task = match.group(1) if match else ""
    if task not in PATCHES:
        print(json.dumps({"result": f"unknown task {task!r}", "commands": []}))
        return 1
    rel, old, new = PATCHES[task]
    path = Path(rel)
    text = path.read_text(encoding="utf-8")
    if task == "validate-order-id" and "import re" not in text:
        text = text.replace('"""Order id helpers."""\n', '"""Order id helpers."""\n\nimport re\n')
    path.write_text(text.replace(old, new), encoding="utf-8")
    print(
        json.dumps(
            {
                "result": f"Applied fix for {task} in {rel}.",
                "tokens_in": 850,
                "tokens_out": 120,
                "cost_usd": 0.0,
                "commands": [f"cat {rel}", f"edit {rel}", "pytest -q tests"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
