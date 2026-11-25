"""Make targets without make: `uv run python -m evalkit.tasks <target>` (Windows-friendly).

Mirrors the Makefile: install, lint, test, demo, demo-buggy, demo-local, demo-real.
"""

from __future__ import annotations

import os
import subprocess
import sys

PY = sys.executable
DEMO = [
    "-m",
    "evalkit.cli",
    "run",
    "--profile",
    "nightly",
    "--provider",
    "fake",
    "--set",
    "calibration.allow_uncalibrated=true",
]
LOCAL_HELP = """No local model server answered. Start one, then retry:
  Ollama:  ollama serve   &&  ollama pull <model>   (base_url http://localhost:11434/v1)
  vLLM:    vllm serve <model> --port 8000           (base_url http://localhost:8000/v1)
and set EVALKIT_LOCAL_MODEL=<model>."""


def sh(args: list[str], env: dict[str, str] | None = None) -> int:
    print("+", " ".join(args), flush=True)
    return subprocess.call(args, env={**os.environ, **(env or {})})


def evalctl(*args: str, env: dict[str, str] | None = None) -> int:
    return sh([PY, "-m", "evalkit.cli", *args], env)


def demo_local() -> int:
    if evalctl("providers", "check", "--provider", "local") != 0:
        print(LOCAL_HELP)
        return 2
    return evalctl(
        "run",
        "--profile",
        "smoke",
        "--provider",
        "local",
        "--set",
        "target=targets/shopease_real.yaml",
        "--set",
        "calibration.allow_uncalibrated=true",
    )


def demo_real() -> int:
    judge = os.environ.get("JUDGE_PROVIDER", "openrouter")
    model = os.environ.get("EVALKIT_JUDGE_MODEL", "")
    if not model:
        print("Set EVALKIT_JUDGE_MODEL to the judge model id (and its API key env var).")
        return 2
    return evalctl(
        "run",
        "--profile",
        "smoke",
        "--provider",
        "local",
        "--role",
        f"judge={judge}:{model}",
        "--set",
        "target=targets/shopease_real.yaml",
    )


TARGETS = {
    "install": lambda: sh(["uv", "sync", "--extra", "dev"]),
    "lint": lambda: (
        sh([PY, "-m", "ruff", "check", "."])
        or sh([PY, "-m", "ruff", "format", "--check", "."])
        or sh([PY, "-m", "mypy", "evalkit"])
    ),
    "test": lambda: sh([PY, "-m", "pytest", "-q"]),
    "demo": lambda: sh([PY, *DEMO]),
    "demo-buggy": lambda: sh([PY, *DEMO], {"EVALKIT_DEMO_BUGGY": "1"}),
    "demo-local": demo_local,
    "demo-real": demo_real,
}


def main(argv: list[str]) -> int:
    if len(argv) != 1 or argv[0] not in TARGETS:
        print(f"usage: uv run python -m evalkit.tasks {{{'|'.join(TARGETS)}}}")
        return 2
    return int(TARGETS[argv[0]]())


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
