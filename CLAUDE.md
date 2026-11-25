# evalkit — instructions for Claude Code

evalkit is a lightweight, config-driven evaluation framework for GenAI and agent systems.
**`Plan.md` is the source of truth.** Read it fully before writing code.

## How to work
- Build milestone by milestone, in the order of Plan.md §16. Keep a task list with one item per milestone.
- After each milestone, run its ✔ verification commands. Continue only when they pass.
- At the end of each green milestone, commit locally with the message `M<n>: <milestone title>`. Never push.
- When Plan.md is ambiguous, pick the simplest option that keeps the public interfaces in §5 stable and record it in `docs/DECISIONS.md`.
- Before writing a wrapper for a third-party library or CLI, check the installed version's real API (`help(...)`, `--help`). Do not code from memory.
- Finish by running the Definition of Done (Plan.md §17) and report anything that did not pass.

## Hard rules
- **No paid API calls.** Tests and demos use `--provider fake` (or `local` if a local server is running). Never run `make demo-real` or set `--provider openrouter|anthropic` unless the user asks.
- **No vendor SDK in core dependencies.** The built-in model client is the `httpx`-based OpenAI-compatible client. `anthropic` is an optional extra.
- **Optional libraries are imported lazily** and report `skipped: "<package> not installed"`; the core install must work without them.
- **Never hard-code** API keys, prices, or real model ids. Use config and `${ENV_VAR}` placeholders.
- **Do not read or print** `.env` files or API keys.
- **Do not weaken tests or thresholds to get green.** Fix the code. If a check in Plan.md is genuinely wrong, explain why in `docs/DECISIONS.md` before changing it.
- Do not commit `results/`, `reports/`, `.evalkit/`, or `.venv/`.
- **uv only.** Use uv for Python, environments and dependencies; do not use pip or `python -m venv`.
  Add or change dependencies in `pyproject.toml`, then run `uv lock` and commit `uv.lock`.

## Commands
```bash
uv sync --extra dev                                   # .venv from uv.lock: core + dev tools only
uv run ruff check . && uv run ruff format --check . && uv run mypy evalkit
uv run pytest -q                                      # fake model only, no network
make demo                                             # offline demo, gate should pass
make demo-buggy                                       # gate should fail with clear reasons
# without make: uv run python -m evalkit.tasks demo | demo-buggy
```

## Code style
- Python 3.11+, type hints everywhere, `from __future__ import annotations`.
- Small modules: functions ≈ 40 lines max, files ≈ 400 lines max. Plain functions and Pydantic models; at most one level of inheritance.
- One decorator registry for metrics, suites, targets, engines and providers. No `if metric == "x"` chains.
- Deterministic: seeded randomness, sorted outputs, stable ids.
- Every metric gets a unit test with fixtures; every bug fix gets a regression test.
