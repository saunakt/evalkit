# evalkit make targets (all via uv). Without make: uv run python -m evalkit.tasks <target>
DEMO = uv run evalctl run --profile nightly --provider fake --set calibration.allow_uncalibrated=true

.PHONY: install setup setup-check setup-ci lint test demo demo-buggy demo-local demo-real catalog
PROFILE ?= nightly

install:
	uv sync --extra dev

# Install what install.yaml + evalkit.yaml need (extras via one uv sync, external tools).
setup:
	uv sync --inexact
	uv run --no-sync evalctl setup

# Compare what is installed with install.yaml; exit 1 if something required is missing.
setup-check:
	uv run --no-sync evalctl setup --check

# CI: install exactly requirements.lock if present, else resolve for PROFILE (no external tools).
setup-ci:
	@if [ -f requirements.lock ]; then \
		uv venv --allow-existing && uv pip sync requirements.lock && uv pip install -e . --no-deps; \
	else \
		uv sync --inexact && uv run --no-sync evalctl setup --profile $(PROFILE) --no-external; \
	fi

lint:
	uv run ruff check . && uv run ruff format --check . && uv run mypy evalkit

test:
	uv run pytest -q

catalog:
	uv run evalctl catalog

# Offline demo: every offline suite on the fake provider. Gate should PASS (exit 0).
demo:
	$(DEMO)

# Planted bugs (no stemming in retrieval, cancel rule skipped). Gate should FAIL (exit 1).
demo-buggy:
	EVALKIT_DEMO_BUGGY=1 $(DEMO)

# Every role on a local model server (zero cost). Needs Ollama/vLLM running.
demo-local:
	uv run python -m evalkit.tasks demo-local

# SUT/simulator/perturb/generator local; judge on OpenRouter (or JUDGE_PROVIDER=anthropic).
demo-real:
	uv run python -m evalkit.tasks demo-real
