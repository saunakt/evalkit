# evalkit

A lightweight, config-driven evaluation framework for GenAI and agent systems. One YAML file
picks **which suites** (test types) and **which metrics** run; every run writes a separate
report per selected suite plus a summary with a quality gate.

evalkit is a thin orchestrator: one runner, one results store, one reporting layer. External
tools (DeepEval, Ragas, Detoxify, Presidio, jiwer, garak, Promptfoo, Locust, Langfuse…) are
optional metric providers or case generators, so statistics and reports look the same no
matter which engine scored a metric.

**Zero cost except judges.** Every model call belongs to a role (`judge`, `simulator`,
`perturb`, `generator`, `sut`, `candidate`, `fallback`). Only `judge` is expected to be paid;
everything else defaults to a local OpenAI-compatible server (Ollama, vLLM, LM Studio…). Tests
and the demo use a deterministic `fake` provider and need no network.

## Quickstart (first report in under 10 commands)

evalkit uses [uv](https://docs.astral.sh/uv/) for Python, environments and dependencies
(`uv.lock` is committed). Install uv once: `winget install astral-sh.uv` (Windows) or
`curl -LsSf https://astral.sh/uv/install.sh | sh` (Linux/macOS).

```bash
uv sync --extra dev       # creates .venv (installs Python 3.12 if needed) from uv.lock
uv run evalctl validate
uv run evalctl plan --profile nightly --provider fake
make demo                 # or: uv run python -m evalkit.tasks demo
# open reports/<run_id>/summary.html
make demo-buggy           # planted bugs → the gate fails with named suites/metrics/cases
```

Run every command below inside the project environment: prefix it with `uv run` (or activate
`.venv` first).

`make` is optional: `uv run python -m evalkit.tasks <install|lint|test|demo|demo-buggy|demo-local|demo-real>`
runs the same targets.

## Concepts

| Concept | Meaning |
|---|---|
| Target | Adapter that calls the system under test: `python`, `http`, `cli` or `fake`. |
| Case | One JSONL row in the unified schema (`evalkit/schema.py`). |
| Suite | A test type (`quality`, `rag`, `agent`, `security`, … 18 in total). |
| Metric | A registered function, case- or dataset-scope (217 built in). |
| Judge | An LLM-graded criterion (`judges/*.yaml`) with a calibration record. |
| Profile | An overlay on `evalkit.yaml` (`profiles/*.yaml`). |
| Run | One execution → `results/<run_id>/` + `reports/<run_id>/`. |

See `docs/suites.md` (generated) for every suite and metric.

## Commands

```bash
evalctl validate                                   # config, targets, datasets, judges
evalctl list suites|metrics|engines
evalctl plan  [--profile P] [--provider fake]      # suites × metrics × cases × calls × cost
evalctl run   [--profile P] [--suites a,b] [--metrics x,y] [--tags t] [--split heldout]
              [--repeats N] [--provider local] [--role judge=openrouter:<model>] [--set k=v]
evalctl report <run_id|last> --suites rag --metrics recall_at_k,mrr --format md
evalctl providers check [--provider fake]
evalctl calibrate <judge_id> [--provider fake]
evalctl baseline set <run_id|last>
evalctl evidence <run_id> [--release v1.4]
evalctl catalog [--check]
evalctl sample-size --pass-rate 0.9 --margin 0.05
evalctl generate --suite rag --docs examples/shopease/docs --n 20   # drafts only
evalctl push <run_id> · evalctl pull-traces --suite rag --limit 50   # Langfuse ([observe])
evalctl label export --judge policy_compliance · evalctl label import <export.json> --judge <id>
```

Exit codes: `0` gate passed · `1` gate failed · `2` config/data error · `3` runtime error.

## Configuration

`evalkit.yaml` → `profiles/<name>.yaml` (deep merge) → CLI flags. `${ENV_VAR}` placeholders keep
keys and model ids out of the repo: set `EVALKIT_JUDGE_MODEL`, `EVALKIT_LOCAL_MODEL` and the
provider API key variables (`OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`). Prices are never
hard-coded: add them under `providers.<name>.prices` to get cost estimates and the
`run.max_cost_usd` cap on paid spend.

Judges must be calibrated (`evalctl calibrate <id>` against `datasets/human_labels/<id>.jsonl`,
kappa ≥ 0.6 and FAIL recall ≥ 0.8 for the exact judge model) before their metrics count; set
`calibration.allow_uncalibrated: true` to run them flagged (the demo does this).

## Demo system: ShopEase

`examples/shopease/` is a small support agent (policy RAG + order tools + guardrail) that runs
offline with a scripted `FakeLLM`, or on any provider through the `sut` role
(`targets/shopease_real.yaml`). `EVALKIT_DEMO_BUGGY=1` plants two bugs: retrieval loses
stemming and `cancel_order` skips its status check.

## Choosing what to install

`install.yaml` (repo root) decides what gets installed; `evalkit.yaml` decides what gets
evaluated. Each pyproject extra has a switch — `true`, `false` or `auto` (install only if
evalkit.yaml needs it) — and `mode` picks `choose` (use the switches), `auto` (exactly what is
needed) or `all`. External tools (local model server, promptfoo, docker, WireMock, Label
Studio, …) are `install`, `start`, `check` or `off`.

```yaml
extras:
  mode: choose
  rai: auto          # Presidio + Detoxify (pulls PyTorch): only if a selected metric needs it
  textmetrics: false # BLEU/ROUGE/BERTScore: never
external:
  local_model: { action: check }
  wiremock: { action: start }   # docker compose up -d wiremock
```

```bash
uv run evalctl setup --dry-run   # show the plan and the exact commands; change nothing
make setup                       # install: one `uv sync --extra …`, external tools, requirements.lock
make setup-check                 # what is missing vs install.yaml (exit 1 if something required is)
```

If evalkit.yaml needs an extra you switched to `false`, setup and `evalctl validate` print a
warning naming the metrics that will show as **Not tested**; a `false` is never overridden.
CI uses `make setup-ci PROFILE=<profile>` (installs `requirements.lock` when present).

## Optional extras

`uv sync --extra dev --extra <name>` for any of `anthropic deepeval ragas textmetrics rai
security multilingual voice perf coding agent ui observe`, or `--extra all` for everything.
`security` (garak) cannot share an environment with `anthropic`/`all` because garak pins an old
anthropic SDK; semgrep is a separate CLI (`uv tool install semgrep`). See `docs/DECISIONS.md`. Metrics whose library is missing are reported under **Not tested** as
`<package> not installed`, and the suite gate becomes WARN, never a crash.

`sandbox/docker-compose.yml` starts WireMock (stubs and fault scenarios in
`mocks/wiremock/mappings/`) and Label Studio for local integration work.

## More

- `docs/GUIDE.md` — the user guide: from first run to testing your own system
- `docs/suites.md` — generated suites and metrics reference
- `docs/extending.md` — add a target, metric, suite or engine
- `docs/DECISIONS.md` — choices made where the plan left room
- `Plan.md` — the full specification
