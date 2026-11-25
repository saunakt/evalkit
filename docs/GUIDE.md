# evalkit user guide

evalkit tests GenAI and agent systems (chatbots, RAG apps, tool-using agents, coding agents)
from a config file. You describe **what system to call**, **which test suites to run** and
**what scores count as good enough**; evalkit runs the cases, scores them, writes reports and
gives a PASS/FAIL gate you can use in CI.

This guide goes from a first run to testing your own system. Commands assume Windows
Git Bash or any POSIX shell, run from the repository root.

---

## 1. Install

Requirements: Python 3.11+ and [uv](https://docs.astral.sh/uv/) (`winget install astral-sh.uv`).
uv is the only supported installer: no pip, no `python -m venv`.

```bash
uv sync --extra dev          # creates .venv from uv.lock: core + dev tools
uv run evalctl --help        # the CLI
```

Without `make`, every Make target has a Python equivalent:
`uv run python -m evalkit.tasks demo` (or `demo-buggy`, `install`, …).

### Optional features: install.yaml

The core install runs every suite offline. Some metrics need extra libraries (e.g. BERTScore
needs PyTorch). Rather than installing everything, `install.yaml` decides what to add:

```yaml
extras:
  mode: choose          # choose = use the switches · auto = exactly what evalkit.yaml needs · all
  multilingual: true    # true = install, false = never, auto = only if a selected metric needs it
  textmetrics: false    # BLEU/ROUGE/BERTScore: pulls PyTorch
external:
  local_model: { action: check }   # install | start | check | off
```

```bash
uv run evalctl setup --dry-run   # show what would be installed and the exact commands
make setup                       # install it (one `uv sync`), write requirements.lock
make setup-check                 # what is missing? exit 1 if something required is
```

If a suite needs an extra you set to `false`, `setup` and `validate` warn you and the
affected metrics appear as **Not tested** in reports. They never fail silently.

---

## 2. Your first run: the ShopEase demo

The repo ships with a demo support bot (`examples/shopease/`): it answers return, shipping and
payment questions from a small knowledge base and can look up or cancel orders. All model
calls go to a scripted **fake** provider, so the demo is free and offline.

```bash
make demo          # nightly profile, 17 suites, fake provider → GATE PASSED (exit 0)
make demo-buggy    # same, with planted bugs in the bot → GATE FAILED (exit 1)
```

Open the report printed at the end, e.g. `reports/<run_id>/summary.html`. Compare the two
runs to see what a failure looks like: the summary names the failing suites, metrics and
must-pass cases, and each suite report lists the failing cases with the checks they failed.

Note: in the demo, LLM judges are fake too (they answer PASS), so judge-based scores there
prove the plumbing works, not the bot's quality.

---

## 3. Core concepts

| Concept | What it is | Where it lives |
|---|---|---|
| **Target** | Adapter that calls the system under test: `python`, `http`, `cli` or `fake` | `targets/*.yaml` |
| **Case** | One test: an input plus expectations, one JSON object per line | `datasets/**/*.jsonl` |
| **Suite** | A kind of testing (quality, rag, agent, security…), 18 in all | `evalkit.yaml → suites:` |
| **Metric** | A scoring function, per case or per dataset (217 built in) | `docs/suites.md` |
| **Judge** | An LLM-graded criterion with a version and a calibration record | `judges/*.yaml` |
| **Role** | Which model does which job: `judge`, `sut`, `simulator`, `perturb`, `generator`… | `evalkit.yaml → models:` |
| **Profile** | An overlay on evalkit.yaml (smoke, nightly, release…) | `profiles/*.yaml` |
| **Thresholds** | What counts as passing; drives the gate | `thresholds.yaml` |
| **Run** | One execution | `results/<run_id>/`, `reports/<run_id>/` |

**Config merge order:** `evalkit.yaml` → `profiles/<name>.yaml` → CLI flags (`--set`, `--role`,
`--suites`…). Later wins.

---

## 4. The everyday workflow

```bash
uv run evalctl validate --profile nightly        # check config, targets, datasets, judges
uv run evalctl plan --profile nightly            # suites × metrics × cases × calls × cost
uv run evalctl run --profile nightly             # run, store, report, gate
uv run evalctl report <run_id> --suites rag --format md   # re-render a subset, no re-run
```

- **validate** catches typos with suggestions (`unknown key 'metircs' — did you mean 'metrics'?`)
  and exits 2 on any config error.
- **plan** shows how many model calls a run will make and its estimated **paid** cost.
  A run estimated above `run.max_cost_usd` is refused before anything is called.
- **run** exit codes: `0` gate PASS (or WARN) · `1` gate FAIL · `2` config error · `3` runtime error.

### Narrowing a run

```bash
uv run evalctl run --suites quality,rag                 # only these suites
uv run evalctl run --metrics recall_at_k,faithfulness   # only these metrics
uv run evalctl run --tags refund,money                  # only cases with these tags
uv run evalctl run --split dev,heldout --repeats 5
uv run evalctl run --set run.parallel=4 --set suites.rag.params.k=3
```

Reports only show what you selected; everything else is listed under **Not tested** with the
reason, so a report never implies coverage it does not have.

---

## 5. Testing your own system

### Step 1: write a target

**HTTP API** (copy `targets/http_chat.example.yaml`):

```yaml
name: my-chat-api
type: http
url: https://chat.example.internal/v1/answer
method: POST
headers: { Authorization: "Bearer ${CHAT_API_TOKEN}" }     # secrets come from env vars
body: |
  {"question": {{ input | tojson }}, "history": {{ messages | tojson }}}
response:                       # dotted paths into the JSON response
  text: data.answer
  tool_calls: data.trace.tools  # needed for agent metrics
  retrieved: data.trace.chunks  # needed for RAG retrieval metrics
  tokens_in: usage.input_tokens
  tokens_out: usage.output_tokens
stream: sse                     # optional: measures time-to-first-token
reset_url: https://chat.example.internal/v1/test/reset    # optional: clean state per case
state_url: https://chat.example.internal/v1/test/state    # optional: for end-state checks
timeout_s: 60
retries: 3
```

**Python function** (see `targets/shopease_agent.yaml`): `type: python`,
`entry: my_pkg.bot:ask`, plus optional `reset`, `state`, `stream` hooks.

**CLI program** (e.g. a coding agent): see `targets/cli_coding_agent.example.yaml`.

Point evalkit at it: `target: targets/my-chat-api.yaml` in `evalkit.yaml`.
The more the target returns (tool calls, retrieved chunks, tokens, state), the more metrics
can run; metrics whose inputs are missing are reported as Not tested, not as failures.

### Step 2: write cases

One JSON object per line. Only `id` and an input are required; add the expectations you care
about. Common fields:

```json
{"id": "q-refund-window", "tags": ["refund", "money"], "input": "Can I return shoes delivered 10 days ago?",
 "must_include": ["30-day"], "must_not_include": ["three months"],
 "criteria": ["States that items can be returned within 30 days of delivery"]}
```

| Field | Used for |
|---|---|
| `input` / `turns` / `persona` + `hidden_goal` | single question / scripted conversation / simulated user |
| `reference`, `must_include`, `must_not_include`, `regex`, `schema` | deterministic answer checks |
| `criteria` | LLM-judge checks (plain-language PASS/FAIL criteria) |
| `relevant_ids`, `relevance` | RAG retrieval metrics (recall@k, MRR, nDCG) |
| `expect_tools`, `forbid_tools`, `expect_trajectory`, `expect_state`, `policies` | agent metrics |
| `attack` | security cases (`{"owasp": "LLM01", "vector": "direct", "goal": "..."}`) |
| `language` | multilingual (`hi`, `hi-Latn`, `ta`…) |
| `must_pass: true` or a tag in `must_pass_tags` | the case must pass in **every** repeat |
| `split: heldout` | kept out of everyday runs; used in release runs |
| `repeats` | per-case override of `run.repeats` |

The full schema is `evalkit/schema.py` (Plan.md §5). `datasets/golden/*.jsonl` has an example
file for every suite.

**Need more cases?** `uv run evalctl generate --suite rag --docs <folder> --n 20` drafts cases from your documents into
`datasets/drafts/` (uses the `generator` role). Review them before moving them to a golden file.
`uv run evalctl sample-size --pass-rate 0.9 --margin 0.05` tells you how many cases you need for
a given margin of error.

### Step 3: enable suites

```yaml
suites:
  quality:
    enabled: true
    dataset: datasets/golden/quality.jsonl
    metrics: [contains, not_contains, regex, judge_binary]   # omit = all metrics of the suite
    params: { max_words: 120 }
  rag:
    enabled: true
    dataset: datasets/golden/rag.jsonl
    metrics: [recall_at_k, mrr, faithfulness, answer_relevancy]
    params: { k: 5 }
```

`uv run evalctl list suites` / `list metrics` / `list engines` show what exists;
`docs/suites.md` documents every suite, metric, parameter and required extra.

The suites: `quality`, `prompt`, `robustness`, `conversation`, `rag`, `agent`, `coding_agent`,
`skills`, `security`, `guardrails`, `responsible_ai`, `compliance`, `performance`,
`multimodal`, `multilingual`, `finetune`, `classic`, `migration`.

### Step 4: set thresholds

`thresholds.yaml`:

```yaml
defaults: { pass_rate: 0.90 }
must_pass_tags: [security, legal, money]      # every repeat must pass
suites:
  rag: { recall_at_k: 0.90, faithfulness: 0.85 }       # a number = minimum for higher-is-better
  security: { attack_success_rate: { max: 0.02 } }     # lower-is-better metrics use max
  performance: { latency_p95: { max: 5.0 } }
regression:
  baseline: auto
  max_slice_drop: 0.05
  require_significance: true
```

The gate passes when every selected suite passes **and** every must-pass case passed in every
repeat. A suite gets **WARN** (exit 0) instead of PASS when a judge is uncalibrated, metrics
were skipped, or there are too few cases for a trustworthy number.

---

## 6. Models, judges and cost

Roles map jobs to providers in `evalkit.yaml`:

```yaml
providers:
  local:      { type: openai_compatible, base_url: http://localhost:11434/v1, api_key: none }
  openrouter: { type: openai_compatible, base_url: https://openrouter.ai/api/v1,
                api_key: "${OPENROUTER_API_KEY}", prices: {} }
  anthropic:  { type: anthropic, api_key: "${ANTHROPIC_API_KEY}" }   # needs the anthropic extra
  fake:       { type: fake, script: examples/shopease/fake_script.yaml }
models:
  judge:     { provider: openrouter, model: "${EVALKIT_JUDGE_MODEL}" }
  simulator: { provider: local, model: "${EVALKIT_LOCAL_MODEL}" }
```

- Model ids, keys and prices come from env vars and config, never from code.
- `--provider fake` maps every role to the fake provider (free, offline: use it for tests).
- `--role judge=openrouter:<model id>` overrides a single role.
- `uv run evalctl providers check` probes each provider in use: reachable, latency,
  free or paid, JSON mode.
- Localhost providers count as free. Paid spend is tracked separately, capped by
  `run.max_cost_usd`, and shown at the end of each run (`paid spend $…`).
- Responses are cached in `.evalkit/` (`run.cache: read_write | replay_only | off`), so
  re-runs of unchanged cases cost nothing.

### Judges

A judge is a YAML file (`judges/*.yaml`) with an id, a version, a style (`binary`, `rubric`,
`reference`, `pairwise` with position swap, `geval`, `claims`) and a criterion. Before you trust
a judge, **calibrate** it against human labels:

```bash
uv run evalctl label export --run <run_id> --judge faithfulness   # outputs → labelstudio/ for labelling
uv run evalctl label import <export.json> --judge faithfulness     # → datasets/human_labels/faithfulness.jsonl
uv run evalctl calibrate --role judge=openrouter:<model id>
```

Calibration writes `judges/calibration/<id>@v<version>.json` with Cohen's kappa and FAIL
recall for that judge version and model. Below `calibration.min_kappa` / `min_fail_recall`,
or with no calibration record, suites using the judge get WARN. For dry runs you can set
`--set calibration.allow_uncalibrated=true` (the demo does this).

---

## 7. Profiles and CI

| Profile | Use |
|---|---|
| `smoke` | fast PR check: quality, a few rag/agent/security metrics, 1 repeat, $2 cap |
| `nightly` | broad coverage, every offline suite |
| `release` | everything including the heldout split, evidence pack |
| `security` | red-team focus |
| `migration` | compare a `candidate` model against the current one |

GitHub workflows in `.github/workflows/` (`pr-smoke`, `nightly`, `release`, `self-test`)
install with `make setup-ci PROFILE=<profile>` and use the exit code as the check result; JUnit
output (`junit.xml`) makes failures show up natively in CI.

### Regression tracking

```bash
uv run evalctl baseline set <run_id>     # mark a good run as the baseline
```

With `regression.baseline: auto`, each run is compared against it: the report lists fixed and
**newly broken** cases, and fails on slice drops larger than `max_slice_drop`, only when the drop
is statistically significant (McNemar / bootstrap) if `require_significance` is on.

---

## 8. Reading reports

`reports/<run_id>/`:

- `summary.html` / `summary.md`: gate verdict, one row per suite (status, key metrics with
  confidence intervals, n, cost), a go/no-go checklist, and the **Not tested** list.
- `<suite>/report.html` (also `.md`, `.json`, `junit.xml`, `per_case.csv`), in this order:
  verdict, what was tested, results vs thresholds, comparison with baseline, results by slice
  (tags, language…), failing and flaky cases with the checks they failed and the judge's
  reasoning, suite panels (OWASP table, confusion matrix, latency percentiles…), judge validity,
  not tested.
- `compare.html` when a baseline exists.

HTML reports are single self-contained files: no external assets, safe to attach or archive.
Raw data is in `results/<run_id>/` (`results.jsonl`, `observations.jsonl`, `cases_snapshot.jsonl`,
`run_info.json`). `observations.jsonl` holds full model outputs and may contain sensitive data;
it is git-ignored.

**Evidence pack** for audits and releases: `uv run evalctl evidence <run_id>` writes
`evidence/<release>/` with a system card (from the `system:` block), risk register (from
`risks:`, linked to case ids and their latest results), test plan, dataset cards, the reports
and judge calibration records.

---

## 9. Observability (optional)

With the `observe` extra and `observe.langfuse: true`:

```bash
uv run evalctl push <run_id>        # datasets per suite, items per case, scores per metric
uv run evalctl pull-traces          # flagged production traces → PII-scrubbed draft cases
```

`observe.trace_url_template` adds a link to each failing case's trace in the reports.

---

## 10. Extending evalkit

Metrics, suites, targets, engines and providers all use one decorator registry, so adding one
is a single function or class. See `docs/extending.md`. Example:

```python
@metric("polite", suites=("quality",), kind="deterministic")
def polite(case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    """Output says please or thank you."""
    return boolean(any(w in obs.text.lower() for w in ("please", "thank")))
```

Then run `uv run evalctl catalog` to regenerate `metrics/catalog.yaml` and `docs/suites.md`
(CI checks they are up to date with `evalctl catalog --check`), and add a unit test.
Metrics that need an optional library declare `requires_extra="..."`, import it lazily, and
report `skipped: "<package> not installed"` when it is missing.

---

## 11. Cheat sheet

```bash
# install
uv sync --extra dev
uv run evalctl setup --dry-run && make setup

# check before spending anything
uv run evalctl validate --profile nightly
uv run evalctl plan --profile nightly
uv run evalctl providers check

# run
uv run evalctl run --profile smoke --provider fake       # free, offline
uv run evalctl run --profile nightly                      # real providers from evalkit.yaml
uv run evalctl run --suites rag --tags refund --repeats 1

# after a run
uv run evalctl report <run_id> --suites rag --format md
uv run evalctl baseline set <run_id>
uv run evalctl evidence <run_id>

# develop evalkit itself
uv run ruff check . && uv run ruff format --check . && uv run mypy evalkit
uv run pytest -q
uv run evalctl catalog --check
```

## 12. Troubleshooting

| Symptom | Fix |
|---|---|
| `error: …: did you mean …?`, exit 2 | Typo in a config file; `evalctl validate` lists every one |
| Suite shows **WARN** | Judge uncalibrated, metrics skipped, or too few cases: read the suite report's Not tested and Judge validity sections |
| Metric **Not tested: `<pkg>` not installed** | Turn its extra on in `install.yaml` and run `make setup` |
| Metric **Not tested: needs …** | Add the missing case field or target output (e.g. `relevant_ids`, `response.tool_calls`) |
| Run refused: estimated cost above cap | Narrow the run, lower `--repeats`, or raise `run.max_cost_usd` deliberately |
| `local_model` missing in `setup --check` | Start Ollama / vLLM, or ignore it if no role uses the `local` provider |
| Garbled characters in a Windows console | `export PYTHONIOENCODING=utf-8` |
| `uv sync` fails with access denied under OneDrive | Clear read-only flags: `attrib -R .venv\* /S /D` |
