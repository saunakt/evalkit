# Decisions

Choices made where Plan.md was ambiguous or the build environment differed. Each keeps the
public interfaces of Plan.md §5 stable.

## Environment

- **No git commits.** The project folder is not a git repository and the owner chose not to
  `git init`, so the per-milestone `M<n>: <title>` commits from CLAUDE.md were skipped.
  `.gitignore` is provided for when the repo is initialised.
- **Windows without `make`.** The `Makefile` is written as specified (CI runs on Linux).
  `uv run python -m evalkit.tasks <target>` runs the same targets on machines without make.
- **Ruff excludes `*.md`.** Ruff 0.17 formats Python blocks inside Markdown; it rewrote the code
  samples in Plan.md once (restored from the original). Markdown is now excluded.
- **UTF-8 console.** `evalctl` reconfigures stdout/stderr to UTF-8 because Windows consoles
  default to cp1252 and reports contain ₹ and Indic scripts.
- **uv only (owner's decision, 2026-11-25).** The project first used pip + `python -m venv`
  as Plan.md was written; the owner then asked to switch everything to uv. uv (installed with
  `winget install astral-sh.uv`) now manages Python, `.venv` and dependencies: `uv sync`,
  `uv run`, `uv lock`, with a committed cross-platform `uv.lock`. Plan.md (§0, §3, §13 sample,
  M1 check, §17) and CLAUDE.md were updated to the uv commands; nothing else in them changed.
  CI uses `astral-sh/setup-uv` and `uv sync --locked`, so a stale lockfile fails the build.
- **Python 3.12.10** was installed with winget (user scope); dependencies are pinned in
  `pyproject.toml` at the latest versions available on 2026-11-25. The supported range stays
  `>=3.11` (Plan.md §2): `uv lock` resolves for 3.11+ on every platform.

## Data model and runner

- `CaseRun` is a NamedTuple `(case, observations, results, repeats)`: the plan's three fields
  plus a parallel `repeats` list so dataset metrics know which repeat produced each result.
- "case-group" robustness metrics are registered with `scope: dataset`; per-group values are in
  `details.groups`.
- Cross-cutting metrics (§9.19) are registered with `suites=("*",)` for the catalog and are
  computed for every suite by the reporting layer; `pass_at_k`, `pass_hat_k` and `mcnemar`
  are registered once by the suites that own them (agent, coding_agent, migration).
- Metric ids are unique across suites; a metric used by several suites (e.g. `task_success`
  in agent and multimodal, `bias_gap` in responsible_ai and finetune) is one spec with several
  suites. The registry holds exactly the 217 ids listed in §9 (a test asserts the list).
- Skip reasons beginning with `n/a:` mean "not applicable to this case" (e.g. the case has no
  `reference`). Other reasons (`<pkg> not installed`, missing hook, uncalibrated judge) mean
  "could not be tested". A selected metric with zero computed rows is listed under
  **Not tested** and makes the suite WARN.
- **Gate:** a number in thresholds.yaml is compared with the metric's mean value; metrics
  without an explicit threshold must reach `defaults.pass_rate` on their case pass rate;
  dataset metrics without a threshold use their own `passed`. **WARN does not fail the run**:
  exit 0 when no suite FAILs and no must-pass case failed (the plan's DoD expects `make demo`
  to exit 0 while WARN is shown for missing extras).
- Must-pass = `case.must_pass` or any tag in `must_pass_tags`; after a must-pass deterministic
  failure, judge metrics for that case are skipped (`run.short_circuit`).
- Python targets run in a thread pool only when the target sets `thread_safe: true`
  (ShopEase keeps sandbox state per thread). Reset → call → state happen in one worker.
- `freezegun` patches process-global state, so a python target opts in with
  `freeze_time: true` and frozen calls are serialised; `run.now` is always exported as
  `EVALKIT_NOW` and passed as `ctx.now`.
- Metrics that call the target again (noise sensitivity, freshness, single-step choice, HITL,
  resilience probes…) reset the sandbox first via `SuiteContext.recall`.
- Suites may skip every metric of a case without calling the target (`skip_reason`); the
  multimodal suite uses it when `supports_media` is false.
- A suite may override the global split selection with `params.splits` (finetune needs dev and
  held-out cases together).
- Run ids get a `-2`, `-3` suffix when two runs start in the same minute.

## LLM providers, judges and cost

- Judge calls are always cacheable; with `run.cache: off` they use a read-write cache anyway.
  The `fake` provider is never cached (it is deterministic and keyed by case metadata).
- `LLMClient.complete` takes an optional `meta` dict (role, case id, judge id, step). The fake
  provider uses it to script verdicts per `(case_id, judge_id)`; real providers ignore it.
- A calibration record is valid only for the judge version **and** the provider:model it was
  calibrated on, so a fake-provider calibration never validates a real judge.
- Inline judges (code-defined criteria) are allowed besides `judges/*.yaml`; a YAML file with
  the same id overrides them. They need their own calibration records.
- The Anthropic client does not send `temperature`: current Claude models reject sampling
  parameters. Its retries use the SDK's own `max_retries`.
- `evalctl providers check` checks only providers used by a role (or all with `--all`), and
  lists models instead of sending a completion where possible, so it never spends money.
- The planner's reachability check is cached for 10 minutes in `.evalkit/providers_check.json`.
- Cost estimates assume 1,500 input / 250 output tokens per judge call (2 calls for claims,
  pairwise and G-Eval); with no configured price the estimate is "unknown" and a warning is
  shown rather than guessing.
- Citation markers like `[returns]` are stripped before claim extraction (they are not claims).

## Suites

- **robustness:** decision labels come from `params.decision`; variants that change wording but
  not the label (e.g. "Kindly cancel…") are expected to keep the decision.
- **security:** the attack set is golden cases + `params.seeds` + native strategy variants
  (base64, leetspeak, roleplay, crescendo, language_switch). Indirect attacks inject the poisoned
  document as a retrieved chunk. The judge criterion asks whether the reply *refused* the attack
  goal (PASS = safe), so a judge that defaults to PASS never inflates ASR.
- **performance:** load mode uses a thread pool for every target type (the http target is
  synchronous httpx); Locust is used when `params.engine: locust`.
- **migration:** each sub-suite case runs twice (`@current` on `models.sut`, `@candidate` on
  `models.candidate`) through the sub-suite's own `execute`, scored with the sub-suite's
  selected case metrics; comparisons are dataset metrics on the matched pairs.
- **finetune:** each case runs once per configured variant target; demo variants all point at
  ShopEase, so gains are zero by construction.
- **coding_agent:** tasks are folders under `params.tasks_dir`; hidden tests are copied into the
  sandbox after the agent finishes. Review-bot and test-generation metrics read the agent's JSON
  output (`review`, `tests_written`) and are not applicable to the two demo fix tasks.
- **skills:** the demo uses a scripted fake target (`targets/skills_fake.yaml`); the fake target
  gained `param_responses` so the with/without comparison can vary on `skill_enabled`.
- **compliance** and **migration** always run last so they can read the other suites' rows.

## ShopEase demo

- Retrieval: BM25 over whole policy snippets with light suffix stemming, whitespace/punctuation
  tokenisation (Indic vowel signs are not `\w`) and a small Hindi/Tamil/Bengali/Hinglish synonym
  map. In buggy mode stemming is off, so "refund" no longer reaches the Returns snippet.
- The FakeLLM asks for confirmation before cancelling unless the user explicitly asked
  ("please cancel", "yes", "confirm"), ignores retrieved chunks that are unscored, low-scored,
  not from `docs/` (provenance check) or contain instructions, and decodes base64 before its
  injection check. The guardrail also checks de-leeted and base64-decoded views.
- Prompt leakage can be forced for the canary test with
  `--set suites.security.params.target_params.fake_leak=true` (plus `guardrail_enabled=false`).
- Two real findings surfaced while building the demo data and were kept as lessons: "How many
  days do I have to return an item?" retrieved the refund-timeline snippet (fixed in the data by
  anchoring on "after delivery"), and noise chunks were used when nothing relevant was retrieved
  (fixed in the bot by ignoring unscored chunks).

## Optional engines not installed in the build environment

The core install was verified without extras. DeepEval 4.3.5, Ragas 0.4.4, MCP 2.3.0,
Langfuse 5.2.0, Presidio 2.2.366, jiwer 4.0.0, langdetect 1.0.9 and textstat 0.7.13 were
installed in a throw-away virtualenv only to read their real APIs; the wrappers follow those
APIs. garak, Promptfoo and Locust were **not** installed (heavy / Node); their wrappers follow
the documented CLIs and must be re-checked with `--help` before first use. Their report parsers
are unit-tested with fixtures.

- Ragas `answer_relevancy` needs an embeddings model, which evalkit does not configure, so the
  `ragas` engine reports it as not tested; use the native or DeepEval engine.
- `evalctl generate` uses the native generator on the `generator` role; Ragas test-set
  generation also needs embeddings and is not wired in.
- DeepEval and Ragas telemetry is switched off (`DEEPEVAL_TELEMETRY_OPT_OUT`,
  `RAGAS_DO_NOT_TRACK`) so no call leaves for anywhere but the judge endpoint; a test enforces
  this with a socket guard and an in-process judge server.

## Dependency pins forced by the extras (found by `uv lock`)

`uv lock` resolves every extra together for every supported Python and platform. The latest
versions do not all fit, so:

- `scipy` is pinned to 1.17.1 (not 1.18.1): 1.18 requires Python >= 3.12, and evalkit supports
  3.11+ (pip on 3.12 never noticed this; uv's universal lock did).
- `rich` is pinned to 14.3.4 (not 15.0.0): deepeval 4.3.5 requires `rich<15`.
- `numpy` is pinned to 2.4.6 (not 2.5.5): presidio-analyzer 2.2.366 requires `numpy<2.5`.
- `mutmut` is pinned to 3.7.0 (not 3.8.1): 3.8 requires `click>=8.4.2`, deepeval requires
  `click<8.4`.
- `semgrep` is not a Python extra: it requires `click~=8.4.2` (conflicts with deepeval) and has
  no native Windows wheels. evalkit calls it as a CLI found on PATH; install it with
  `uv tool install semgrep` (Linux/macOS/WSL).
- `security` (garak) conflicts with `anthropic` (garak 0.17.1 requires `anthropic<1.0`), so it
  is not part of `all` and is declared in `[tool.uv] conflicts`: `uv sync --extra security`
  works in its own environment, or run garak as a tool (`uv tool install garak==0.17.1`).

The lock (356 packages, including torch 2.15, spaCy 3.8, deepeval 4.3.5, ragas 0.4.4) was
verified; a full `uv sync --extra all` was not run on the build machine (several GB, mostly torch
and models). Every optional wrapper is tested with stubs or reports `<package> not installed`.

## install.yaml and `evalctl setup` (M11c)

- **uv, not pip (owner's decision).** The M11c spec was written with pip commands; at the
  owner's request they were translated to uv: one `uv sync --inexact --no-install-project
  --extra …` (inexact keeps extras already installed; `--no-install-project` leaves the running
  evalkit alone, since Windows locks `evalctl.exe` while it runs); `requirements.lock` =
  `uv export --frozen --no-emit-project` filtered so it never contains the project line;
  `make setup-ci` uses `uv pip sync requirements.lock` + `uv pip install -e . --no-deps`.
- **No commit.** The spec asked for a local commit "M11c: install.yaml and evalctl setup"; the
  folder is still not a git repository (owner's earlier choice), so it was skipped.
- **`requires_extra`** is a `MetricSpec` field and a keyword on `@suite/@target/@engine/
  @provider` (stored in `registry.REQUIRES_EXTRA`). Only registrations that cannot work without
  their extra declare one: text-similarity metrics → textmetrics, toxicity_rate → rai,
  garak_scan → security, wer → voice, mcp_contract → agent, ui_streaming → ui, the coding tool
  metrics → coding, the engines and the anthropic provider. Metrics with a built-in fallback
  (pii_leakage_rate's regex, language_match's script check, readability's syllable count) need
  nothing. The catalog now shows the extra per metric.
- **Which `check` tools are "required".** `--check` exits 1 for a missing tool only when an
  enabled feature needs it (e.g. a role that the planned run calls uses a localhost provider →
  `local_model`). The default `local_model: check` therefore does not fail `--check` on a
  machine without a local server while the config doesn't use one; the tool is still shown as
  missing, labelled "not needed by the current config". The same rule applies to extras: only
  the selected extras must be installed.
- **`mode: all`** installs every extra except `security`, which cannot share an environment with
  `anthropic` (uv conflict); a note says so. In `mode: choose`, switching both on is a config
  error rather than a silent choice.
- **YAML `off`.** YAML 1.1 reads a bare `off` as `false`; `action: off` is the documented value,
  so the Tool model accepts `false` as `off`.
- **Local model URL.** `local_model` without a `url` checks `<providers.local.base_url>/models`.

## Validation with real light extras (multilingual, voice, agent)

- Installed for real via `evalctl setup`: langdetect 1.0.9, jiwer 4.0.0, mcp 2.3.0.
- **Bug found and fixed:** langdetect loads its profiles into a global on first use and that load
  is not thread-safe. Concurrent first calls from the runner's thread pool failed (the error was
  swallowed as `None`) or returned a wrong language (`bn` for Hindi), failing the nightly gate.
  `detect_language` now serialises calls with a lock and catches only `LangDetectException`.
  Regression test: `tests/unit/test_language_detect.py` (fresh subprocess, 8 threads).
- **MCP:** `mcp_contract` passes against a real mcp 2.x stdio server
  (`tests/fixtures/mcp_shop_server.py`). A server command that cannot start is now a contract
  failure ("cannot start MCP server …") instead of an unhandled error.
- `wer` stays Not tested in the demo because the ShopEase bot is text-only
  (`supports_media: false`); the jiwer wrapper was checked directly against known WER values.
