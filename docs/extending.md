# Extending evalkit

Everything is registered with one decorator registry (`evalkit/registry.py`). Built-in modules
under `evalkit/{llm,metrics,suites,targets,engines}/` are imported automatically; for your own
code, import the module before running (e.g. from a target's `entry` module or a small plugin
package listed on `PYTHONPATH`). After adding metrics or suites, run `evalctl catalog` so
`metrics/catalog.yaml` and `docs/suites.md` stay current (CI runs `evalctl catalog --check`).

## Add a target

Python callables need no code at all — point a YAML file at them:

```yaml
# targets/my_bot.yaml
name: my-bot
type: python
entry: my_package.bot:answer        # answer(input_or_messages, ctx) -> str | dict | Observation
reset: my_package.bot:reset         # optional hooks: reset, state, ingest, update_doc, stream
state: my_package.bot:state
thread_safe: true                   # cases run in parallel only when true
```

A new target *type* is a factory registered with `@target`:

```python
from pathlib import Path
from typing import Any

from evalkit.registry import target
from evalkit.schema import Case, Observation
from evalkit.targets.base import TargetContext


class GrpcTarget:
    def __init__(self, cfg: dict[str, Any], root: Path) -> None:
        self.name = cfg.get("name", "grpc")
        self.address = cfg["address"]
        self.thread_safe = True

    def call(self, case: Case, ctx: TargetContext) -> Observation:
        reply = my_grpc_client(self.address).ask(case.input, history=ctx.messages)
        return Observation(text=reply.text, tokens_in=reply.usage.input)


@target("grpc")
def make_grpc(cfg: dict[str, Any], root: Path) -> GrpcTarget:
    return GrpcTarget(cfg, root)
```

## Add a metric

```python
from evalkit.metrics.common import boolean, na
from evalkit.registry import metric
from evalkit.schema import Case, Observation
from evalkit.suites.context import MetricContext


@metric("mentions_order_id", suites=("quality", "agent"), kind="deterministic",
        needs=("extra.order_id",))
def mentions_order_id(case: Case, obs: Observation, ctx: MetricContext):
    """The reply repeats the customer's order id."""
    return boolean(case.extra["order_id"] in obs.text)
```

- `needs` entries skip the metric with a clear reason when missing: case fields (`reference`,
  `a|b` for either), `obs.<field>`, `hook:<name>`, `params.<key>`, `tag:<tag>`,
  `pkg:<module>=<PyPI name>` (reported as `<PyPI name> not installed`).
- Dataset metrics use `scope="dataset"` and receive `(rows: list[CaseRun], ctx)`.
- Judge metrics call `ctx.judge.binary/rubric/pairwise/geval/claims(...)`; define the criterion
  in `judges/<id>.yaml` or with `inline_judge(...)`, and calibrate it.
- Parameters come from `suites.<suite>.params` (`ctx.p("key")` also checks
  `params.<metric_id>.key`).
- Add a unit test with fixtures (see `tests/unit/test_metrics_*.py` and
  `tests/metric_helpers.py`).

An alternative implementation of an existing metric: `@metric_impl("faithfulness",
engine="myengine")`, selected with `engines: {faithfulness: myengine}` in `evalkit.yaml`.

## Add a suite

```python
from evalkit.registry import suite
from evalkit.schema import Case
from evalkit.suites.base import BaseSuite
from evalkit.suites.context import SuiteContext


@suite("billing")
class BillingSuite(BaseSuite):
    """Invoices and payment questions."""

    name = "billing"
    default_metrics = ("contains", "numeric_tolerance", "judge_binary")

    def prepare(self, cases: list[Case], sctx: SuiteContext) -> list[Case]:
        return [c for c in cases if "billing" in c.tags]   # expand / filter cases here
```

Override only what differs: `prepare`, `execute` (how a case calls the target),
`case_metric_ids`, `skip_reason`, `panels` (report tables). Then add a block under `suites:`
in `evalkit.yaml` and register metrics for it (`suites=("billing", ...)`).

## Add an engine

Engines wrap optional libraries. Keep each in one file, import the library lazily with
`optional_import`, and check the installed version's real API first.

```python
from evalkit.engines.judge_bridge import JudgeBridge, disable_telemetry
from evalkit.metrics.common import result
from evalkit.registry import engine, metric_impl
from evalkit.util import optional_import


@engine("mylib")
class MyLibEngine:
    package = "mylib"


@metric_impl("answer_relevancy", engine="mylib")
def answer_relevancy(case, obs, ctx):
    mylib = optional_import("mylib")                    # → skipped: "mylib not installed"
    disable_telemetry()
    judge = JudgeBridge(ctx.judge.runtime.client("judge"), "mylib", case.id)
    score = mylib.relevancy(case.input, obs.text, llm=judge.generate)   # judge role only
    return result(score, score >= 0.5)
```

Route every LLM call a library makes through the judge role (as above), so it is cached,
priced, capped and counted under `judge` in the cost ledger.
