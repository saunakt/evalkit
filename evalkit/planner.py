"""Config validation and selection → RunPlan (Plan.md §13 "Planner")."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evalkit.config import Project, SuiteConfig, load_judge, read_yaml
from evalkit.registry import IMPLS, METRICS, SUITES, MetricSpec, load_all, metrics_for_suite
from evalkit.schema import Case
from evalkit.util import ConfigError, DataError, did_you_mean


def selected_metric_ids(project: Project, suite_name: str, sc: SuiteConfig) -> list[str]:
    """Resolve a suite's metric selection (default | all | list, minus exclude, CLI filter)."""
    own = [m.id for m in metrics_for_suite(suite_name)]
    suite_cls = SUITES.get(suite_name)
    if sc.metrics == "all":
        ids = own
    elif sc.metrics == "default":
        ids = list(getattr(suite_cls, "default_metrics", own))
    else:
        ids = list(sc.metrics)
    ids = [m for m in ids if m not in sc.exclude]
    cli = project.overrides.metrics
    if cli is not None:
        ignored = [m for m in cli if m not in own]
        if ignored:
            project.notes.append(f"{suite_name}: --metrics ignores {ignored} (not in this suite)")
        ids = [m for m in cli if m in own]
    return ids


def validate_suite_selection(project: Project) -> list[str]:
    errors: list[str] = []
    for name, sc in project.config.suites.items():
        if name not in SUITES:
            errors.append(f"suites: unknown suite {name!r}{did_you_mean(name, SUITES.names())}")
            continue
        if sc.metrics in ("all", "default"):
            continue
        own = {m.id for m in metrics_for_suite(name)}
        for mid in sc.metrics:
            if mid not in METRICS:
                errors.append(
                    f"suites.{name}.metrics: unknown metric {mid!r}"
                    f"{did_you_mean(mid, METRICS.names())}"
                )
            elif mid not in own:
                errors.append(f"suites.{name}.metrics: {mid!r} is not a {name} metric")
    for mid, eng in project.config.engines.items():
        if mid not in METRICS:
            errors.append(f"engines: unknown metric {mid!r}{did_you_mean(mid, METRICS.names())}")
        elif (mid, eng) not in IMPLS:
            avail = sorted(e for (m, e) in IMPLS if m == mid)
            errors.append(f"engines.{mid}: no {eng!r} implementation (available: {avail})")
    return errors


def validate_thresholds(project: Project) -> list[str]:
    errors = []
    for suite_name, metrics in project.thresholds.suites.items():
        if suite_name not in SUITES:
            errors.append(f"thresholds: unknown suite {suite_name!r}")
            continue
        for mid in metrics:
            if mid not in METRICS:
                errors.append(
                    f"thresholds.suites.{suite_name}: unknown metric {mid!r}"
                    f"{did_you_mean(mid, METRICS.names())}"
                )
    return errors


def load_suite_cases(project: Project, suite_name: str, sc: SuiteConfig) -> list[Case]:
    """Dataset rows filtered by tags and splits; heldout file merged when selected."""
    if not sc.dataset:
        return []
    paths = [project.path(sc.dataset)]
    if "heldout" in (sc.params.get("splits") or project.config.splits):
        held = project.path(sc.dataset.replace("golden", "heldout"))
        if held != paths[0] and held.exists():
            paths.append(held)
    cases: list[Case] = []
    for p in paths:
        if not p.exists():
            raise DataError(f"suites.{suite_name}.dataset: file not found: {p}")
        cases.extend(Case.from_jsonl(p))
    splits = set(sc.params.get("splits") or project.config.splits)
    tags = set(sc.tags)
    return [c for c in cases if c.split in splits and (not tags or tags & set(c.tags))]


def validate_project(project: Project) -> list[str]:
    """Return a list of problems (empty = valid). Never spends tokens."""
    load_all()
    errors = validate_suite_selection(project) + validate_thresholds(project)
    tpath = project.path(project.config.target)
    if not tpath.exists():
        errors.append(f"target: file not found: {tpath}")
    else:
        try:
            read_yaml(tpath)
        except ConfigError as exc:
            errors.append(str(exc))
    for name, sc in project.config.suites.items():
        if not sc.enabled or name not in SUITES:
            continue
        try:
            load_suite_cases(project, name, sc)
        except (DataError, ConfigError) as exc:
            errors.append(str(exc))
    errors += validate_judges(project.path(project.config.judges_dir))
    return errors


def validate_judges(judges_dir: Path) -> list[str]:
    errors = []
    for path in sorted(judges_dir.glob("*.yaml")) if judges_dir.exists() else []:
        try:
            judge = load_judge(path)
        except ConfigError as exc:
            errors.append(str(exc))
            continue
        if judge.id != path.stem:
            errors.append(f"{path}: id {judge.id!r} must match the file name {path.stem!r}")
    return errors


def ordered_specs(ids: list[str]) -> list[MetricSpec]:
    """Deterministic metrics first, then statistical/human, then judges (cheap first)."""
    rank = {"deterministic": 0, "statistical": 1, "human": 2, "judge": 3}
    specs = [METRICS.get(i) for i in ids]
    return sorted(specs, key=lambda s: (s.scope == "dataset", rank[s.kind], ids.index(s.id)))


# --------------------------------------------------------------------------- RunPlan

EST_TOKENS_IN = 1500
EST_TOKENS_OUT = 250
TWO_CALL_METRICS = {
    "faithfulness",
    "context_recall",
    "judge_pairwise",
    "pairwise_win_rate",
    "g_eval",
    "citation_accuracy",
}
LLM_PERTURB = {"paraphrase", "translit_hinglish", "language_switch"}
LAST_SUITES = ("migration", "compliance")


@dataclass
class SuitePlan:
    name: str
    suite: Any
    sc: SuiteConfig
    cases: list[Case]
    metric_ids: list[str]
    params: dict[str, Any]
    repeats: int
    calls: dict[str, int] = field(default_factory=dict)


@dataclass
class RunPlan:
    suites: list[SuitePlan]
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    calls: dict[str, int] = field(default_factory=dict)
    cost: dict[str, float | None] = field(default_factory=dict)
    paid_roles: list[str] = field(default_factory=list)
    judges: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def paid_calls(self) -> int:
        return sum(self.calls.get(r, 0) for r in self.paid_roles)

    @property
    def free_calls(self) -> int:
        return sum(n for r, n in self.calls.items() if r not in self.paid_roles)

    @property
    def paid_cost(self) -> float | None:
        vals = [self.cost.get(r) for r in self.paid_roles if self.calls.get(r)]
        return None if any(v is None for v in vals) else sum(v or 0.0 for v in vals)


def _judge_calls(spec: MetricSpec, case: Case) -> int:
    n = max(1, len(case.criteria)) if "criteria" in spec.needs else 1
    return n * (2 if spec.id in TWO_CALL_METRICS else 1)


def estimate_suite(sp: SuitePlan, target: Any) -> dict[str, int]:
    """Estimated LLM calls by role for one suite (judge, simulator, perturb, sut)."""
    calls: dict[str, int] = {}

    def add(role: str, n: int) -> None:
        if n:
            calls[role] = calls.get(role, 0) + n

    specs = [METRICS.get(m) for m in sp.metric_ids]
    for case in sp.cases:
        reps = case.repeats or sp.repeats
        for spec in specs:
            if spec.kind == "judge":
                add("judge", _judge_calls(spec, case) * (reps if spec.scope == "case" else 1))
        if case.persona and case.hidden_goal and case.input is None:
            add("simulator", case.max_turns * reps)
        if "sut" in getattr(target, "uses_roles", []):
            add("sut", 2 * reps)
    perturbs = [
        p for p in sp.params.get("perturbations", []) if str(p).split(":")[0] in LLM_PERTURB
    ]
    add("perturb", len(perturbs) * len(sp.cases))
    return calls


def _role_cost(project: Project, role: str, calls: int) -> tuple[float | None, bool]:
    """(estimated cost, is_paid) for `calls` calls on a role."""
    from evalkit.llm.roles import is_free

    rm = project.config.models.get(role)
    if rm is None:
        return 0.0, False
    cfg = project.config.providers[rm.provider]
    if is_free(cfg):
        return 0.0, False
    price = cfg.prices.get(rm.model)
    if price is None:
        return None, True
    cost = calls * (EST_TOKENS_IN * price.input_per_mtok + EST_TOKENS_OUT * price.output_per_mtok)
    return cost / 1e6, True


def _family(model: str) -> str:
    return "-".join(re.split(r"[-/:._]", model.lower())[:2])


def role_warnings(project: Project, used_roles: set[str]) -> list[str]:
    from evalkit.llm.roles import is_free

    warns = []
    models = project.config.models
    for role in sorted(used_roles - {"judge"}):
        rm = models.get(role)
        if rm is not None and not is_free(project.config.providers[rm.provider]):
            warns.append(f"role {role} uses paid provider {rm.provider}: this role will cost money")
    judge, sut = models.get("judge"), models.get("sut")
    if judge is None or sut is None or "fake" in (judge.provider, sut.provider):
        return warns
    if judge.model == sut.model or _family(judge.model) == _family(sut.model):
        warns.append(
            f"judge and sut resolve to the same model family ({judge.model} / "
            f"{sut.model}): self-preference risk"
        )
    return warns


def build_plan(project: Project, target: Any = None, check_providers: bool = True) -> RunPlan:
    """Validate, select suites/metrics/cases, estimate calls and cost; refuse over the cap."""
    errors = validate_project(project)
    if errors:
        raise ConfigError("\n".join(errors))
    if target is None:
        from evalkit.targets.base import load_target

        target = load_target(project.config.target, project.root)
    plan = RunPlan(suites=[], notes=project.notes)
    names = project.enabled_suites()
    names = [n for n in names if n not in LAST_SUITES] + [n for n in LAST_SUITES if n in names]
    for name in names:
        sc = project.config.suites[name]
        sp = SuitePlan(
            name=name,
            suite=SUITES.get(name)(),
            sc=sc,
            cases=load_suite_cases(project, name, sc),
            metric_ids=[s.id for s in ordered_specs(selected_metric_ids(project, name, sc))],
            params=dict(sc.params),
            repeats=int(sc.params.get("repeats", project.config.run.repeats)),
        )
        sp.calls = estimate_suite(sp, target)
        for role, n in sp.calls.items():
            plan.calls[role] = plan.calls.get(role, 0) + n
        plan.suites.append(sp)
    _finish_plan(project, plan, check_providers)
    return plan


def _finish_plan(project: Project, plan: RunPlan, check_providers: bool) -> None:
    for role, n in sorted(plan.calls.items()):
        cost, is_paid = _role_cost(project, role, n)
        plan.cost[role] = cost
        if is_paid:
            plan.paid_roles.append(role)
    plan.warnings += role_warnings(project, set(plan.calls))
    plan.judges = judge_statuses(project, plan)
    allow = project.config.calibration.allow_uncalibrated
    for jid, st in plan.judges.items():
        if not st.get("calibrated"):
            action = "will run, flagged" if allow else "its metrics will be skipped"
            plan.warnings.append(f"judge {jid} is uncalibrated ({st.get('reason')}); {action}")
    if check_providers:
        reachability_check(project, set(plan.calls))
    cap = project.config.run.max_cost_usd
    paid = plan.paid_cost
    if cap is not None and paid is not None and paid > cap:
        raise ConfigError(
            f"refusing to run: estimated paid spend ${paid:.2f} exceeds run.max_cost_usd=${cap:.2f}"
        )
    if paid is None and plan.paid_calls:
        plan.warnings.append("paid calls have no configured price: cost cannot be estimated")


def judge_statuses(project: Project, plan: RunPlan) -> dict[str, dict[str, Any]]:
    """Calibration status of every judge the selected metrics are known to use."""
    from evalkit.judges.runner import METRIC_JUDGES, JudgeRunner
    from evalkit.llm.roles import LLMRuntime

    ids = sorted(
        {METRIC_JUDGES[m] for sp in plan.suites for m in sp.metric_ids if m in METRIC_JUDGES}
    )
    if not ids or project.config.models.get("judge") is None:
        return {}
    jr = JudgeRunner(
        LLMRuntime(project.config, project.root),
        project.path(project.config.judges_dir),
        project.config.calibration,
    )
    out = {}
    for jid in ids:
        try:
            out[jid] = jr.calibration_status(jr.get(jid))
        except ConfigError as exc:
            out[jid] = {"calibrated": False, "reason": str(exc)}
    return out


def reachability_check(project: Project, roles: set[str]) -> None:
    """Providers used by the plan's roles must answer (results cached for 10 min)."""
    from evalkit.llm.check import check_provider
    from evalkit.llm.roles import LLMRuntime

    cache_path = project.path(".evalkit/providers_check.json")
    cache: dict[str, Any] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    rt = LLMRuntime(project.config, project.root)
    problems = []
    for role in sorted(roles):
        rm = project.config.models.get(role)
        if rm is None or project.config.providers[rm.provider].type == "fake":
            continue
        hit = cache.get(rm.provider)
        if not hit or time.time() - hit["t"] > 600:
            st = check_provider(rt, rm.provider, [role])
            hit = cache[rm.provider] = {"t": time.time(), "ok": st.reachable, "detail": st.detail}
        if not hit["ok"]:
            problems.append(f"provider {rm.provider} (role {role}) unreachable: {hit['detail']}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache), encoding="utf-8")
    if problems:
        raise ConfigError("\n".join(problems))
