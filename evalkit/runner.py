"""Execute a RunPlan and write the results store (Plan.md §7, §9, §11)."""

from __future__ import annotations

import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from evalkit import __version__
from evalkit.config import Project
from evalkit.env.clock import exported_now
from evalkit.execution import run_case_metric, run_dataset_metric
from evalkit.gate import GateResult, evaluate
from evalkit.judges.runner import JudgeRunner
from evalkit.llm.roles import LLMRuntime, set_runtime
from evalkit.planner import RunPlan, SuitePlan
from evalkit.registry import METRICS, MetricSpec
from evalkit.reporting.aggregate import SuiteSummary, summarize_suite
from evalkit.reporting.store import ResultStore, StoredRun, find_baseline
from evalkit.schema import Case, CaseRun, MetricResult, Observation, ResultRow, RunInfo
from evalkit.suites.context import MetricContext, SuiteContext
from evalkit.targets.base import has_hook, load_target
from evalkit.util import sha256_file


@dataclass
class RunOutcome:
    run_id: str
    run_dir: Path
    info: RunInfo
    rows: list[ResultRow]
    summaries: dict[str, SuiteSummary]
    gate: GateResult
    panels: dict[str, list[dict[str, Any]]] = field(default_factory=dict)


@dataclass
class CaseOutcome:
    case: Case
    repeat: int
    obs: Observation
    results: list[MetricResult]


# --------------------------------------------------------------------------- ids


def git_info(root: Path) -> tuple[str | None, bool | None]:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=10
        )
        if sha.returncode != 0:
            return None, None
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, timeout=10
        )
        return sha.stdout.strip(), bool(dirty.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return None, None


def make_run_id(results_dir: Path, sha: str | None, profile: str | None) -> str:
    base = f"{datetime.now():%Y-%m-%d_%H%M}_{(sha or 'nogit')[:7]}_{profile or 'default'}"
    run_id, n = base, 1
    while (results_dir / run_id).exists():
        n += 1
        run_id = f"{base}-{n}"
    return run_id


# --------------------------------------------------------------------------- per case


def must_pass(case: Case, tags: list[str]) -> bool:
    return case.must_pass or bool(set(case.tags) & set(tags))


def case_metrics(
    case: Case,
    obs: Observation,
    repeat: int,
    specs: list[MetricSpec],
    sctx: SuiteContext,
    engines: dict[str, str],
    short_circuit: bool,
    mp_tags: list[str],
) -> list[MetricResult]:
    """Deterministic first, then judges; short-circuit judges after a must-pass failure."""
    must = must_pass(case, mp_tags)
    failed: str | None = None
    out = []
    for spec in specs:
        if spec.kind == "judge" and short_circuit and must and failed:
            out.append(MetricResult(metric=spec.id, skipped=f"short_circuit: {failed} failed"))
            continue
        mctx = MetricContext(sctx.params, sctx.judge, sctx, repeat, spec.id)
        res = run_case_metric(spec, engines.get(spec.id, "native"), case, obs, mctx)
        if spec.kind == "deterministic" and res.passed is False and failed is None:
            failed = spec.id
        out.append(res)
    return out


def run_one(
    sp: SuitePlan,
    sctx: SuiteContext,
    case: Case,
    repeat: int,
    specs: list[MetricSpec],
    project: Project,
) -> CaseOutcome:
    run = project.config.run
    reason = sp.suite.skip_reason(case, sctx)
    if reason:
        skipped = [MetricResult(metric=s.id, skipped=reason) for s in specs]
        return CaseOutcome(case, repeat, Observation(error=None), skipped)
    if run.reset_between_cases:
        sctx.reset(run.seed + repeat)
    obs = sp.suite.execute(case, repeat, sctx)
    if not obs.state and has_hook(sctx.target, "state"):
        try:
            obs.state = sctx.state()
        except Exception as exc:  # state hook failures are recorded, not fatal
            obs.error = obs.error or f"state hook failed: {exc}"
    ids = set(sp.suite.case_metric_ids(case, [s.id for s in specs], sctx))
    chosen = [s for s in specs if s.id in ids] + [
        METRICS.get(i) for i in sorted(ids - {s.id for s in specs})
    ]
    results = case_metrics(
        case,
        obs,
        repeat,
        chosen,
        sctx,
        project.config.engines,
        run.short_circuit,
        project.thresholds.must_pass_tags,
    )
    return CaseOutcome(case, repeat, obs, results)


def run_cases(
    sp: SuitePlan, sctx: SuiteContext, cases: list[Case], project: Project
) -> list[CaseOutcome]:
    specs = [METRICS.get(m) for m in sp.metric_ids if METRICS.get(m).scope == "case"]
    jobs = [(c, r) for c in cases for r in range(c.repeats or sp.repeats)]
    parallel = project.config.run.parallel
    if parallel > 1 and getattr(sctx.target, "thread_safe", False):
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            futures = [pool.submit(run_one, sp, sctx, c, r, specs, project) for c, r in jobs]
            return [f.result() for f in futures]
    return [run_one(sp, sctx, c, r, specs, project) for c, r in jobs]


def to_case_runs(outcomes: list[CaseOutcome]) -> list[CaseRun]:
    grouped: dict[str, list[CaseOutcome]] = {}
    for o in outcomes:
        grouped.setdefault(o.case.id, []).append(o)
    runs = []
    for items in grouped.values():
        items.sort(key=lambda o: o.repeat)
        results = [r for o in items for r in o.results]
        reps = [o.repeat for o in items for _ in o.results]
        runs.append(CaseRun(items[0].case, [o.obs for o in items], results, reps))
    return runs


def row_for(
    run_id: str,
    suite: str,
    res: MetricResult,
    case: Case | None = None,
    repeat: int | None = None,
    obs: Observation | None = None,
    must: bool = False,
) -> ResultRow:
    return ResultRow(
        run_id=run_id,
        suite=suite,
        case_id=case.id if case else None,
        repeat=repeat,
        metric=res.metric,
        value=res.value,
        passed=res.passed,
        threshold=res.threshold,
        skipped=res.skipped,
        details=res.details,
        tags=list(case.tags) if case else [],
        language=case.language if case else None,
        split=case.split if case else None,
        must_pass=must,
        latency_s=obs.latency_s if obs else None,
        tokens=(obs.tokens_in + obs.tokens_out) if obs else None,
        cost_usd=obs.cost_usd if obs else None,
        trace_id=obs.trace_id if obs else None,
    )


# --------------------------------------------------------------------------- per suite


@dataclass
class SuiteOutput:
    rows: list[ResultRow]
    observations: list[dict[str, Any]]
    cases: list[Case]
    panels: list[dict[str, Any]]
    data_paths: list[Path]


def run_suite(sp: SuitePlan, sctx: SuiteContext, project: Project, run_id: str) -> SuiteOutput:
    cases = sp.suite.prepare(sp.cases, sctx)
    outcomes = run_cases(sp, sctx, cases, project)
    mp_tags = project.thresholds.must_pass_tags
    rows = [
        row_for(run_id, sp.name, r, o.case, o.repeat, o.obs, must_pass(o.case, mp_tags))
        for o in outcomes
        for r in o.results
    ]
    runs = to_case_runs(outcomes)
    for mid in sp.metric_ids:
        spec = METRICS.get(mid)
        if spec.scope != "dataset":
            continue
        mctx = MetricContext(sctx.params, sctx.judge, sctx, 0, mid)
        for res in run_dataset_metric(spec, project.config.engines.get(mid, "native"), runs, mctx):
            rows.append(row_for(run_id, sp.name, res))
    obs_rows = [
        {
            "suite": sp.name,
            "case_id": o.case.id,
            "repeat": o.repeat,
            "observation": o.obs.model_dump(mode="json"),
        }
        for o in outcomes
    ]
    try:
        panels = sp.suite.panels(rows, runs, sctx)
    except Exception as exc:  # a panel must never break the run
        panels = [{"title": "Panel error", "kind": "text", "text": f"{type(exc).__name__}: {exc}"}]
    paths = [project.path(sp.sc.dataset)] if sp.sc.dataset else []
    return SuiteOutput(rows, obs_rows, cases, panels, [p for p in paths if p.exists()])


def suite_context(
    sp: SuitePlan,
    project: Project,
    rt: LLMRuntime,
    judge: JudgeRunner,
    target: Any,
    run_dir: Path,
    baseline: StoredRun | None,
    all_rows: list[ResultRow],
) -> SuiteContext:
    target_ref = sp.sc.target
    tgt = load_target(target_ref, project.root) if target_ref else target
    return SuiteContext(
        name=sp.name,
        project=project,
        sc=sp.sc,
        params=sp.params,
        target=tgt,
        runtime=rt,
        judge=judge,
        run_dir=run_dir,
        seed=project.config.run.seed,
        repeats=sp.repeats,
        now=project.config.run.now,
        baseline=baseline,
        run_rows=lambda: list(all_rows),
    )


# --------------------------------------------------------------------------- run


def run(project: Project, plan: RunPlan, target: Any = None) -> RunOutcome:
    """Execute every suite in the plan; write results/<run_id>/; evaluate the gate."""
    cfg = project.config
    results_dir = project.path(cfg.results_dir)
    sha, dirty = git_info(project.root)
    run_id = make_run_id(results_dir, sha, project.profile)
    run_dir = results_dir / run_id
    store = ResultStore(run_dir)
    started = datetime.now().astimezone().isoformat(timespec="seconds")
    rt = LLMRuntime(cfg, project.root)
    set_runtime(rt)
    judge = JudgeRunner(rt, project.path(cfg.judges_dir), cfg.calibration)
    target = target or load_target(cfg.target, project.root)
    baseline = find_baseline(results_dir, project.thresholds.regression.baseline, run_id)
    all_rows: list[ResultRow] = []
    obs_rows: list[dict[str, Any]] = []
    snapshot: list[dict[str, Any]] = []
    panels: dict[str, list[dict[str, Any]]] = {}
    datasets: dict[str, dict[str, Any]] = {}
    try:
        with exported_now(cfg.run.now):
            for sp in plan.suites:
                sctx = suite_context(sp, project, rt, judge, target, run_dir, baseline, all_rows)
                out = run_suite(sp, sctx, project, run_id)
                all_rows.extend(out.rows)
                obs_rows.extend(out.observations)
                snapshot.extend(
                    {**c.model_dump(mode="json", by_alias=True), "_suite": sp.name}
                    for c in out.cases
                )
                panels[sp.name] = out.panels
                datasets[sp.name] = {
                    "paths": {
                        str(p.relative_to(project.root)): sha256_file(p) for p in out.data_paths
                    },
                    "cases": len(sp.cases),
                    "expanded_cases": len(out.cases),
                }
    finally:
        set_runtime(None)
        rt.close()
    summaries = {
        sp.name: summarize_suite(
            sp.name, [r for r in all_rows if r.suite == sp.name], sp.metric_ids
        )
        for sp in plan.suites
    }
    gate = evaluate(project.thresholds, summaries, all_rows, baseline.rows if baseline else None)
    info = build_info(
        project, plan, rt, judge, target, run_id, started, sha, dirty, datasets, baseline, gate
    )
    store.write_rows(all_rows)
    store.write_observations(obs_rows)
    store.write_cases(snapshot)
    store.write_panels(panels)
    store.write_info(info)
    return RunOutcome(run_id, run_dir, info, all_rows, summaries, gate, panels)


def build_info(
    project: Project,
    plan: RunPlan,
    rt: LLMRuntime,
    judge: JudgeRunner,
    target: Any,
    run_id: str,
    started: str,
    sha: str | None,
    dirty: bool | None,
    datasets: dict[str, dict[str, Any]],
    baseline: StoredRun | None,
    gate: GateResult,
) -> RunInfo:
    ledger = rt.ledger.summary()
    prompts = {
        str(p.relative_to(project.root)): sha256_file(p)
        for p in sorted(project.path("prompts").glob("*"))
        if p.is_file()
    }
    return RunInfo(
        run_id=run_id,
        started=started,
        finished=datetime.now().astimezone().isoformat(timespec="seconds"),
        git_commit=sha,
        git_dirty=dirty,
        profile=project.profile,
        cli_overrides=project.overrides.as_dict(),
        target=str(getattr(target, "name", "")),
        target_config_hash=getattr(target, "config_hash", None),
        roles=rt.roles_summary(),
        ledger=ledger,
        prompt_hashes=prompts,
        datasets=datasets,
        judges=judge.used,
        suites={sp.name: sp.metric_ids for sp in plan.suites},
        repeats=project.config.run.repeats,
        seed=project.config.run.seed,
        evalkit_version=__version__,
        engines=engine_versions(project, plan),
        total_tokens=rt.ledger.total_tokens(),
        total_cost_usd=round(rt.ledger.paid_spend(), 6),
        paid_cost_usd=round(rt.ledger.paid_spend(), 6),
        baseline=baseline.run_id if baseline else None,
        gate=gate.as_dict(),
    )


def engine_versions(project: Project, plan: RunPlan) -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    used = {
        e
        for m, e in project.config.engines.items()
        if e != "native" and any(m in sp.metric_ids for sp in plan.suites)
    }
    out = {}
    for eng in sorted(used):
        try:
            out[eng] = version(eng)
        except PackageNotFoundError:
            out[eng] = "not installed"
    return out
