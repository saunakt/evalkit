"""Builders for metric unit tests: a MetricContext over the fake provider and a fake target."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from evalkit.config import (
    CalibrationConfig,
    EvalkitConfig,
    Overrides,
    Project,
    ProviderConfig,
    RoleModel,
    SuiteConfig,
    Thresholds,
)
from evalkit.execution import run_case_metric, run_dataset_metric
from evalkit.judges.runner import JudgeRunner
from evalkit.llm.roles import LLMRuntime
from evalkit.registry import METRICS, load_all
from evalkit.schema import Case, CaseRun, MetricResult, Observation
from evalkit.suites.context import MetricContext, SuiteContext
from evalkit.targets.base import load_target

ROOT = Path(__file__).resolve().parents[1]


def make_ctx(
    tmp_path: Path,
    params: dict[str, Any] | None = None,
    script: dict[str, Any] | None = None,
    target: Any = None,
    suite: str = "quality",
    baseline: Any = None,
) -> MetricContext:
    load_all()
    (tmp_path / "fake.yaml").write_text(yaml.safe_dump(script or {}), encoding="utf-8")
    cfg = EvalkitConfig(
        providers={"fake": ProviderConfig(type="fake", script="fake.yaml")},
        models={
            r: RoleModel(provider="fake", model="fake")
            for r in ("judge", "simulator", "perturb", "generator", "sut")
        },
        calibration=CalibrationConfig(allow_uncalibrated=True),
    )
    project = Project(cfg, tmp_path, None, Overrides(), Thresholds(), {})
    rt = LLMRuntime(cfg, tmp_path)
    jr = JudgeRunner(rt, ROOT / "judges", cfg.calibration)
    tgt = target or load_target({"type": "fake"}, ROOT)
    sctx = SuiteContext(
        name=suite,
        project=project,
        sc=SuiteConfig(),
        params=params or {},
        target=tgt,
        runtime=rt,
        judge=jr,
        run_dir=tmp_path,
        seed=42,
        repeats=1,
        now=None,
        baseline=baseline,
    )
    return MetricContext(params or {}, jr, sctx)


def run_case(metric_id: str, case: Case, obs: Observation, ctx: MetricContext) -> MetricResult:
    ctx.metric_id = metric_id
    return run_case_metric(METRICS.get(metric_id), "native", case, obs, ctx)


def run_dataset(metric_id: str, rows: list[CaseRun], ctx: MetricContext) -> MetricResult:
    ctx.metric_id = metric_id
    return run_dataset_metric(METRICS.get(metric_id), "native", rows, ctx)[0]


def case_run(
    case: Case,
    texts: list[str] | None = None,
    results: list[MetricResult] | None = None,
    observations: list[Observation] | None = None,
) -> CaseRun:
    obs = observations or [Observation(text=t) for t in (texts or [""])]
    res = results or []
    return CaseRun(case, obs, res, [i % max(1, len(obs)) for i in range(len(res))])


def mr(metric: str, passed: bool | None, value: float | None = None) -> MetricResult:
    return MetricResult(
        metric=metric,
        passed=passed,
        value=value if value is not None else (None if passed is None else float(passed)),
    )
