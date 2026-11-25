"""DoD: only the judge role is paid by default; `--provider local` makes zero paid calls."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evalkit.config import Overrides, load_project
from evalkit.llm.roles import is_free
from evalkit.planner import build_plan
from evalkit.runner import run
from tests.conftest import ROOT, FakeServer, Reply, chat_reply

ENV = {"EVALKIT_JUDGE_MODEL": "judge-model-x", "EVALKIT_LOCAL_MODEL": "local-model-y"}


@pytest.fixture(autouse=True)
def model_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)


def test_default_config_only_judge_is_paid() -> None:
    project = load_project(ROOT / "evalkit.yaml", "nightly")
    cfg = project.config
    for role, rm in cfg.models.items():
        if rm is None:
            continue
        free = is_free(cfg.providers[rm.provider])
        assert free == (role != "judge"), role
    plan = build_plan(project, check_providers=False)
    assert plan.paid_roles == ["judge"]
    assert plan.calls["judge"] > 0 and plan.paid_calls == plan.calls["judge"]


def test_deepeval_metrics_count_under_judge() -> None:
    project = load_project(
        ROOT / "evalkit.yaml",
        None,
        Overrides(suites=["rag"], metrics=["faithfulness"], sets=["engines.faithfulness=deepeval"]),
    )
    plan = build_plan(project, check_providers=False)
    assert set(plan.calls) == {"judge"} and plan.paid_roles == ["judge"]


def test_run_with_local_provider_makes_zero_paid_calls(
    http_server: FakeServer, tmp_path: Path
) -> None:
    verdict = json.dumps({"reasoning": "fine", "verdict": "PASS"})
    http_server.default = Reply(
        body={
            **chat_reply(verdict, {"prompt_tokens": 40, "completion_tokens": 8}),
            "data": [{"id": "local-model-y"}],
        }
    )
    project = load_project(
        ROOT / "evalkit.yaml",
        None,
        Overrides(
            provider="local",
            suites=["quality"],
            repeats=1,
            sets=[
                f"providers.local.base_url={http_server.url}",
                f"results_dir={tmp_path}",
                "calibration.allow_uncalibrated=true",
                "run.cache=off",
                f"run.cache_path={tmp_path / 'cache.sqlite'}",  # judges always cache
            ],
        ),
    )
    plan = build_plan(project)  # reachability check hits the in-process server
    outcome = run(project, plan)
    ledger = outcome.info.ledger
    assert ledger["judge"]["calls"] > 0
    assert sum(v["paid_calls"] for v in ledger.values()) == 0
    assert outcome.info.paid_cost_usd == 0.0
    assert any(r["path"] == "/chat/completions" for r in http_server.requests)
