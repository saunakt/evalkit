from __future__ import annotations

import os
from pathlib import Path

import pytest

from evalkit.env import perturb
from evalkit.env.clock import exported_now, frozen
from evalkit.env.faults import FaultCounter, ToolFault, apply_tool_fault, find_fault
from evalkit.env.sandbox import Sandbox
from evalkit.env.simulator import simulate
from evalkit.llm.client import CostLedger, RoleClient
from evalkit.llm.fake import FakeLLM
from evalkit.schema import Observation


def rc(fake: FakeLLM, role: str) -> RoleClient:
    return RoleClient(
        role=role,
        provider_name="fake",
        client=fake,
        model="f",
        free=True,
        price=None,
        ledger=CostLedger(),
        cache=None,
        max_cost_usd=None,
    )


def test_perturbations_are_seeded_and_named() -> None:
    text = "Can I return my shoes after delivery please?"
    a = perturb.apply("typo", text, seed=1, rate=0.5)
    assert a == perturb.apply("typo", text, seed=1, rate=0.5)
    assert a[1] == "typo"
    assert perturb.apply("lowercase", "AbC", seed=0) == ("abc", "lowercase")
    assert perturb.apply("strip_punct", "Hi, there!", seed=0)[0] == "Hi there"
    assert perturb.apply("leetspeak", "test", seed=0)[0] == "7357"
    assert "aGk=" in perturb.apply("encode_base64", "hi", seed=0)[0]
    padded, _ = perturb.apply("pad_noise", "core", seed=3, n_words=4)
    assert "core" in padded and len(padded.split()) == 5


def test_llm_perturbation_uses_client() -> None:
    fake = FakeLLM(script={"rules": [{"match": "Paraphrase", "response": "reworded"}]})
    assert perturb.apply("paraphrase", "orig", seed=0, llm=rc(fake, "perturb")) == (
        "reworded",
        "paraphrase",
    )
    with pytest.raises(ValueError):
        perturb.apply("paraphrase", "orig", seed=0)
    assert (
        perturb.apply("language_switch:ta", "x", seed=0, llm=rc(fake, "perturb"))[1]
        == "language_switch:ta"
    )


def test_faults_fire_times_then_recover() -> None:
    faults = [{"tool": "get_order", "error": "timeout", "times": 1}]
    spec = find_fault(faults, tool="get_order")
    assert spec is not None and find_fault(faults, tool="other") is None
    fn = apply_tool_fault(spec, lambda: "ok", FaultCounter())
    with pytest.raises(ToolFault):
        fn()
    assert fn() == "ok"
    empty = apply_tool_fault(find_fault([{"tool": "t", "error": "empty"}], tool="t"), lambda: 1)
    assert empty() is None
    slept: list[float] = []
    slow = apply_tool_fault(
        find_fault([{"tool": "t", "error": "slow", "delay_s": 2}], tool="t"),
        lambda: 1,
        sleep=slept.append,
    )
    assert slow() == 1 and slept == [2]


def test_simulator_runs_until_done() -> None:
    fake = FakeLLM(
        script={
            "rules": [
                {"role": "simulator", "match": "order 1", "response": "<DONE>"},
                {"role": "simulator", "match": ".", "response": "cancel order 1"},
            ]
        }
    )
    sim = rc(fake, "simulator")
    res = simulate(
        persona="busy",
        hidden_goal="cancel",
        max_turns=5,
        llm=sim,
        call_target=lambda msgs: Observation(text="Cancelled order 1."),
    )
    assert res.done and res.user_turns == 1
    assert res.messages[0] == {"role": "user", "content": "cancel order 1"}


def test_simulator_respects_max_turns() -> None:
    fake = FakeLLM(script={"rules": [{"match": ".", "response": "more?"}]})
    res = simulate(
        persona="p",
        hidden_goal="g",
        max_turns=3,
        llm=rc(fake, "simulator"),
        call_target=lambda msgs: Observation(text="ok"),
    )
    assert not res.done and res.user_turns == 3


def test_sandbox_copies_and_cleans_up(tmp_path: Path) -> None:
    fixture = tmp_path / "repo"
    fixture.mkdir()
    (fixture / "a.py").write_text("x = 1\n", encoding="utf-8")
    with Sandbox(fixture) as sbx:
        (sbx.path / "a.py").write_text("x = 2\n", encoding="utf-8")
        (sbx.path / "b.py").write_text("y = 1\n", encoding="utf-8")
        assert sbx.changed_files() == ["a.py", "b.py"]
        assert sbx.run("echo hello").returncode == 0
        where = sbx.path
    assert not where.exists()
    assert (fixture / "a.py").read_text(encoding="utf-8") == "x = 1\n"
    assert sbx.log[0].command == "echo hello"


def test_clock_export_and_freeze() -> None:
    from datetime import datetime

    with exported_now("2026-11-25T10:00:00+05:30"):
        assert os.environ["EVALKIT_NOW"].startswith("2026-11-25")
        with frozen(os.environ["EVALKIT_NOW"]):
            assert datetime.now().year == 2026 and datetime.now().month == 11
    assert "EVALKIT_NOW" not in os.environ or os.environ["EVALKIT_NOW"] != ""
