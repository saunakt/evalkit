"""Unit tests for agent metrics (fixtures + the ShopEase target where hooks are needed)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from evalkit.metrics.agent import deep_subset, is_subsequence
from evalkit.schema import Case, MetricResult, Observation
from evalkit.targets.base import load_target
from tests.metric_helpers import ROOT, case_run, make_ctx, mr, run_case, run_dataset


def C(**kw: object) -> Case:
    return Case.model_validate({"id": "c1", **kw})


def O(*calls: str, text: str = "ok", **kw: object) -> Observation:
    tc = [
        {
            "name": c.split(":")[0],
            "args": {"order_id": c.split(":")[1]} if ":" in c else {},
            "agent": "orders_agent" if c.startswith(("get", "cancel")) else "policy_agent",
        }
        for c in calls
    ]
    return Observation.model_validate({"text": text, "tool_calls": tc, **kw})


def shopease() -> object:
    return load_target(ROOT / "targets" / "shopease_agent.yaml", ROOT)


TRAJ = C(
    expect_trajectory=[
        {"name": "get_order", "args": {"order_id": "ORD-1"}},
        {"name": "cancel_order"},
    ]
)


@pytest.mark.parametrize(
    ("mode", "calls", "ok"),
    [
        ("exact", ["get_order", "cancel_order"], True),
        ("exact", ["get_order", "search_policy", "cancel_order"], False),
        ("in_order", ["get_order", "search_policy", "cancel_order"], True),
        ("in_order", ["cancel_order", "get_order"], False),
        ("any_order", ["cancel_order", "get_order"], True),
        ("subset", ["get_order", "cancel_order", "get_order"], True),
        ("subset", ["get_order"], False),
    ],
)
def test_trajectory_modes(tmp_path: Path, mode: str, calls: list[str], ok: bool) -> None:
    r = run_case(f"trajectory_{mode}", TRAJ, O(*calls), make_ctx(tmp_path, suite="agent"))
    assert r.passed is ok


def test_trajectory_uses_case_mode(tmp_path: Path) -> None:
    case = TRAJ.model_copy(update={"trajectory_mode": "any_order"})
    assert run_case("trajectory", case, O("cancel_order", "get_order"), make_ctx(tmp_path)).passed


def test_selection_args_forbidden_efficiency(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path, suite="agent")
    case = C(
        expect_tools=["get_order"],
        forbid_tools=["cancel_order"],
        expect_trajectory=[{"name": "get_order", "args": {"order_id": "ord-1"}}],
    )
    good = O("get_order:ORD-1")
    assert run_case("tool_selection", case, good, ctx).passed
    bad = run_case("tool_selection", case, O("get_order:ORD-1", "cancel_order"), ctx)
    assert bad.passed is False and bad.details["forbidden"] == ["cancel_order"]
    assert run_case("argument_accuracy", case, good, ctx).value == 1.0
    assert run_case("argument_accuracy", case, O("get_order:ORD-2"), ctx).passed is False
    assert run_case("forbidden_tools", case, O("cancel_order"), ctx).passed is False
    eff = run_case("step_efficiency", case, O("get_order", "get_order", "get_order"), ctx)
    assert eff.value == pytest.approx(1 / 3) and eff.passed is False


def test_end_state_and_task_success(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path, suite="agent")
    case = C(expect_state={"orders": {"A": {"status": "processing"}}}, criteria=["polite"])
    ok = O(state={"orders": {"A": {"status": "processing"}, "B": {}}})
    assert run_case("end_state", case, ok, ctx).passed is True
    assert deep_subset({"a": {"b": 1}}, {"a": {"b": 2}}) == ["a.b: expected 1, got 2"]
    assert is_subsequence(["a", "c"], ["a", "b", "c"])
    ts = run_case("task_success", case, ok, ctx)
    assert ts.passed is True and ts.details["parts"]["end_state"] is True
    bad = run_case("task_success", case, O(state={"orders": {"A": {"status": "x"}}}), ctx)
    assert bad.passed is False
    err = run_case("task_success", C(), O(error="boom"), ctx)
    assert err.passed is False


def test_policy_rules(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path, {"confirm_regex": r"(?i)please cancel"}, suite="agent")
    rules = [
        {"before": ["get_order", "cancel_order"]},
        {"requires_confirmation": "cancel_order"},
        {"never": "delete_account"},
        {"max_calls": {"tool": "cancel_order", "n": 1}},
    ]
    case = C(input="Please cancel ORD-1", policies=rules)
    assert run_case("policy_compliance", case, O("get_order", "cancel_order"), ctx).passed
    bad = run_case(
        "policy_compliance",
        C(input="cancel it", policies=rules),
        O("cancel_order", "cancel_order", "delete_account"),
        ctx,
    )
    assert len(bad.details["violations"]) == 4
    amount = C(policies=[{"max_arg": {"tool": "refund", "arg": "amount", "max": 500}}])
    over = Observation(tool_calls=[{"name": "refund", "args": {"amount": 900}}])
    assert run_case("policy_compliance", amount, over, ctx).passed is False


def test_confirmation_compliance(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path, {"confirm_tools": ["cancel_order"]}, suite="agent")
    assert run_case(
        "confirmation_compliance", C(input="Please cancel ORD-1"), O("cancel_order"), ctx
    ).passed
    assert (
        run_case("confirmation_compliance", C(input="cancel ORD-1"), O("cancel_order"), ctx).passed
        is False
    )
    assert run_case(
        "confirmation_compliance", C(input="x"), O("get_order"), ctx
    ).skipped.startswith("n/a")


def _runs(results: list[list[bool]]) -> list[object]:
    out = []
    for i, reps in enumerate(results):
        res: list[MetricResult] = [mr("task_success", p) for p in reps]
        out.append(case_run(C(id=f"c{i}", tags=["safety"]), ["x"] * len(reps), res))
    return out


def test_reliability_and_dataset_metrics(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path, {"k": 3}, suite="agent")
    rows = _runs([[True, True, True], [True, False, True]])
    assert run_dataset("pass_at_k", rows, ctx).value == 1.0
    hat = run_dataset("pass_hat_k", rows, ctx)
    assert hat.value == 0.5 and hat.details["failing"] == ["c1"]
    assert hat.details["estimate"] == pytest.approx((1 + (2 / 3) ** 3) / 2)
    unsafe = [case_run(C(id="s", tags=["safety"]), ["x"], [mr("forbidden_tools", False)])]
    assert run_dataset("unsafe_action_rate", unsafe, ctx).value == 1.0
    assert run_dataset("time_per_task", rows, ctx).passed is None
    cost_rows = [
        case_run(
            C(id="a"), observations=[Observation(cost_usd=0.02)], results=[mr("task_success", True)]
        )
    ]
    assert run_dataset("cost_per_success", cost_rows, ctx).value == pytest.approx(0.02)


def test_tool_unit_and_mcp(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path, {"tool_tests": "tests/demo_unit/test_guard.py"}, suite="agent")
    ctx.suite.project.root = ROOT
    assert run_dataset("tool_unit", [], ctx).passed is True
    mcp = run_dataset("mcp_contract", [], make_ctx(tmp_path, {"mcp_server": "x"}))
    if mcp.skipped is None:  # mcp installed: a server that cannot start is a contract failure
        assert mcp.passed is False and "cannot start MCP server" in mcp.details["problems"][0]
    else:
        assert mcp.skipped == "mcp not installed"


def test_mcp_contract_against_real_server(tmp_path: Path) -> None:
    pytest.importorskip("mcp")
    server = {
        "command": sys.executable,
        "args": [str(ROOT / "tests" / "fixtures" / "mcp_shop_server.py")],
    }
    calls = [
        {"name": "get_order", "args": {"order_id": "ORD-100001"}},
        {"name": "get_order", "args": {"order_id": "ORD-999"}, "expect_error": True},
        {"name": "ping"},
    ]
    from evalkit.engines.mcp_engine import check_server

    report = check_server(server, calls)
    assert report["tools"] == ["get_order", "ping"]
    assert report["problems"] == [] and report["score"] == 1.0
    wrong = check_server(server, [{"name": "ping", "expect_error": True}])
    assert wrong["problems"] and wrong["score"] < 1.0


def test_handoff_hitl_resume_single_step_on_shopease(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path, {"approval_tools": ["cancel_order"]}, target=shopease(), suite="agent")
    case = C(
        input="Please cancel ORD-100004.",
        expect_trajectory=[
            {"name": "get_order", "agent": "orders_agent", "args": {"order_id": "ORD-100004"}},
            {"name": "cancel_order", "agent": "orders_agent"},
        ],
        expect_state={"orders": {"ORD-100004": {"status": "cancelled"}}},
        extra={"interrupt_after_steps": 1},
    )
    obs = ctx.suite.recall(case)
    assert run_case("handoff_accuracy", case, obs, ctx).passed is True
    assert run_case("single_step_choice", case, obs, ctx).passed is True
    assert run_case("resume_after_interrupt", case, obs, ctx).passed is True
    deny = case.model_copy(update={"extra": {"approval": "deny"}})
    h = run_case("hitl_approval", deny, obs, ctx)
    assert h.passed is True and h.details["executed"] == []
    assert run_case("hitl_approval", case, obs, ctx).passed is True


def test_error_recovery_and_plan_quality(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path, suite="agent")
    case = C(
        input="Please cancel ORD-1",
        faults=[{"tool": "get_order", "error": "timeout"}],
        expect_state={"orders": {"A": {"status": "processing"}}},
    )
    calm = O(
        "get_order",
        text="I couldn't look it up; nothing changed.",
        state={"orders": {"A": {"status": "processing"}}},
    )
    assert run_case("error_recovery", case, calm, ctx).passed is True
    crash = O(text="Traceback (most recent call last): ...", state=calm.state)
    assert run_case("error_recovery", case, crash, ctx).passed is False
    assert run_case("plan_quality", case, calm, ctx).value == 1.0
    assert run_case("plan_quality", case, O(), ctx).skipped == "n/a: no tool calls"


def test_simulated_and_browser(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path, suite="agent")
    sim = C(persona="p", hidden_goal="g")
    r = run_case("simulated_user_success", sim, O(raw={"user_turns": 2}), ctx)
    assert r.passed is True and r.details["user_turns"] == 2
    assert run_case("browser_task_success", C(), O(), ctx).skipped == "n/a: not a browser task"
    assert run_case("browser_task_success", C(tags=["browser"]), O(), ctx).passed is True
