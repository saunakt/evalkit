"""ShopEase tool unit tests (run by the classic `unit_fake_llm` and agent `tool_unit` metrics)."""

from __future__ import annotations

import pytest

from examples.shopease import tools


@pytest.fixture(autouse=True)
def fresh() -> None:
    tools.reset(42)


def test_get_order_returns_status() -> None:
    assert tools.get_order("ORD-100003")["status"] == "shipped"


def test_unknown_order_raises() -> None:
    with pytest.raises(tools.ToolError):
        tools.get_order("ORD-999999")


def test_cancel_processing_order() -> None:
    assert tools.cancel_order("ORD-100001")["status"] == "cancelled"
    assert tools.state()["orders"]["ORD-100001"]["status"] == "cancelled"


def test_cannot_cancel_delivered_order() -> None:
    with pytest.raises(tools.ToolError):
        tools.cancel_order("ORD-100002")
    assert tools.state()["orders"]["ORD-100002"]["status"] == "delivered"


def test_search_policy_finds_returns() -> None:
    ids = [c["id"] for c in tools.search_policy("refund for shoes delivered 10 days ago")]
    assert "returns" in ids


def test_tool_fault_is_reported_not_raised() -> None:
    result, error = tools.call_tool(
        "get_order", {"order_id": "ORD-100001"}, [{"tool": "get_order", "error": "timeout"}]
    )
    assert result is None and error is not None and "timeout" in error
