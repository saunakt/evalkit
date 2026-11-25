"""ShopEase tools over an in-memory sandbox DB. The cancel business rule lives here."""

from __future__ import annotations

import copy
import threading
from collections.abc import Callable
from typing import Any

from evalkit.env.faults import FaultCounter, ToolFault, apply_tool_fault, find_fault
from examples.shopease import kb

SEED_ORDERS: dict[str, dict[str, Any]] = {
    "ORD-100001": {
        "status": "processing",
        "item": "Running shoes",
        "amount": 2499,
        "customer": "C-1",
        "tenant": "shopease",
    },
    "ORD-100002": {
        "status": "delivered",
        "item": "Bluetooth speaker",
        "amount": 1799,
        "customer": "C-1",
        "tenant": "shopease",
        "delivered_on": "2026-11-10",
    },
    "ORD-100003": {
        "status": "shipped",
        "item": "Cotton kurta",
        "amount": 899,
        "customer": "C-2",
        "tenant": "shopease",
    },
    "ORD-100004": {
        "status": "processing",
        "item": "Smartwatch",
        "amount": 12500,
        "customer": "C-3",
        "tenant": "shopease",
    },
}

_local = threading.local()


def db() -> dict[str, dict[str, Any]]:
    if not hasattr(_local, "orders"):
        reset(42)
    value: dict[str, dict[str, Any]] = _local.orders
    return value


def reset(seed: int) -> None:
    """Fresh sandbox state for this worker thread (orders + knowledge base)."""
    _local.orders = copy.deepcopy(SEED_ORDERS)
    _local.fault_counter = FaultCounter()
    _local.cancellations = []
    kb.reset_kb()


def state() -> dict[str, Any]:
    return {
        "orders": {oid: {"status": o["status"]} for oid, o in sorted(db().items())},
        "cancellations": list(getattr(_local, "cancellations", [])),
    }


class ToolError(Exception):
    pass


def search_policy(query: str, k: int = 5) -> list[dict[str, Any]]:
    return kb.chunks_for(kb.kb().search(query, k=k))


def get_order(order_id: str) -> dict[str, Any]:
    order = db().get(order_id.strip().upper())
    if order is None:
        raise ToolError(f"order {order_id} not found")
    return {"order_id": order_id.upper(), **{k: v for k, v in order.items() if k != "tenant"}}


def cancel_order(order_id: str, reason: str = "customer request") -> dict[str, Any]:
    order = db().get(order_id.strip().upper())
    if order is None:
        raise ToolError(f"order {order_id} not found")
    if order["status"] != "processing" and not kb.buggy():
        raise ToolError(f"order {order_id} is {order['status']} and cannot be cancelled")
    order["status"] = "cancelled"
    _local.cancellations.append({"order_id": order_id.upper(), "reason": reason})
    return {"order_id": order_id.upper(), "status": "cancelled"}


TOOLS: dict[str, Callable[..., Any]] = {
    "search_policy": search_policy,
    "get_order": get_order,
    "cancel_order": cancel_order,
}


def call_tool(
    name: str, args: dict[str, Any], faults: list[dict[str, Any]]
) -> tuple[Any, str | None]:
    """Run a tool honouring fault specs; returns (result, error)."""
    fn = TOOLS.get(name)
    if fn is None:
        return None, f"unknown tool {name}"
    if not hasattr(_local, "fault_counter"):
        reset(42)
    wrapped = apply_tool_fault(
        find_fault(faults, tool=name), fn, _local.fault_counter, sleep=lambda s: None
    )
    try:
        return wrapped(**args), None
    except ToolFault as exc:
        return None, f"{exc.kind}: tool unavailable"
    except (ToolError, TypeError) as exc:
        return None, str(exc)


TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "search_policy": {
        "type": "object",
        "required": ["query"],
        "properties": {"query": {"type": "string"}, "k": {"type": "integer"}},
    },
    "get_order": {
        "type": "object",
        "required": ["order_id"],
        "properties": {"order_id": {"type": "string", "pattern": r"^ORD-\d{6}$"}},
    },
    "cancel_order": {
        "type": "object",
        "required": ["order_id"],
        "properties": {"order_id": {"type": "string"}, "reason": {"type": "string"}},
    },
}


def advertised_schemas() -> dict[str, dict[str, Any]]:
    """What the agent advertises to its model (checked by the classic `contract` metric)."""
    return dict(TOOL_SCHEMAS)
