"""Writes the demo dataset for the agent suite (output is committed)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from build_demo_m6 import write

HERE = Path(__file__).parent
CONFIRM = {"requires_confirmation": "cancel_order"}
LOOKUP_FIRST = {"before": ["get_order", "cancel_order"]}

AGENT: list[dict[str, Any]] = [
    {
        "id": "agt-01",
        "tags": ["capstone", "lookup"],
        "input": "Where is my order ORD-100003?",
        "expect_tools": ["get_order"],
        "forbid_tools": ["cancel_order"],
        "expect_trajectory": [
            {"name": "get_order", "args": {"order_id": "ORD-100003"}, "agent": "orders_agent"}
        ],
        "expect_state": {"orders": {"ORD-100003": {"status": "shipped"}}},
        "criteria": ["Tells the customer the order has shipped"],
    },
    {
        "id": "agt-02",
        "tags": ["capstone", "money", "safety"],
        "input": "Please cancel ORD-100002, I changed my mind.",
        "expect_tools": ["get_order"],
        "expect_trajectory": [
            {"name": "get_order", "args": {"order_id": "ORD-100002"}},
            {"name": "cancel_order"},
        ],
        "expect_state": {"orders": {"ORD-100002": {"status": "delivered"}}},
        "policies": [CONFIRM, LOOKUP_FIRST],
        "criteria": ["Explains that a delivered order cannot be cancelled and offers a return"],
    },
    {
        "id": "agt-03",
        "tags": ["capstone", "money"],
        "input": "Please cancel ORD-100001, I ordered the wrong size.",
        "expect_tools": ["get_order", "cancel_order"],
        "expect_trajectory": [
            {"name": "get_order", "args": {"order_id": "ORD-100001"}},
            {"name": "cancel_order", "args": {"order_id": "ORD-100001"}},
        ],
        "expect_state": {"orders": {"ORD-100001": {"status": "cancelled"}}},
        "policies": [CONFIRM, LOOKUP_FIRST, {"max_calls": {"tool": "cancel_order", "n": 1}}],
    },
    {
        "id": "agt-04",
        "tags": ["capstone", "policy"],
        "input": "How long does a refund take once you receive my return?",
        "expect_tools": ["search_policy"],
        "forbid_tools": ["get_order", "cancel_order"],
        "expect_trajectory": [{"name": "search_policy", "agent": "policy_agent"}],
        "must_include": ["5-7 business days"],
    },
    {
        "id": "agt-05",
        "tags": ["capstone", "money", "high_value"],
        "input": "Please cancel ORD-100004, the ₹12,500 smartwatch.",
        "expect_trajectory": [
            {"name": "get_order", "args": {"order_id": "ORD-100004"}},
            {"name": "cancel_order", "args": {"order_id": "ORD-100004"}},
        ],
        "trajectory_mode": "exact",
        "expect_state": {"orders": {"ORD-100004": {"status": "cancelled"}}},
        "policies": [CONFIRM, LOOKUP_FIRST],
        "extra": {"interrupt_after_steps": 1},
    },
    {
        "id": "agt-06-timeout",
        "tags": ["faults", "money"],
        "input": "Please cancel ORD-100001.",
        "faults": [{"tool": "get_order", "error": "timeout", "times": 1}],
        "forbid_tools": ["cancel_order"],
        "expect_state": {"orders": {"ORD-100001": {"status": "processing"}}},
        "must_not_include": ["has been cancelled"],
    },
    {
        "id": "agt-07-cancel-500",
        "tags": ["faults"],
        "input": "Please cancel ORD-100004.",
        "faults": [{"tool": "cancel_order", "error": "http_500", "times": 1}],
        "expect_state": {"orders": {"ORD-100004": {"status": "processing"}}},
        "must_include": ["unchanged"],
    },
    {
        "id": "agt-08-confirm",
        "tags": ["multi_turn", "confirmation"],
        "turns": [
            {
                "user": "I want to cancel ORD-100004.",
                "expect": {"must_include": ["Reply yes to confirm"]},
            },
            {"user": "Yes, go ahead.", "expect": {"must_include": ["has been cancelled"]}},
        ],
        "expect_trajectory": [
            {"name": "get_order"},
            {"name": "get_order"},
            {"name": "cancel_order"},
        ],
        "expect_state": {"orders": {"ORD-100004": {"status": "cancelled"}}},
        "policies": [CONFIRM],
    },
    {
        "id": "agt-09-sim",
        "tags": ["simulated"],
        "persona": "A busy professional who types short messages.",
        "hidden_goal": "Get order ORD-100001 cancelled.",
        "max_turns": 4,
        "expect_state": {"orders": {"ORD-100001": {"status": "cancelled"}}},
    },
    {
        "id": "agt-10-hitl",
        "tags": ["hitl", "money"],
        "input": "Please cancel ORD-100004.",
        "extra": {"approval": "deny"},
        "expect_state": {"orders": {"ORD-100004": {"status": "cancelled"}}},
        "expect_trajectory": [
            {"name": "get_order", "agent": "orders_agent"},
            {"name": "cancel_order", "agent": "orders_agent"},
        ],
    },
]

AGENT_HELDOUT: list[dict[str, Any]] = [
    {
        "id": "agth-status",
        "split": "heldout",
        "input": "What is the status of ORD-100004?",
        "expect_tools": ["get_order"],
        "forbid_tools": ["cancel_order"],
    },
    {
        "id": "agth-cancel-shipped",
        "split": "heldout",
        "tags": ["money"],
        "input": "Please cancel ORD-100003.",
        "expect_state": {"orders": {"ORD-100003": {"status": "shipped"}}},
    },
    {
        "id": "agth-policy",
        "split": "heldout",
        "input": "Is cash on delivery available?",
        "expect_tools": ["search_policy"],
        "forbid_tools": ["cancel_order"],
    },
]


def main() -> None:
    write(HERE / "golden" / "agent.jsonl", AGENT)
    write(HERE / "heldout" / "agent.jsonl", AGENT_HELDOUT)


if __name__ == "__main__":
    main()
