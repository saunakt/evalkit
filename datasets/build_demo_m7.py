"""Writes the demo datasets for rag and conversation (output is committed)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from build_demo_m6 import write

HERE = Path(__file__).parent

RAG: list[dict[str, Any]] = [
    {
        "id": "rag-refund",
        "tags": ["refund", "happy"],
        "input": "I want a refund for shoes delivered 10 days ago, is that allowed?",
        "relevant_ids": ["returns"],
        "must_include": ["30-day"],
        "reference": "Yes. Items can be returned within 30 days of delivery and are refunded to "
        "the original payment method.",
    },
    {
        "id": "rag-refund-timeline",
        "tags": ["refund"],
        "input": "How long does a refund take?",
        "relevant_ids": ["refund-timeline"],
        "must_include": ["5-7 business days"],
    },
    {
        "id": "rag-shipping",
        "tags": ["shipping"],
        "input": "Is shipping free above 499?",
        "relevant_ids": ["shipping"],
        "relevance": {"shipping": 3, "cod": 1},
        "must_include": ["free"],
    },
    {
        "id": "rag-cod",
        "tags": ["payments", "boundary"],
        "input": "Can I pay cash on delivery for a 12000 rupee order?",
        "relevant_ids": ["cod"],
        "must_include": ["₹10,000"],
    },
    {
        "id": "rag-unans-install",
        "tags": ["unanswerable"],
        "input": "Do you offer AC installation?",
        "must_include": ["human agent"],
    },
    {
        "id": "rag-unans-pricematch",
        "tags": ["unanswerable"],
        "input": "Do you price match other stores?",
        "must_not_include": ["we match"],
    },
    {
        "id": "rag-ingest-policies",
        "extra": {
            "stage": "ingestion",
            "expected_snippets": ["30-day window from delivery", "₹10,000"],
            "keep_together": ["5-7 business days"],
        },
    },
    {
        "id": "rag-ingest-privacy",
        "extra": {
            "stage": "ingestion",
            "expected_snippets": ["DPDP Act, 2023"],
            "keep_together": ["9 am to 9 pm IST"],
        },
    },
    {
        "id": "rag-fresh-cod",
        "tags": ["freshness"],
        "input": "What is the COD limit?",
        "relevant_ids": ["cod"],
        "extra": {
            "freshness": {
                "doc_id": "cod",
                "new_fact": "₹15,000",
                "old_fact": "₹10,000",
                "text": "Cash on delivery (COD) is available for orders up to "
                "₹15,000. Orders above ₹15,000 must be prepaid.",
            }
        },
    },
    {
        "id": "rag-access-goodwill",
        "tags": ["access_control"],
        "input": "How much goodwill credit can team leads approve?",
        "extra": {"role": "customer"},
        "must_not_include": ["₹500"],
    },
]

RAG_HELDOUT: list[dict[str, Any]] = [
    {
        "id": "ragh-exchange",
        "split": "heldout",
        "input": "Can I exchange a kurta for a different size?",
        "relevant_ids": ["exchanges"],
        "must_include": ["30-day return window"],
    },
    {
        "id": "ragh-gift",
        "split": "heldout",
        "input": "Do you sell gift cards?",
        "relevant_ids": ["gift-cards"],
        "must_include": ["does not sell gift cards"],
    },
    {
        "id": "ragh-hours",
        "split": "heldout",
        "input": "What are your support hours?",
        "relevant_ids": ["contact"],
        "must_include": ["9 am to 9 pm"],
    },
]

CONVERSATION: list[dict[str, Any]] = [
    {
        "id": "conv-cancel-ref",
        "tags": ["multi_turn", "reference"],
        "turns": [
            {
                "user": "Hi, where is my order ORD-100001?",
                "expect": {"must_include": ["processing"]},
            },
            {"user": "Please cancel it.", "expect": {"must_include": ["has been cancelled"]}},
        ],
        "criteria": ["The assistant cancels ORD-100001 when the user says 'it'"],
    },
    {
        "id": "conv-name",
        "tags": ["multi_turn", "memory"],
        "turns": [
            {"user": "My name is Priya.", "expect": {"must_include": ["Priya"]}},
            {"user": "Is shipping free above 499?", "expect": {"must_include": ["free"]}},
            {"user": "What's my name?", "expect": {"must_include": ["Priya"]}},
        ],
        "extra": {
            "memory": {
                "tell": "My name is Priya.",
                "ask": "What's my name?",
                "fact": "Priya",
                "customer_id": "C-77",
            }
        },
    },
    {
        "id": "conv-policy-followup",
        "tags": ["multi_turn"],
        "turns": [
            {
                "user": "Can I return shoes delivered 10 days ago?",
                "expect": {"must_include": ["30-day"]},
            },
            {
                "user": "And how long does the refund take after that?",
                "expect": {"must_include": ["5-7 business days"]},
            },
            {"user": "Thanks!"},
        ],
    },
    {
        "id": "conv-sim-cancel",
        "tags": ["simulated", "persona"],
        "persona": "A busy professional who types short messages and wants things done fast.",
        "hidden_goal": "Get order ORD-100001 cancelled.",
        "max_turns": 4,
    },
    {
        "id": "conv-sim-return",
        "tags": ["simulated", "persona"],
        "persona": "A first-time online shopper, polite but unsure about the rules.",
        "hidden_goal": "Find out how many days you have to return shoes after delivery.",
        "max_turns": 4,
    },
]


def main() -> None:
    write(HERE / "golden" / "rag.jsonl", RAG)
    write(HERE / "heldout" / "rag.jsonl", RAG_HELDOUT)
    write(HERE / "golden" / "conversation.jsonl", CONVERSATION)


if __name__ == "__main__":
    main()
