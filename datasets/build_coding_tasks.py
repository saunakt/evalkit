"""Writes the two demo coding tasks under datasets/coding_tasks/ (output is committed)."""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).parent / "coding_tasks"

FILES: dict[str, str] = {
    # ---------------------------------------------------------------- task 1
    "fix-refund-window/task.yaml": """prompt: >
  refund_window_days(delivered_on, today) in refunds.py should return True when a return is
  still allowed: on or before day 30 after delivery. Customers on day 30 are being rejected.
  Fix the off-by-one without changing other behaviour.
repo: repo
hidden_tests: hidden_tests
existing_tests: tests
reference_files: [refunds.py]
rule_checks:
  - "{python} -c \\"import ast; ast.parse(open('refunds.py').read())\\""
  - "{python} -c \\"import refunds; assert refunds.refund_window_days.__doc__\\""
tags: [bugfix]
""",
    "fix-refund-window/repo/refunds.py": '''"""Refund window rules."""

from datetime import date

WINDOW_DAYS = 30


def refund_window_days(delivered_on: date, today: date) -> bool:
    """True if a return is still allowed (within WINDOW_DAYS of delivery)."""
    return (today - delivered_on).days < WINDOW_DAYS
''',
    "fix-refund-window/repo/tests/test_refunds.py": """from datetime import date

from refunds import refund_window_days


def test_day_ten_allowed():
    assert refund_window_days(date(2026, 1, 1), date(2026, 1, 11))


def test_day_forty_five_rejected():
    assert not refund_window_days(date(2026, 1, 1), date(2026, 2, 15))
""",
    "fix-refund-window/hidden_tests/test_hidden_refunds.py": """from datetime import date

from refunds import refund_window_days


def test_day_thirty_allowed():
    assert refund_window_days(date(2026, 1, 1), date(2026, 1, 31))


def test_day_thirty_one_rejected():
    assert not refund_window_days(date(2026, 1, 1), date(2026, 2, 1))
""",
    # ---------------------------------------------------------------- task 2
    "validate-order-id/task.yaml": """prompt: >
  parse_order_id(text) in orders.py must reject anything that is not an order id of the form
  ORD- followed by exactly six digits (case-insensitive) by raising ValueError, and return the
  normalised upper-case id otherwise.
repo: repo
hidden_tests: hidden_tests
existing_tests: tests
reference_files: [orders.py]
rule_checks:
  - "{python} -c \\"import ast; ast.parse(open('orders.py').read())\\""
tags: [validation]
""",
    "validate-order-id/repo/orders.py": '''"""Order id helpers."""


def parse_order_id(text: str) -> str:
    """Return the normalised order id."""
    return text.strip().upper()
''',
    "validate-order-id/repo/tests/test_orders.py": """from orders import parse_order_id


def test_normalises_case():
    assert parse_order_id(" ord-100001 ") == "ORD-100001"
""",
    "validate-order-id/hidden_tests/test_hidden_orders.py": """import pytest

from orders import parse_order_id


def test_valid_id():
    assert parse_order_id("ORD-123456") == "ORD-123456"


@pytest.mark.parametrize("bad", ["ORD-12345", "ORD-1234567", "ABC-123456", "", "ORD-12a456"])
def test_invalid_ids_raise(bad):
    with pytest.raises(ValueError):
        parse_order_id(bad)
""",
}


def main() -> None:
    for rel, text in FILES.items():
        path = HERE / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
