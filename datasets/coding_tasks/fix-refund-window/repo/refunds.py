"""Refund window rules."""

from datetime import date

WINDOW_DAYS = 30


def refund_window_days(delivered_on: date, today: date) -> bool:
    """True if a return is still allowed (within WINDOW_DAYS of delivery)."""
    return (today - delivered_on).days < WINDOW_DAYS
