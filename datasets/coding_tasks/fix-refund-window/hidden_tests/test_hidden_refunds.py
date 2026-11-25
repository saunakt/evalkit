from datetime import date

from refunds import refund_window_days


def test_day_thirty_allowed():
    assert refund_window_days(date(2026, 1, 1), date(2026, 1, 31))


def test_day_thirty_one_rejected():
    assert not refund_window_days(date(2026, 1, 1), date(2026, 2, 1))
