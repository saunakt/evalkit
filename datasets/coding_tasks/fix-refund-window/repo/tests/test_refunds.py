from datetime import date

from refunds import refund_window_days


def test_day_ten_allowed():
    assert refund_window_days(date(2026, 1, 1), date(2026, 1, 11))


def test_day_forty_five_rejected():
    assert not refund_window_days(date(2026, 1, 1), date(2026, 2, 15))
