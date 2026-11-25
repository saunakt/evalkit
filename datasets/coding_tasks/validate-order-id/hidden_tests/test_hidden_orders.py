import pytest

from orders import parse_order_id


def test_valid_id():
    assert parse_order_id("ORD-123456") == "ORD-123456"


@pytest.mark.parametrize("bad", ["ORD-12345", "ORD-1234567", "ABC-123456", "", "ORD-12a456"])
def test_invalid_ids_raise(bad):
    with pytest.raises(ValueError):
        parse_order_id(bad)
