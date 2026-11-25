from orders import parse_order_id


def test_normalises_case():
    assert parse_order_id(" ord-100001 ") == "ORD-100001"
