"""Order id helpers."""


def parse_order_id(text: str) -> str:
    """Return the normalised order id."""
    return text.strip().upper()
