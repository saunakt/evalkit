"""Fill the refund letter template (demo skill script; no network, no env access)."""

import sys

TEMPLATE = "Dear customer, we have processed the refund for {order}."

if __name__ == "__main__":
    print(TEMPLATE.format(order=sys.argv[1] if len(sys.argv) > 1 else "your order"))
