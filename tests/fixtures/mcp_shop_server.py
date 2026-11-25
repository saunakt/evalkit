"""A tiny real MCP server (stdio) for testing `mcp_contract` against the MCP SDK (mcp 2.x)."""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

server = MCPServer("shop-test")
ORDERS = {"ORD-100001": "processing", "ORD-100002": "delivered"}


@server.tool()
def get_order(order_id: str) -> dict[str, str]:
    """Look up an order's status."""
    if order_id not in ORDERS:
        raise ValueError(f"order {order_id} not found")
    return {"order_id": order_id, "status": ORDERS[order_id]}


@server.tool()
def ping() -> str:
    """Health check."""
    return "pong"


if __name__ == "__main__":
    server.run("stdio")
