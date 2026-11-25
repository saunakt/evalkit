"""MCP contract checks (extra `[agent]`, MCP SDK imported lazily). Checked against mcp 2.3.0:
`mcp.client.stdio.stdio_client`, `mcp.client.streamable_http.streamable_http_client`,
`ClientSession.list_tools()` (Tool.input_schema) and `call_tool()` (CallToolResult.is_error)."""

from __future__ import annotations

import asyncio
import shlex
from typing import Any

import jsonschema

from evalkit.registry import engine
from evalkit.util import optional_import


@engine("mcp", requires_extra="agent")
class McpEngine:
    """Engine marker: provides `mcp_contract` via check_server()."""

    package = "mcp"


async def _session_checks(session: Any, calls: list[dict[str, Any]]) -> dict[str, Any]:
    await session.initialize()
    tools = (await session.list_tools()).tools
    problems: list[str] = []
    for tool in tools:
        try:
            jsonschema.Draft202012Validator.check_schema(tool.input_schema or {})
        except jsonschema.SchemaError as exc:
            problems.append(f"{tool.name}: invalid input schema: {exc.message}")
    names = {t.name for t in tools}
    for call in calls:
        res = await session.call_tool(call["name"], call.get("args", {}))
        want_error = bool(call.get("expect_error", False))
        if bool(getattr(res, "is_error", False)) != want_error:
            problems.append(
                f"{call['name']}: is_error={getattr(res, 'is_error', None)}, expected {want_error}"
            )
    try:
        res = await session.call_tool("__evalkit_unknown_tool__", {})
        if not getattr(res, "is_error", False):
            problems.append("unknown tool call did not return an error")
    except Exception:  # an MCP error response is the expected outcome
        pass
    if tools:
        try:
            res = await session.call_tool(tools[0].name, {"__bad_arg__": object.__name__})
        except Exception:  # rejected bad args: handled
            res = None
        if res is not None and not getattr(res, "is_error", False) and tools[0].input_schema:
            problems.append(f"{tools[0].name}: bad arguments were not rejected")
    checks = len(tools) + len(calls) + 2
    return {
        "tools": sorted(names),
        "problems": problems,
        "score": 1 - len(problems) / max(1, checks),
    }


async def _run(server: str | dict[str, Any], calls: list[dict[str, Any]]) -> dict[str, Any]:
    mcp = optional_import("mcp")
    if isinstance(server, str) and server.startswith(("http://", "https://")):
        http = optional_import("mcp.client.streamable_http", "mcp")
        async with (
            http.streamable_http_client(server) as streams,
            mcp.ClientSession(streams[0], streams[1]) as session,
        ):
            return await _session_checks(session, calls)
    stdio = optional_import("mcp.client.stdio", "mcp")
    cmd = (
        shlex.split(server)
        if isinstance(server, str)
        else [server["command"], *server.get("args", [])]
    )
    params = stdio.StdioServerParameters(command=cmd[0], args=cmd[1:])
    async with (
        stdio.stdio_client(params) as (read, write),
        mcp.ClientSession(read, write) as session,
    ):
        return await _session_checks(session, calls)


def check_server(server: str | dict[str, Any], calls: list[dict[str, Any]]) -> dict[str, Any]:
    """Connect (stdio command or URL), list tools, validate schemas, run calls and probes."""
    try:
        return asyncio.run(_run(server, calls))
    except OSError as exc:  # the stdio server command could not be started
        return {
            "tools": [],
            "problems": [f"cannot start MCP server {server!r}: {exc}"],
            "score": 0.0,
        }
