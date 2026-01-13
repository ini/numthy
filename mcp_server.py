#!/usr/bin/env python3
"""
MCP Server for numthy - exposes number theory functions to Claude and other AI assistants.

Usage:
    uvx mcp-server-numthy
    # or
    python mcp_server.py
"""

import ast
import inspect
import json
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

import numthy as nt

server = Server("numthy")


def serialize_result(result: Any) -> str:
    """Convert numthy results to JSON-serializable format."""
    if result is None:
        return "None"

    # Handle booleans first (before other checks)
    if isinstance(result, bool):
        return "true" if result else "false"

    # Handle generators/iterators - take first 100 items
    if hasattr(result, '__next__'):
        items = []
        for i, item in enumerate(result):
            if i >= 100:
                items.append("... (truncated)")
                break
            items.append(item)
        result = items

    # Handle tuples, convert to lists for JSON
    if isinstance(result, tuple):
        result = list(result)

    # Handle Fraction
    if hasattr(result, 'numerator') and hasattr(result, 'denominator'):
        return f"{result.numerator}/{result.denominator}"

    # Handle complex
    if isinstance(result, complex):
        return f"{result.real}+{result.imag}j" if result.imag >= 0 else f"{result.real}{result.imag}j"

    # Handle callables (like dirichlet_character returns)
    if callable(result):
        return f"<function {getattr(result, '__name__', 'anonymous')}>"

    try:
        return json.dumps(result)
    except (TypeError, ValueError):
        return str(result)


def get_function_description(func) -> str:
    """Extract first line of docstring as description."""
    doc = inspect.getdoc(func)
    if doc:
        return doc.split('\n')[0]
    return f"Call {func.__name__}"


def get_function_params(func) -> dict:
    """Extract parameter info from function signature and docstring."""
    sig = inspect.signature(func)
    hints = getattr(func, '__annotations__', {})

    properties = {}
    required = []

    for name, param in sig.parameters.items():
        if name in ('self', 'cls'):
            continue

        prop = {"type": "string"}  # Default to string, will be parsed

        # Try to get type hint
        hint = hints.get(name)
        if hint:
            hint_str = str(hint)
            if 'int' in hint_str:
                prop["type"] = "integer"
            elif 'float' in hint_str:
                prop["type"] = "number"
            elif 'bool' in hint_str:
                prop["type"] = "boolean"
            elif 'str' in hint_str:
                prop["type"] = "string"

        # Add description from docstring if available
        doc = inspect.getdoc(func)
        if doc and name in doc:
            # Try to extract parameter description
            for line in doc.split('\n'):
                if line.strip().startswith(name):
                    parts = line.split(':', 1)
                    if len(parts) > 1:
                        prop["description"] = parts[1].strip()
                    break

        properties[name] = prop

        # Check if required (no default value)
        if param.default is inspect.Parameter.empty and param.kind not in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD
        ):
            required.append(name)

    return {"type": "object", "properties": properties, "required": required}


# Build tools from numthy.__all__
TOOLS: dict[str, tuple[Tool, callable]] = {}

for name in nt.__all__:
    obj = getattr(nt, name, None)

    # Skip non-callables and type aliases
    if not callable(obj) or name in ('Number', 'Vector', 'Matrix'):
        continue

    tool = Tool(
        name=name,
        description=get_function_description(obj),
        inputSchema=get_function_params(obj)
    )
    TOOLS[name] = (tool, obj)


@server.list_tools()
async def list_tools() -> list[Tool]:
    """Return list of available numthy tools."""
    return [tool for tool, _ in TOOLS.values()]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Execute a numthy function."""
    if name not in TOOLS:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]

    _, func = TOOLS[name]

    try:
        # Convert string arguments to appropriate types based on annotations
        hints = getattr(func, '__annotations__', {})
        converted_args = {}

        for key, value in arguments.items():
            hint = hints.get(key)
            if hint:
                hint_str = str(hint)
                if 'int' in hint_str and isinstance(value, str):
                    # Handle expressions like "2**64 + 1"
                    try:
                        converted_args[key] = int(value)
                    except ValueError:
                        # Try evaluating as expression (safe subset)
                        converted_args[key] = eval(value, {"__builtins__": {}}, {})
                elif 'int' in hint_str:
                    converted_args[key] = int(value)
                elif 'float' in hint_str:
                    converted_args[key] = float(value)
                elif 'bool' in hint_str:
                    converted_args[key] = bool(value)
                else:
                    converted_args[key] = value
            else:
                converted_args[key] = value

        result = func(**converted_args)
        serialized = serialize_result(result)

        return [TextContent(type="text", text=serialized)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error: {type(e).__name__}: {e}")]


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
