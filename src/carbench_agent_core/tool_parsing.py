"""Helpers for tool-call arguments and tool-result payloads."""

from __future__ import annotations

import json
from typing import Any

from .tool_index import ToolIndex


def _tool_result_name(tool_result: dict[str, Any]) -> str:
    return (
        tool_result.get("tool_name")
        or tool_result.get("toolName")
        or tool_result.get("name")
        or ""
    )


def _parse_tool_call_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _parse_tool_result_content(tool_result: dict[str, Any]) -> dict[str, Any]:
    content = tool_result.get("content", "")
    if isinstance(content, dict):
        return content
    if not isinstance(content, str):
        return {}
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _tool_argument_available(
    tool_index: ToolIndex, tool_name: str, argument_name: str
) -> bool:
    return argument_name in tool_index.arg_names(tool_name)
