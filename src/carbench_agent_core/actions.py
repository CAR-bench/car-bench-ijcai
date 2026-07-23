"""Benchmark-visible controller actions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class NextAction:
    """One benchmark-visible assistant action."""

    action: Literal["respond", "tool_calls"]
    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    reason: str = "policy"

    @classmethod
    def respond(cls, content: str, *, reason: str = "policy") -> "NextAction":
        return cls(action="respond", content=content, reason=reason)

    @classmethod
    def tool_call(
        cls,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        *,
        reason: str = "policy",
    ) -> "NextAction":
        return cls(
            action="tool_calls",
            content="",
            tool_calls=[{"tool_name": tool_name, "arguments": arguments or {}}],
            reason=reason,
        )
