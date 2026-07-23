"""Shared scalar parsing helpers."""

from __future__ import annotations

from typing import Any


def _safe_int(
    value: Any,
    default: int | None = None,
    *,
    allow_bool: bool = True,
) -> int | None:
    if isinstance(value, bool) and not allow_bool:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bounded_percentage(value: Any) -> int | None:
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return None
    return max(0, min(100, number))


def _json_number(value: float | int | None) -> float | int | None:
    if isinstance(value, bool) or value is None:
        return value
    if float(value).is_integer():
        return int(value)
    return float(value)
