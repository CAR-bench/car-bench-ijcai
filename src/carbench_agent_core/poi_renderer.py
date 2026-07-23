"""POI category and presentation helpers for the CAR-bench controller."""

from __future__ import annotations

import re
from typing import Any


POI_CATEGORY_TERMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("restaurants", ("restaurant", "restaurants", "meal", "dinner", "lunch")),
    ("fast_food", ("fast food", "burger", "drive-through", "drive through")),
    ("charging_stations", ("charging station", "charging stations", "charger")),
    ("public_toilets", ("toilet", "toilets", "restroom", "bathroom")),
    ("supermarkets", ("supermarket", "grocery", "groceries")),
    ("parking", ("parking", "car park")),
    ("bakery", ("bakery", "bakeries")),
    ("airports", ("airport", "airports")),
)


def _extract_poi_category(text: str) -> str:
    return _poi_category_from_text(text) or "restaurants"


def _poi_category_from_text(text: str) -> str | None:
    for category, terms in POI_CATEGORY_TERMS:
        if any(term in text for term in terms):
            return category
    return None


def _format_poi_choice_prompt(category: str, pois: list[dict[str, Any]]) -> str:
    label = _friendly_poi_category(category)
    options = []
    for index, poi in enumerate(pois[:4], start=1):
        name = poi.get("name")
        if not isinstance(name, str) or not name.strip():
            name = f"Option {index}"
        details = []
        opening_hours = _format_opening_hours(
            poi.get("opening_hours") or poi.get("opening_times") or poi.get("hours")
        )
        if opening_hours:
            details.append(f"open {opening_hours}")
        rating = poi.get("rating")
        if isinstance(rating, (int, float)):
            details.append(f"rating {rating:g}")
        suffix = f" ({', '.join(details)})" if details else ""
        options.append(f"{index}. {name}{suffix}")

    singular = label[:-1] if label.endswith("s") else label
    return f"I found these {label}: {'; '.join(options)}. Which {singular} would you like?"


def _friendly_poi_category(category: str) -> str:
    return {
        "restaurants": "restaurants",
        "fast_food": "fast-food places",
        "charging_stations": "charging stations",
        "public_toilets": "public toilets",
        "supermarkets": "supermarkets",
        "parking": "parking options",
        "bakery": "bakeries",
        "airports": "airports",
    }.get(category, category.replace("_", " "))


def _poi_open_at_minutes(poi: dict[str, Any], target_minutes: int) -> bool:
    opening_hours = _format_opening_hours(
        poi.get("opening_hours") or poi.get("opening_times") or poi.get("hours")
    )
    if not opening_hours:
        return False
    match = re.search(
        r"(\d{1,2}):(\d{2})\s*h?\s*-\s*(\d{1,2}):(\d{2})\s*h?",
        opening_hours,
    )
    if not match:
        return False
    open_minutes = int(match.group(1)) * 60 + int(match.group(2))
    close_hour = int(match.group(3))
    close_minutes = min(close_hour, 23) * 60 + int(match.group(4))
    if close_hour == 24:
        close_minutes = 24 * 60 - 1
    if close_minutes < open_minutes:
        return target_minutes >= open_minutes or target_minutes <= close_minutes
    return open_minutes <= target_minutes <= close_minutes


def _format_opening_hours(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, dict):
        for key in ("today", "current_day", "opening_hours", "hours"):
            if key in value:
                return _format_opening_hours(value[key])
        for nested in value.values():
            formatted = _format_opening_hours(nested)
            if formatted:
                return formatted
        return None
    if isinstance(value, list):
        formatted_values = [
            formatted
            for item in value[:2]
            if (formatted := _format_opening_hours(item))
        ]
        return ", ".join(formatted_values) if formatted_values else None
    text = str(value).strip()
    if not text:
        return None

    def convert_ampm(match: re.Match[str]) -> str:
        hour = int(match.group(1))
        minute = match.group(2) or "00"
        suffix = match.group(3).lower()
        if suffix == "pm" and hour != 12:
            hour += 12
        if suffix == "am" and hour == 12:
            hour = 0
        return f"{hour:02d}:{minute}h"

    text = re.sub(
        r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b",
        convert_ampm,
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\b(\d{1,2}:\d{2})(?!\s*h)\b", r"\1h", text)
    text = re.sub(r"\s*[-\u2013\u2014]\s*", " - ", text)
    return text
