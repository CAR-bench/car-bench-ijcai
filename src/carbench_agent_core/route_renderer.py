"""Route option selection and user-visible route text helpers."""

from __future__ import annotations

from typing import Any, Literal

from .value_parsing import _safe_int


def _navigation_completion_message(
    action_summary: str,
    route: dict[str, Any],
    preference: Literal["fastest", "shortest"] | None,
    considered_routes: list[dict[str, Any]] | None = None,
) -> str:
    parts = [f"Done, {action_summary}."]
    route_choice = _route_choice_label(route, preference)
    if route_choice:
        parts.append(f"I used the {route_choice} route.")
    toll_notice = _route_toll_notice_parts(
        [route],
        considered_routes,
        selected_text="This route includes toll roads.",
        alternative_text="An available alternative route includes toll roads.",
        alternative_question="Would you like more information on the alternative routes?",
    )
    parts.extend(toll_notice)
    if _has_route_alternatives(
        [route], considered_routes
    ) and not _includes_alternative_route_question(parts):
        parts.append("Would you like more information on the alternative routes?")
    return " ".join(parts)


def _navigation_multi_route_completion_message(
    action_summary: str,
    routes: list[dict[str, Any]],
    preference: Literal["fastest", "shortest"] | None,
    considered_routes: list[dict[str, Any]] | None = None,
) -> str:
    parts = [f"Done, {action_summary}."]
    route_choice = _route_choice_label(routes[0], preference) if routes else None
    if route_choice:
        parts.append(f"I used the {route_choice} route segments.")
    toll_notice = _route_toll_notice_parts(
        routes,
        considered_routes,
        selected_text="At least one selected route segment includes toll roads.",
        alternative_text="An available alternative route segment includes toll roads.",
        alternative_question="Would you like more information on the alternative route segments?",
    )
    parts.extend(toll_notice)
    if _has_route_alternatives(
        routes, considered_routes
    ) and not _includes_alternative_route_question(parts):
        parts.append("Would you like more information on the alternative route segments?")
    return " ".join(parts)


def _route_choice_label(
    route: dict[str, Any], preference: Literal["fastest", "shortest"] | None
) -> str | None:
    if preference is not None:
        return preference
    aliases = route.get("alias")
    if isinstance(aliases, list):
        lowered_aliases = {str(alias).lower() for alias in aliases}
        if "fastest" in lowered_aliases and "shortest" in lowered_aliases:
            return "fastest and shortest"
        if "fastest" in lowered_aliases:
            return "fastest"
        if "shortest" in lowered_aliases:
            return "shortest"
        if "second" in lowered_aliases:
            return "second"
        if "third" in lowered_aliases:
            return "third"
        if "first" in lowered_aliases:
            return "first"
    return None


def _route_has_toll(route: dict[str, Any]) -> bool:
    if route.get("includes_toll") is True:
        return True
    road_types = route.get("road_types")
    if isinstance(road_types, list):
        return any("toll" in str(road_type).lower() for road_type in road_types)
    return False


def _route_toll_notice_parts(
    selected_routes: list[dict[str, Any]],
    considered_routes: list[dict[str, Any]] | None,
    *,
    selected_text: str,
    alternative_text: str,
    alternative_question: str | None = None,
    fastest_alternative_text: str | None = None,
) -> list[str]:
    if any(_route_has_toll(route) for route in selected_routes):
        return [selected_text]
    if considered_routes:
        toll_routes = [route for route in considered_routes if _route_has_toll(route)]
        if not toll_routes:
            return []
        selected_ids = {
            route_id
            for route in selected_routes
            if isinstance(route_id := route.get("route_id"), str)
        }
        fastest_toll_alternative = any(
            _route_has_alias(route, "fastest")
            and (
                not selected_ids
                or not isinstance(route.get("route_id"), str)
                or route.get("route_id") not in selected_ids
            )
            for route in toll_routes
        )
        parts = [
            fastest_alternative_text
            if fastest_alternative_text and fastest_toll_alternative
            else alternative_text
        ]
        if alternative_question:
            parts.append(alternative_question)
        return parts
    return []


def _route_has_alias(route: dict[str, Any], alias: str) -> bool:
    aliases = route.get("alias")
    if not isinstance(aliases, list):
        return False
    return alias in {str(value).lower() for value in aliases}


def _select_route(
    routes: list[dict[str, Any]],
    preference: Literal["fastest", "shortest"] | None,
) -> dict[str, Any] | None:
    if not routes:
        return None
    if preference is None:
        return routes[0] if len(routes) == 1 else None
    for route in routes:
        if _route_has_alias(route, preference):
            return route
    return routes[0] if len(routes) == 1 else None


def _select_requested_route(
    routes: list[dict[str, Any]],
    preference: Literal["fastest", "shortest"] | None,
    selected_index: int | None,
) -> dict[str, Any] | None:
    if selected_index is not None:
        if 0 <= selected_index < len(routes):
            return routes[selected_index]
        return None
    return _select_route(routes, preference)


def _select_toll_aware_route(
    routes: list[dict[str, Any]],
    preference: Literal["fastest", "shortest"],
    tolerance_minutes: int,
) -> dict[str, Any] | None:
    if not routes:
        return None
    baseline = _select_route(routes, preference) or routes[0]
    toll_free_routes = [route for route in routes if not _route_has_toll(route)]
    if not toll_free_routes:
        return baseline
    best_toll_free = min(
        toll_free_routes,
        key=lambda route: _route_duration_minutes(route) or float("inf"),
    )
    baseline_minutes = _route_duration_minutes(baseline)
    toll_free_minutes = _route_duration_minutes(best_toll_free)
    if baseline_minutes is None or toll_free_minutes is None:
        return best_toll_free
    if toll_free_minutes - baseline_minutes <= tolerance_minutes:
        return best_toll_free
    return baseline


def _route_duration_minutes(route: dict[str, Any]) -> int | None:
    hours = _safe_int(route.get("duration_hours"), 0, allow_bool=False) or 0
    minutes = _safe_int(route.get("duration_minutes"), 0, allow_bool=False) or 0
    total = hours * 60 + minutes
    return total if total > 0 else None


def _route_choice_is_unambiguous(routes: list[dict[str, Any]]) -> bool:
    if len(routes) == 1:
        return True
    fastest = _select_route(routes, "fastest")
    shortest = _select_route(routes, "shortest")
    return (
        fastest is not None
        and shortest is not None
        and fastest.get("route_id") == shortest.get("route_id")
    )


def _format_route_choice_prompt(routes: list[dict[str, Any]]) -> str:
    fastest = _select_route(routes, "fastest")
    shortest = _select_route(routes, "shortest")
    if (
        fastest is not None
        and shortest is not None
        and fastest.get("route_id") == shortest.get("route_id")
    ):
        second_route = _select_route_by_alias(routes, "second")
        if second_route is None and len(routes) > 1:
            second_route = routes[1]
        if second_route is not None and second_route.get("route_id") != fastest.get(
            "route_id"
        ):
            return (
                f"I found routes. Fastest and shortest: {_route_summary(fastest)}. "
                f"Second route: {_route_summary(second_route)}. "
                "Do you want the first route or the second route?"
            )
        return (
            f"I selected the fastest route: {_route_summary(fastest)}. "
            "It is also the shortest route. Do you want me to apply it?"
        )
    if fastest is not None and shortest is not None:
        extra_count = max(len(routes) - len({id(fastest), id(shortest)}), 0)
        extra_note = (
            f" There are {extra_count} other alternatives." if extra_count else ""
        )
        return (
            f"I found routes. Fastest: {_route_summary(fastest)}. "
            f"Shortest: {_route_summary(shortest)}.{extra_note} "
            "Do you want the fastest or shortest route?"
        )
    if len(routes) > 1:
        second_route = _select_route_by_alias(routes, "second") or routes[1]
        return (
            f"I found routes. First route: {_route_summary(routes[0])}. "
            f"Second route: {_route_summary(second_route)}. "
            "Do you want the first route or the second route?"
        )
    return "I found multiple routes. Do you want the fastest or shortest route?"


def _format_route_alternative_detail_prompt(routes: list[dict[str, Any]]) -> str:
    if not routes:
        return "I don't have route alternatives available. Do you want the fastest or shortest route?"

    labels = ("First", "Second", "Third", "Fourth", "Fifth")
    parts = []
    for index, route in enumerate(routes[:5]):
        label = _route_display_label(route, index, labels)
        parts.append(f"{label}: {_route_summary(route)}.")

    choices = _route_choice_labels(routes[:5])
    if choices:
        parts.append(f"Which route would you like: {choices}?")
    else:
        parts.append("Which route would you like?")
    return " ".join(parts)


def _route_summary(route: dict[str, Any]) -> str:
    distance = route.get("distance_km")
    hours = _safe_int(route.get("duration_hours"), 0, allow_bool=False) or 0
    minutes = _safe_int(route.get("duration_minutes"), 0, allow_bool=False) or 0
    via = route.get("name_via")
    parts = []
    if isinstance(via, str) and via:
        parts.append(f"via {via}")
    if isinstance(distance, (int, float)):
        parts.append(f"{distance:g} km")
    if hours or minutes:
        parts.append(_format_duration(hours, minutes))
    if _route_has_toll(route):
        parts.append("includes toll roads")
    return ", ".join(parts) if parts else "route details are available"


def _select_route_by_alias(
    routes: list[dict[str, Any]], alias: str
) -> dict[str, Any] | None:
    for route in routes:
        if _route_has_alias(route, alias):
            return route
    return None


def _route_display_label(
    route: dict[str, Any], index: int, labels: tuple[str, ...]
) -> str:
    aliases = route.get("alias") or []
    if isinstance(aliases, list):
        lowered = {str(alias).lower() for alias in aliases}
        for preferred in ("fastest", "shortest", "second", "third", "first"):
            if preferred in lowered:
                return preferred.capitalize()
    return labels[index] if index < len(labels) else f"Route {index + 1}"


def _route_choice_labels(routes: list[dict[str, Any]]) -> str:
    labels: list[str] = []
    for index, route in enumerate(routes):
        label = _route_display_label(route, index, ("first", "second", "third"))
        lowered = label.lower()
        if lowered not in labels:
            labels.append(lowered)
    if not labels:
        return ""
    if len(labels) == 1:
        return labels[0]
    return f"{', '.join(labels[:-1])}, or {labels[-1]}"


def _format_duration(hours: int, minutes: int) -> str:
    if hours and minutes:
        return f"{hours} h {minutes} min"
    if hours:
        return f"{hours} h"
    return f"{minutes} min"


def _has_route_alternatives(
    selected_routes: list[dict[str, Any]],
    considered_routes: list[dict[str, Any]] | None,
) -> bool:
    if not considered_routes:
        return False
    selected_ids = {
        route_id
        for route in selected_routes
        if isinstance(route_id := route.get("route_id"), str)
    }
    considered_ids = {
        route_id
        for route in considered_routes
        if isinstance(route_id := route.get("route_id"), str)
    }
    if selected_ids and considered_ids:
        return bool(considered_ids - selected_ids)
    return len(considered_routes) > len(selected_routes)


def _includes_alternative_route_question(parts: list[str]) -> bool:
    return any("would you like more information" in part.lower() for part in parts)
