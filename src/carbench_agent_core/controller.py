"""Policy-aware CAR-bench controller.

The controller handles deterministic safety/disambiguation flows before the
agent falls back to an LLM. It deliberately uses only information visible to
the agent under test: the system/wiki prompt, current tool schema, user turns,
and tool observations.
"""

from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Literal

from .actions import NextAction
from .flows import (
    AirCirculationFlow,
    AirConditioningFlow,
    AirQualityFlow,
    AmbientLightFlow,
    CabinTemperatureFlow,
    ClimateEnergyInspectionFlow,
    ClimateInspectionFlow,
    ControllerState,
    DefrostFlow,
    DriverComfortTemperatureFlow,
    EmailFlow,
    FanSpeedFlow,
    FogLightFlow,
    HighBeamFlow,
    MultiStopNavigationFlow,
    NavigationFlow,
    OccupancyComfortFlow,
    POIFlow,
    PassengerComfortMatchFlow,
    ReadingLightFlow,
    ReadingLightOccupancyFlow,
    RouteEnergyFlow,
    RuntimeContext,
    SteeringWheelHeatingFlow,
    SunroofFlow,
    SunshadeFlow,
    TrunkFlow,
    WindowFlow,
    WindowMatchFlow,
)
from .poi_renderer import (
    _extract_poi_category,
    _format_poi_choice_prompt,
    _friendly_poi_category,
    _poi_category_from_text,
    _poi_open_at_minutes,
)
from .response_renderer import (
    clean_user_content,
    render_malformed_tool_arguments,
    render_malformed_tool_call,
)
from .route_renderer import (
    _format_route_alternative_detail_prompt,
    _format_route_choice_prompt,
    _navigation_completion_message,
    _navigation_multi_route_completion_message,
    _route_choice_is_unambiguous,
    _route_choice_label,
    _route_duration_minutes,
    _route_has_alias,
    _route_summary,
    _route_toll_notice_parts,
    _select_requested_route,
    _select_route,
    _select_toll_aware_route,
)
from .text_utils import _normalized_text
from .tool_index import ToolIndex
from .tool_parsing import (
    _parse_tool_call_arguments,
    _parse_tool_result_content,
    _tool_argument_available,
    _tool_result_name,
)
from .value_parsing import (
    _bounded_percentage,
    _json_number,
    _safe_float,
    _safe_int,
)
from .vehicle_renderer import (
    _format_air_conditioning_completion,
    _format_child_seat_heating_status,
    _format_climate_energy_inspection_summary,
    _format_climate_inspection_summary,
    _format_combined_window_completion,
    _format_percentage,
    _format_recent_window_ac_status,
    _format_temperature_target,
    _format_unoccupied_seat_heating_status,
    _friendly_air_mode,
    _friendly_defrost_window,
    _friendly_reading_light_position,
    _join_completed_actions,
    friendly_window_name,
)


BAD_SUNROOF_WEATHER = {"rainy", "cloudy_and_rain", "foggy", "snowy"}
SAFE_SUNROOF_WEATHER = {"sunny", "cloudy", "partly_cloudy"}
SAFE_FOG_LIGHT_WEATHER = {"cloudy_and_thunderstorm", "cloudy_and_hail"}


@dataclass(frozen=True)
class _FlowSpec:
    """Declarative lifecycle and dispatch metadata for one controller flow."""

    name: str
    handler_name: str
    factory: type[Any] | None = None
    tracks_obligation: bool = False
    reset_while_other_pending: bool = False
    done_reason: str | None = None

    @property
    def completion_reason(self) -> str:
        return self.done_reason or f"{self.name}_done"


# Order is part of the controller contract: when multiple flows are active, the
# first one owns the next action exactly as it did in the original branch chain.
_FLOW_SPECS = (
    _FlowSpec("sunroof", "_next_sunroof_action", SunroofFlow, True),
    _FlowSpec("sunshade", "_next_sunshade_action", SunshadeFlow, True),
    _FlowSpec("window_match", "_next_window_match_action", WindowMatchFlow, True),
    _FlowSpec("window", "_next_window_action", WindowFlow, True, True),
    _FlowSpec("ambient_light", "_next_ambient_light_action", AmbientLightFlow, True),
    _FlowSpec("trunk", "_next_trunk_action", TrunkFlow, True),
    _FlowSpec(
        "air_conditioning",
        "_next_air_conditioning_action",
        AirConditioningFlow,
        True,
        True,
    ),
    _FlowSpec(
        "air_circulation",
        "_next_air_circulation_action",
        AirCirculationFlow,
        True,
        True,
    ),
    _FlowSpec("air_quality", "_next_air_quality_action"),
    _FlowSpec("fan_speed", "_next_fan_speed_action", FanSpeedFlow, True, True),
    _FlowSpec(
        "cabin_temperature",
        "_next_cabin_temperature_action",
        CabinTemperatureFlow,
        True,
        True,
    ),
    _FlowSpec(
        "steering_wheel_heating",
        "_next_steering_wheel_heating_action",
        SteeringWheelHeatingFlow,
        True,
        True,
    ),
    _FlowSpec(
        "climate_energy_inspection",
        "_next_climate_energy_inspection_action",
        ClimateEnergyInspectionFlow,
        True,
        True,
    ),
    _FlowSpec(
        "climate_inspection",
        "_next_climate_inspection_action",
        ClimateInspectionFlow,
        True,
        True,
    ),
    _FlowSpec(
        "occupancy_comfort",
        "_next_occupancy_comfort_action",
        OccupancyComfortFlow,
        True,
        True,
    ),
    _FlowSpec(
        "driver_comfort_temperature",
        "_next_driver_comfort_temperature_action",
        DriverComfortTemperatureFlow,
        True,
        True,
        "driver_comfort_done",
    ),
    _FlowSpec(
        "passenger_comfort_match",
        "_next_passenger_comfort_match_action",
        PassengerComfortMatchFlow,
        True,
        True,
    ),
    _FlowSpec("reading_light_occupancy", "_next_reading_light_occupancy_action"),
    _FlowSpec("reading_light", "_next_reading_light_action"),
    _FlowSpec("defrost", "_next_defrost_action", DefrostFlow, True),
    _FlowSpec("route_energy", "_next_route_energy_action", RouteEnergyFlow, True, True),
    _FlowSpec("poi", "_next_poi_action", POIFlow, True, True),
    _FlowSpec(
        "multi_stop_navigation",
        "_next_multi_stop_navigation_action",
        MultiStopNavigationFlow,
        True,
        True,
    ),
    _FlowSpec("navigation", "_next_navigation_action", tracks_obligation=True),
    _FlowSpec("email", "_next_email_action", tracks_obligation=True),
    _FlowSpec("high_beam", "_next_high_beam_action"),
    _FlowSpec("fog_lights", "_next_fog_lights_action"),
)


class PolicyAwareController:
    """Deterministic guard layer for high-value CAR-bench policies."""

    def __init__(self) -> None:
        self._states: dict[str, ControllerState] = {}

    def reset(self, context_id: str) -> None:
        self._states.pop(context_id, None)

    def pending_obligations(self, context_id: str) -> tuple[str, ...]:
        """Return unfinished deterministic flows for diagnostics and auditing."""

        state = self._states.get(context_id)
        if state is None:
            return ()
        return tuple(
            spec.name
            for spec in _FLOW_SPECS
            if spec.tracks_obligation
            and getattr(state, spec.name).active
            and not getattr(state, spec.name).completed
        )

    def decide(
        self,
        *,
        context_id: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        latest_user_text: str | None = None,
        latest_tool_results: list[dict[str, Any]] | None = None,
    ) -> NextAction | None:
        """Return a policy action, or None to let the model decide."""

        state = self._states.setdefault(context_id, ControllerState())
        tool_index = ToolIndex(tools)
        self._sync_runtime_context(state, messages)
        self._sync_recent_tool_call_arguments(state, messages)

        if latest_user_text:
            self._observe_user_text(state, latest_user_text)

        if latest_tool_results:
            self._observe_tool_results(state, latest_tool_results)

        self._clear_completed_flows_with_pending_work(state)

        action: NextAction | None = None
        if state.pending_direct_response:
            action = NextAction.respond(
                state.pending_direct_response,
                reason="direct_status_response",
            )
            state.pending_direct_response = None
        elif state.pending_set_navigation_route_ids:
            action = self._next_pending_navigation_setup_action(state, tool_index)
        else:
            action = self._next_active_flow_action(state, tool_index)

        if (
            action is None
            and latest_user_text
            and _is_contact_lookup_or_call_request(latest_user_text.lower())
            and not tool_index.has("get_contact_id_by_contact_name")
        ):
            action = NextAction.respond(
                "I can't look up contacts because contact search is unavailable right now.",
                reason="missing_contact_lookup_tool",
            )

        if action is not None:
            return self._validated_controller_action(action, tool_index)

        return None

    def _next_active_flow_action(
        self, state: ControllerState, tool_index: ToolIndex
    ) -> NextAction | None:
        flow_states = tuple(
            (spec, getattr(state, spec.name)) for spec in _FLOW_SPECS
        )
        for spec, flow in flow_states:
            if not flow.active:
                continue

            handler = getattr(self, spec.handler_name)
            action = handler(state, tool_index)
            if (
                action is not None
                and action.action == "respond"
                and action.reason.endswith("_done")
            ):
                completion = next(
                    (
                        (completion_spec, completion_flow)
                        for completion_spec, completion_flow in flow_states
                        if completion_spec.completion_reason == action.reason
                    ),
                    None,
                )
                completion_verified = completion is not None and (
                    completion[1].completed
                    or getattr(state, completion[0].name).completed
                )
            else:
                completion_verified = True

            if not completion_verified:
                assert action is not None
                return NextAction.respond(
                    "I couldn't verify that the requested action completed, "
                    "so I won't report it as done.",
                    reason=f"{action.reason}_evidence_guard",
                )
            return action
        return None

    def _validated_controller_action(
        self, action: NextAction, tool_index: ToolIndex
    ) -> NextAction:
        if action.action == "respond":
            return NextAction.respond(
                clean_user_content(action.content),
                reason=action.reason,
            )

        if action.action != "tool_calls" or not action.tool_calls:
            return NextAction.respond(
                render_malformed_tool_call(),
                reason=f"{action.reason}_invalid_tool_call",
            )

        for tool_call in action.tool_calls:
            if not isinstance(tool_call, dict):
                return NextAction.respond(
                    render_malformed_tool_call(),
                    reason=f"{action.reason}_invalid_tool_call",
                )
            tool_name = tool_call.get("tool_name")
            arguments = tool_call.get("arguments") or {}
            if not isinstance(tool_name, str) or not tool_name:
                return NextAction.respond(
                    render_malformed_tool_call(),
                    reason=f"{action.reason}_invalid_tool_call",
                )
            if not isinstance(arguments, dict):
                return NextAction.respond(
                    render_malformed_tool_arguments(),
                    reason=f"{action.reason}_invalid_tool_arguments",
                )
            validation_error = tool_index.validate_call(tool_name, arguments)
            if validation_error:
                return NextAction.respond(
                    validation_error,
                    reason=f"{action.reason}_schema_guard",
                )

        return action

    def _clear_completed_flows_with_pending_work(self, state: ControllerState) -> None:
        if not any(
            getattr(state, spec.name).active
            and not getattr(state, spec.name).completed
            for spec in _FLOW_SPECS
            if spec.tracks_obligation
        ):
            return
        self._reset_completed_flows(state, while_other_pending=True)

    def _reset_completed_flows_for_new_intent(self, state: ControllerState) -> None:
        self._reset_completed_flows(state)

    @staticmethod
    def _reset_completed_flows(
        state: ControllerState, *, while_other_pending: bool = False
    ) -> None:
        for spec in _FLOW_SPECS:
            if spec.factory is None:
                continue
            if while_other_pending and not spec.reset_while_other_pending:
                continue
            if getattr(state, spec.name).completed:
                setattr(state, spec.name, spec.factory())

    def _sync_runtime_context(
        self, state: ControllerState, messages: list[dict[str, Any]]
    ) -> None:
        if not messages or messages[0].get("role") != "system":
            return
        system_prompt = messages[0].get("content") or ""

        location = _extract_json_after_label(system_prompt, "CURRENT_LOCATION")
        if isinstance(location, dict):
            state.runtime.location_id = location.get("id") or state.runtime.location_id

        datetime_value = _extract_json_after_label(system_prompt, "DATETIME")
        if isinstance(datetime_value, dict):
            state.runtime.month = _safe_int(
                datetime_value.get("month"), state.runtime.month
            )
            state.runtime.day = _safe_int(datetime_value.get("day"), state.runtime.day)
            state.runtime.hour = _safe_int(
                datetime_value.get("hour"), state.runtime.hour
            )
            state.runtime.minute = _safe_int(
                datetime_value.get("minute"), state.runtime.minute
            )

        business_email_extra_recipients = _extract_business_email_extra_recipients(
            system_prompt
        )
        if business_email_extra_recipients:
            state.business_email_extra_recipients = business_email_extra_recipients

    def _sync_recent_tool_call_arguments(
        self, state: ControllerState, messages: list[dict[str, Any]]
    ) -> None:
        for message in reversed(messages):
            if message.get("role") != "assistant":
                continue
            if not message.get("tool_calls"):
                return
            for tool_call in reversed(message["tool_calls"]):
                function = tool_call.get("function") or {}
                if function.get("name") != "get_location_id_by_location_name":
                    continue
                arguments = _parse_tool_call_arguments(function.get("arguments"))
                location = arguments.get("location")
                if isinstance(location, str):
                    state.pending_location_lookup_name = (
                        _clean_location_query(location) or location.strip()
                    )
                return
            return

    def _observe_user_text(self, state: ControllerState, text: str) -> None:
        text = text.strip()
        if not text:
            return
        previous_user_text = state.last_user_text
        state.last_user_text = text
        lowered = text.lower()

        if "###stop###" in lowered:
            return

        if _contains_new_action_intent(lowered):
            self._reset_completed_flows_for_new_intent(state)

        if _is_planned_navigation_setup_request(lowered) and (
            state.planned_navigation_route_ids
        ):
            state.pending_set_navigation_route_ids = list(
                state.planned_navigation_route_ids
            )
            return

        if _is_window_ac_status_query(lowered):
            response = _format_recent_window_ac_status(state)
            if response is not None:
                state.pending_direct_response = response
                return

        if _is_child_seat_heating_status_query(lowered):
            response = _format_child_seat_heating_status(state.occupancy_comfort)
            if response is not None:
                state.pending_direct_response = response
                return

        quoted_meeting = _extract_quoted_meeting_topic(text)
        if quoted_meeting is not None:
            state.recent_meeting_topic = quoted_meeting

        if (
            state.email.active
            and state.email.mode == "meeting_attendees"
            and _is_route_energy_request(lowered)
            and not _is_email_composition_request(lowered)
        ):
            state.email = EmailFlow()

        sunroof = state.sunroof
        if sunroof.weather_confirmation_requested and _is_affirmative(lowered):
            sunroof.weather_confirmed = True
            sunroof.weather_confirmation_requested = False
            return

        email = state.email
        if email.confirmation_requested:
            if _is_affirmative(lowered):
                email.confirmed = True
                email.confirmation_requested = False
                return
            if _is_negative(lowered):
                state.email = EmailFlow()
                return

        if email.active and email.pending_contact_matches:
            selected = _select_contact_match_from_text(
                email.pending_contact_matches, text
            )
            if selected is not None:
                if email.pending_lookup_role == "recipient":
                    email.recipient_contact_id = selected
                    email.recipient_name = email.pending_contact_matches[selected]
                    _set_email_flow_name(email, "recipient", email.recipient_name)
                elif email.pending_lookup_role == "subject":
                    email.subject_contact_id = selected
                    email.subject_name = email.pending_contact_matches[selected]
                    _set_email_flow_name(email, "subject", email.subject_name)
                email.pending_contact_matches = {}
                email.pending_lookup_role = None
                return

        if email.active and email.mode == "share_contact":
            subject_name = _extract_contact_share_subject(text)
            if subject_name is not None and email.subject_contact_id is None:
                email.subject_name = subject_name
                _set_email_flow_name(email, "subject", subject_name)
                return

        if email.active and email.mode == "meeting_attendees":
            _update_email_weather_request_from_text(email, text)
            if email.calendar_checked and not email.attendee_contact_ids:
                selected = _select_meeting_from_text(
                    state.recent_calendar_meetings, text
                )
                if selected is not None:
                    _record_email_attendee_meeting(state, selected)
                    return
            if (
                email.attendee_contact_ids
                and _email_attendee_addresses(email)
                and email.content_message is None
                and not email.weather_requested
                and not _is_affirmative(lowered)
                and not _is_negative(lowered)
            ):
                email.content_message = _normalize_email_content(text)
                return

        window = state.window
        if window.ac_confirmation_requested and _is_affirmative(lowered):
            window.ac_confirmed = True
            window.ac_confirmation_requested = False
            return

        trunk = state.trunk
        if trunk.confirmation_requested and _is_affirmative(lowered):
            trunk.confirmed = True
            trunk.confirmation_requested = False
            return

        high_beam = state.high_beam
        if high_beam.confirmation_requested:
            if _is_affirmative(lowered):
                high_beam.confirmed = True
                high_beam.confirmation_requested = False
                return
            if _is_negative(lowered):
                high_beam.declined = True
                high_beam.confirmation_requested = False
                return

        fog_lights = state.fog_lights
        if fog_lights.weather_confirmation_requested:
            if _is_affirmative(lowered):
                fog_lights.weather_confirmed = True
                fog_lights.weather_confirmation_requested = False
                return
            if _is_negative(lowered):
                fog_lights.declined = True
                fog_lights.weather_confirmation_requested = False
                return

        if fog_lights.high_beam_confirmation_requested:
            if _is_affirmative(lowered):
                fog_lights.high_beam_confirmed = True
                fog_lights.high_beam_confirmation_requested = False
                return
            if _is_negative(lowered):
                fog_lights.declined = True
                fog_lights.high_beam_confirmation_requested = False
                return

        if sunroof.active and sunroof.target_percentage is None:
            clarified = _extract_percentage(lowered, allow_standalone=True)
            if clarified is not None:
                sunroof.target_percentage = clarified
                return

        if state.sunshade.active and state.sunshade.target_percentage is None:
            clarified = _extract_percentage(lowered, allow_standalone=True)
            if clarified is not None:
                state.sunshade.target_percentage = clarified
                return

        if window.active and window.target_percentage is None:
            clarified = _extract_percentage(lowered, allow_standalone=True)
            if clarified is not None:
                window.target_percentage = clarified
                return

        if state.ambient_light.active and state.ambient_light.target_color is None:
            if _is_ambient_light_match_car_color_request(lowered):
                state.ambient_light.match_car_color = True
                return
            clarified_color = _extract_ambient_color(lowered)
            if clarified_color is not None:
                state.ambient_light.target_color = clarified_color
                return

        if state.air_circulation.active and state.air_circulation.mode is None:
            clarified_mode = _extract_air_circulation_mode(lowered)
            if clarified_mode is not None:
                state.air_circulation.mode = clarified_mode
                return

        if state.air_quality.active:
            clarified_level = _extract_level(lowered)
            clarified_mode = _extract_air_circulation_mode(lowered)
            if clarified_level is not None or "fan" in lowered:
                state.fan_speed = FanSpeedFlow(active=True, level=clarified_level)
                state.air_quality = AirQualityFlow()
                return
            if clarified_mode is not None:
                state.air_circulation = AirCirculationFlow(
                    active=True,
                    mode=clarified_mode,
                )
                state.air_quality = AirQualityFlow()
                return
            if _is_air_conditioning_request(lowered):
                state.air_conditioning = AirConditioningFlow(
                    active=True,
                    on=not _is_air_conditioning_off_request(lowered),
                    fan_target_level=_extract_level(lowered),
                )
                state.air_quality = AirQualityFlow()
                return

        if state.fan_speed.active and state.fan_speed.level is None:
            if _is_current_fan_speed_query(lowered):
                state.fan_speed.query_current = True
                return
            clarified_level = _extract_level(lowered)
            if clarified_level is not None:
                state.fan_speed.level = clarified_level
                return

        if (
            state.steering_wheel_heating.active
            and state.steering_wheel_heating.level is None
        ):
            clarified_level = _extract_heating_level(lowered)
            if clarified_level is not None:
                state.steering_wheel_heating.level = clarified_level
                return

        if state.occupancy_comfort.active:
            clarified_temperature = _resolve_temperature_setting(
                lowered,
                state,
                _extract_temperature_seat_zone(lowered),
            )
            clarified_level = _extract_heating_level(lowered)
            clarified_delta = _extract_heating_delta(lowered)
            updated = False
            if (
                "steering wheel" in lowered
                and clarified_level is not None
                and state.occupancy_comfort.steering_wheel_level is None
            ):
                state.occupancy_comfort.steering_wheel_level = clarified_level
                updated = True
            if (
                clarified_temperature is not None
                and state.occupancy_comfort.target_temperature is None
            ):
                state.occupancy_comfort.needs_temperature = True
                state.occupancy_comfort.target_temperature = clarified_temperature
                state.occupancy_comfort.completed = False
                updated = True
            if (
                clarified_delta is not None
                and state.occupancy_comfort.heating_delta is None
            ):
                state.occupancy_comfort.heating_delta = clarified_delta
                updated = True
            if (
                clarified_level is not None
                and state.occupancy_comfort.target_heating_level is None
                and "steering wheel" not in lowered
                and clarified_delta is None
            ):
                state.occupancy_comfort.target_heating_level = clarified_level
                updated = True
            if _is_unoccupied_seat_heating_status_query(lowered):
                state.occupancy_comfort.query_unoccupied_heating_status = True
                state.occupancy_comfort.completed = False
                updated = True
            if updated:
                return

        if (
            state.driver_comfort_temperature.active
            and state.driver_comfort_temperature.target_temperature is None
        ):
            clarified_temperature = _resolve_temperature_setting(
                lowered,
                state,
                "DRIVER",
            )
            if clarified_temperature is not None:
                state.driver_comfort_temperature.target_temperature = (
                    clarified_temperature
                )
                return

        if state.reading_light.active and state.reading_light.position is None:
            clarified_position = _extract_reading_light_position(lowered)
            if clarified_position is not None:
                state.reading_light.position = clarified_position
                return

        if state.defrost.active and state.defrost.defrost_window is None:
            clarified_window = _extract_defrost_window(lowered)
            if clarified_window is not None:
                state.defrost.defrost_window = clarified_window
                return

        navigation = state.navigation
        if (
            navigation.active
            and navigation.destination_name is None
            and navigation.mode in {"set_new", "replace_final_destination"}
        ):
            clarified_destination = _extract_navigation_clarified_destination(text)
            if clarified_destination is not None:
                navigation.destination_name = clarified_destination
                return

        if navigation.active and navigation.route_choice_requested:
            if _is_route_choice_preview_request(lowered):
                navigation.route_detail_requested = True
                return
            route_index = _extract_route_choice_index(lowered)
            if route_index is not None:
                navigation.selected_route_index = route_index
                navigation.route_choice_requested = False
                return
            route_preference = _extract_route_preference(lowered)
            if route_preference is not None:
                navigation.route_preference = route_preference
                navigation.route_choice_requested = False
                return
            choice_routes = _navigation_choice_routes(navigation)
            if _is_affirmative(lowered) and _route_choice_is_unambiguous(choice_routes):
                navigation.route_preference = "fastest"
                navigation.route_choice_requested = False
                return

        route_energy = state.route_energy
        if route_energy.active:
            if _is_navigation_delete_waypoint_request(lowered):
                state.route_energy = RouteEnergyFlow()
                state.navigation = _new_navigation_delete_waypoint_flow(text)
                return

            if _is_navigation_delete_destination_request(lowered):
                state.route_energy = RouteEnergyFlow()
                state.navigation = _new_navigation_delete_destination_flow(text, state)
                return

            if _is_meeting_attendees_email_request(lowered):
                state.email = _new_meeting_attendees_email_flow(text, state)
                state.route_energy = RouteEnergyFlow()
                return

            if route_energy.route_choice_requested:
                if _is_route_choice_preview_request(lowered):
                    route_energy.route_detail_requested = True
                    return
                route_index = _extract_route_choice_index(lowered)
                if route_index is None:
                    route_index = _select_route_index_by_via(
                        route_energy.routes, lowered
                    )
                if route_index is not None:
                    route_energy.selected_route_index = route_index
                    route_energy.route_choice_requested = False
                    _update_route_energy_from_text(route_energy, text)
                    return
                route_preference = _extract_route_preference(lowered)
                if route_preference is not None:
                    route_energy.route_preference = route_preference
                    route_energy.route_choice_requested = False
                    _update_route_energy_from_text(route_energy, text)
                    return

            if route_energy.selected_poi_id is None and route_energy.pois:
                selected = _select_poi_from_text(route_energy.pois, text)
                if selected is not None:
                    _record_route_energy_selected_poi(route_energy, selected)
                    _update_route_energy_from_text(route_energy, text)
                    return

            if _is_route_energy_request(lowered):
                _update_route_energy_from_text(route_energy, text)
                return

        if state.poi.active and _is_navigation_delete_destination_request(lowered):
            state.poi = POIFlow()

        poi = state.poi
        if poi.active:
            if poi.route_choice_requested:
                route_index = _extract_route_choice_index(lowered)
                if route_index is None:
                    route_index = _select_route_index_by_via(poi.routes, lowered)
                if route_index is not None:
                    poi.selected_route_index = route_index
                    poi.route_choice_requested = False
                    return
                route_preference = _extract_route_preference(lowered)
                if route_preference is not None:
                    poi.route_preference = route_preference
                    poi.route_choice_requested = False
                    return
            if poi.selected_poi_id is None and poi.pois:
                selected = _select_poi_from_text(poi.pois, text)
                if selected is not None:
                    poi.selected_poi_id = selected.get("id")
                    poi.selected_poi_name = selected.get("name")
                    return

        if _is_sunroof_open_request(lowered):
            state.sunroof = SunroofFlow(
                active=True,
                target_percentage=_extract_sunroof_percentage(lowered),
            )
        elif _is_sunshade_request(lowered):
            state.sunshade = SunshadeFlow(
                active=True,
                target_percentage=_extract_sunshade_percentage(lowered),
            )
        elif _is_window_match_request(lowered):
            state.window_match = WindowMatchFlow(
                active=True,
                reference_window=_extract_window_match_reference(lowered),
                all_windows="all windows" in lowered,
                followup_defrost_window=(
                    _extract_defrost_window(lowered) if "defrost" in lowered else None
                ),
            )
        elif _is_window_open_request(lowered):
            window_percentage = _extract_window_percentage(lowered)
            state.window = WindowFlow(
                active=True,
                window=_extract_window_target(lowered),
                target_percentage=window_percentage,
            )
            if _is_air_conditioning_request(lowered):
                state.air_conditioning = AirConditioningFlow(
                    active=True,
                    on=not _is_air_conditioning_off_request(lowered),
                    fan_target_level=_extract_level(lowered),
                    preserve_open_windows=(
                        window_percentage is None or window_percentage > 0
                    ),
                )
            if _is_air_circulation_request(lowered):
                state.air_circulation = AirCirculationFlow(
                    active=True,
                    mode=_extract_air_circulation_mode(lowered),
                    combined_with_air_conditioning=True,
                    combined_window=_extract_window_target(lowered),
                    combined_window_percentage=window_percentage,
                )
        elif _is_ambient_light_request(lowered):
            state.ambient_light = AmbientLightFlow(
                active=True,
                target_color=_extract_ambient_color(lowered),
                on=not _is_light_off_request(lowered),
                match_car_color=_is_ambient_light_match_car_color_request(lowered),
            )
        elif _is_trunk_request(lowered):
            state.trunk = TrunkFlow(
                active=True,
                action="CLOSE" if _is_close_request(lowered) else "OPEN",
            )
        elif _is_air_conditioning_request(lowered):
            state.air_conditioning = AirConditioningFlow(
                active=True,
                on=not _is_air_conditioning_off_request(lowered),
                fan_target_level=_extract_level(lowered),
            )
            if _is_air_circulation_request(lowered):
                state.air_circulation = AirCirculationFlow(
                    active=True,
                    mode=_extract_air_circulation_mode(lowered),
                    combined_with_air_conditioning=True,
                )
        elif _is_air_circulation_request(lowered):
            state.air_circulation = AirCirculationFlow(
                active=True,
                mode=_extract_air_circulation_mode(lowered),
            )
        elif _is_passenger_comfort_match_request(lowered):
            state.passenger_comfort_match = PassengerComfortMatchFlow(
                active=True,
                target_temperature=_resolve_temperature_setting(
                    lowered,
                    state,
                    "PASSENGER",
                ),
            )
        elif _is_occupancy_comfort_request(lowered):
            state.occupancy_comfort = OccupancyComfortFlow(
                active=True,
                needs_temperature=_occupancy_comfort_needs_temperature(lowered),
                target_temperature_zone=(
                    "DRIVER"
                    if _is_driver_temperature_match_passenger_request(lowered)
                    else "ALL_ZONES"
                ),
                target_temperature=_resolve_temperature_setting(
                    lowered,
                    state,
                    _extract_temperature_seat_zone(lowered),
                ),
                target_heating_level=_extract_seat_heating_target_level(lowered),
                heating_delta=_extract_heating_delta(lowered),
                turn_off_unoccupied=_requests_unoccupied_seat_heating_off(lowered),
                query_unoccupied_heating_status=(
                    _is_unoccupied_seat_heating_status_query(lowered)
                ),
                match_driver_to_passenger_temperature=(
                    _is_driver_temperature_match_passenger_request(lowered)
                ),
                steering_wheel_requested="steering wheel" in lowered,
                steering_wheel_level=_extract_steering_wheel_heating_level(lowered),
            )
        elif _is_driver_comfort_temperature_request(lowered):
            state.driver_comfort_temperature = DriverComfortTemperatureFlow(
                active=True,
                passenger_heating_required=_requests_passenger_heating_off(lowered),
                target_temperature=_resolve_temperature_setting(
                    lowered,
                    state,
                    "DRIVER",
                ),
            )
        elif _is_climate_energy_inspection_request(lowered):
            state.climate_energy_inspection = ClimateEnergyInspectionFlow(active=True)
        elif _is_climate_inspection_request(lowered):
            state.climate_inspection = ClimateInspectionFlow(
                active=True,
                needs_temperature=_mentions_temperature_setting(lowered),
                needs_seat_heating=_mentions_seat_heating(lowered),
            )
        elif _is_air_quality_question(lowered):
            state.air_quality = AirQualityFlow(active=True)
        elif _is_current_fan_speed_query(lowered):
            state.fan_speed = FanSpeedFlow(active=True, query_current=True)
        elif _is_fan_speed_request(lowered):
            state.fan_speed = FanSpeedFlow(
                active=True,
                level=0 if _is_light_off_request(lowered) else _extract_level(lowered),
            )
        elif _is_cabin_temperature_request(lowered):
            state.cabin_temperature = CabinTemperatureFlow(
                active=True,
                target_temperature=_resolve_temperature_setting(
                    lowered,
                    state,
                    _extract_temperature_seat_zone(lowered),
                ),
                seat_zone=_extract_temperature_seat_zone(lowered),
            )
        elif _is_steering_wheel_heating_request(lowered):
            state.steering_wheel_heating = SteeringWheelHeatingFlow(
                active=True,
                level=(
                    0
                    if _is_light_off_request(lowered)
                    else _extract_heating_level(lowered)
                ),
            )
        elif _is_reading_light_by_occupancy_request(lowered):
            state.reading_light_occupancy = ReadingLightOccupancyFlow(active=True)
        elif _is_reading_light_request(lowered):
            state.reading_light = ReadingLightFlow(
                active=True,
                position=_extract_reading_light_position(lowered),
                on=not _is_light_off_request(lowered),
            )
        elif _is_defrost_request(lowered):
            state.defrost = DefrostFlow(
                active=True,
                on=not _is_light_off_request(lowered),
                defrost_window=_extract_defrost_window(lowered),
            )
        elif _is_navigation_delete_destination_request(lowered):
            state.navigation = _new_navigation_delete_destination_flow(text, state)
        elif _is_meeting_attendees_email_request(lowered):
            state.email = _new_meeting_attendees_email_flow(text, state)
        elif _is_route_energy_request(lowered) or _is_recent_charging_stop_navigation_request(
            lowered, state
        ):
            state.route_energy = _new_route_energy_flow(text)
            if _is_recent_charging_stop_navigation_request(lowered, state):
                state.route_energy.wants_navigation_setup = True
            _hydrate_route_energy_from_recent(state.route_energy, state)
            if (
                state.route_energy.selected_poi_id is None
                and state.recent_charging_pois
            ):
                selected = _select_poi_from_text(state.recent_charging_pois, text)
                if selected is not None:
                    state.route_energy.pois = list(state.recent_charging_pois)
                    state.route_energy.pois_checked = True
                    state.route_energy.route_id = state.recent_charging_route_id
                    state.route_energy.at_kilometer = (
                        state.recent_charging_at_kilometer
                    )
                    _record_route_energy_selected_poi(state.route_energy, selected)
            if (
                state.route_energy.selected_poi_id is None
                and state.route_energy.pois
            ):
                selected = _select_poi_from_text(state.route_energy.pois, text)
                if selected is not None:
                    _record_route_energy_selected_poi(state.route_energy, selected)
            if state.route_energy.active:
                state.poi = POIFlow()
        elif (multi_stop_destinations := _extract_two_leg_navigation_destinations(text)) is not None:
            first_destination, final_destination = multi_stop_destinations
            state.multi_stop_navigation = MultiStopNavigationFlow(
                active=True,
                first_destination_name=first_destination,
                final_destination_name=final_destination,
                route_preference=_extract_route_preference(lowered),
            )
        elif _is_trip_planning_navigation_request(lowered) and (
            destination := _extract_navigation_destination(text)
        ) is not None:
            pending_poi = None
            if _is_poi_request(lowered):
                search_along_route = _is_along_route_poi_request(lowered)
                pending_poi = POIFlow(
                    active=True,
                    category=_extract_poi_category(lowered),
                    search_along_route=search_along_route,
                    route_preference=_extract_route_preference(lowered),
                    do_not_set_navigation=(
                        search_along_route
                        and not _is_add_waypoint_poi_request(lowered)
                    ),
                    defer_navigation_setup=True,
                )
            state.pending_poi_after_navigation = pending_poi
            state.navigation = NavigationFlow(
                active=True,
                mode="set_new",
                destination_name=destination,
                route_preference=_extract_route_preference(lowered),
                toll_avoidance_tolerance_minutes=_extract_toll_avoidance_tolerance_minutes(
                    lowered
                ),
                planning_only=_is_navigation_planning_only_request(lowered),
            )
        elif _is_poi_request(lowered):
            location_name = _extract_poi_location(text)
            search_along_route = _is_along_route_poi_request(lowered)
            route_id = _route_id_for_poi_followup(state)
            route_prefix_ids = list(state.planned_navigation_route_ids)
            route_prefix_selected_routes = list(state.planned_navigation_selected_routes)
            route_prefix_considered_routes = list(
                state.planned_navigation_considered_routes
            )
            route_start_id = (
                state.planned_navigation_waypoints[-1]
                if state.planned_navigation_waypoints
                else None
            )
            required_open_at_minutes = (
                state.planned_navigation_arrival_minutes
                if route_prefix_ids and "open" in lowered
                else None
            )
            if search_along_route:
                location_name = None
            location_id = None
            if route_prefix_ids and route_start_id and not search_along_route:
                location_id = route_start_id
            elif location_name is None and _is_nearby_poi_request(lowered):
                location_id = state.runtime.location_id
            state.poi = POIFlow(
                active=True,
                category=_extract_poi_category(lowered),
                location_name=location_name,
                location_id=location_id,
                search_along_route=search_along_route,
                route_id=route_id if search_along_route else None,
                route_start_id=route_start_id if route_prefix_ids else None,
                route_prefix_ids=route_prefix_ids,
                route_prefix_selected_routes=route_prefix_selected_routes,
                route_prefix_considered_routes=route_prefix_considered_routes,
                required_open_at_minutes=required_open_at_minutes,
                route_preference=_extract_route_preference(lowered),
                replace_final_destination=_is_replace_destination_request(lowered)
                or "changed my mind" in lowered
                or "instead" in lowered,
                do_not_set_navigation=(
                    True
                    if search_along_route
                    and not _is_add_waypoint_poi_request(lowered)
                    else False
                    if route_prefix_ids
                    else _do_not_set_navigation(lowered)
                ),
                defer_navigation_setup=bool(route_prefix_ids),
            )
        elif _is_navigation_replace_waypoint_request(lowered):
            state.navigation = NavigationFlow(
                active=True,
                mode="replace_one_waypoint",
                new_waypoint_name=_extract_navigation_new_waypoint(text),
                waypoint_name=_extract_navigation_waypoint_to_replace(text),
                route_preference=_extract_route_preference(lowered),
                needs_current_navigation=True,
            )
        elif _is_navigation_delete_waypoint_request(lowered):
            state.navigation = _new_navigation_delete_waypoint_flow(text)
        elif (
            meeting_location := _calendar_meeting_navigation_location(lowered, state)
        ) is not None:
            state.navigation = NavigationFlow(
                active=True,
                mode="set_new",
                destination_name=meeting_location,
                route_preference="fastest",
            )
        elif _is_vague_navigation_destination_request(lowered):
            state.navigation = NavigationFlow(
                active=True,
                mode=(
                    "replace_final_destination"
                    if _is_replace_destination_request(lowered)
                    else "set_new"
                ),
                route_preference=_extract_route_preference(lowered),
                toll_avoidance_tolerance_minutes=_extract_toll_avoidance_tolerance_minutes(
                    lowered
                ),
                planning_only=_is_navigation_planning_only_request(lowered),
                needs_current_navigation=_is_replace_destination_request(lowered),
            )
        elif _is_simple_navigation_request(text):
            destination = _extract_navigation_destination(text)
            if destination is not None:
                previous_navigation = state.navigation
                replace_followup = _is_navigation_destination_change_followup(
                    previous_user_text, lowered
                )
                replace_destination = (
                    _is_replace_destination_request(lowered) or replace_followup
                )
                current_waypoints = (
                    list(previous_navigation.waypoints_id)
                    if replace_destination and previous_navigation.waypoints_id
                    else []
                )
                state.navigation = NavigationFlow(
                    active=True,
                    mode="replace_final_destination"
                    if replace_destination
                    else "set_new",
                destination_name=destination,
                route_preference=_extract_route_preference(lowered),
                toll_avoidance_tolerance_minutes=_extract_toll_avoidance_tolerance_minutes(
                    lowered
                ),
                planning_only=_is_navigation_planning_only_request(lowered),
                needs_current_navigation=(
                    _navigation_replace_needs_current_state(lowered)
                    or bool(current_waypoints)
                        or (
                            replace_followup
                            and ("route" in lowered or "instead" in lowered)
                        )
                    ),
                    current_navigation_checked=bool(current_waypoints),
                    navigation_active=(
                        True
                        if current_waypoints
                        else previous_navigation.navigation_active
                    ),
                    waypoints_id=current_waypoints,
                    routes_to_final_destination_id=list(
                        previous_navigation.routes_to_final_destination_id
                    )
                    if current_waypoints
                    else [],
                )
        elif _is_meeting_delay_email_request(lowered) or (
            state.recent_meeting_topic and _is_late_email_request(lowered)
        ):
            recipient = _extract_email_recipient_name(text)
            if recipient is not None:
                state.email = EmailFlow(
                    active=True,
                    mode="meeting_delay",
                    recipient_name=recipient,
                    meeting_topic=state.recent_meeting_topic,
                    calendar_checked=bool(state.recent_calendar_meetings),
                    user_claimed_late=True,
                )
                _set_email_flow_name(state.email, "recipient", recipient)
        elif _is_contact_share_email_start(lowered):
            recipient = _extract_email_recipient_name(text)
            if recipient is not None:
                state.email = EmailFlow(
                    active=True,
                    mode="share_contact",
                    recipient_name=recipient,
                )
                _set_email_flow_name(state.email, "recipient", recipient)
        elif _is_fog_light_request(lowered):
            state.fog_lights = FogLightFlow(
                active=True,
                on=not _is_light_off_request(lowered),
            )
        elif _is_high_beam_request(lowered):
            state.high_beam = HighBeamFlow(
                active=True,
                on=not _is_light_off_request(lowered),
            )

    def _observe_tool_results(
        self, state: ControllerState, tool_results: list[dict[str, Any]]
    ) -> None:
        for tool_result in tool_results:
            name = _tool_result_name(tool_result)
            payload = _parse_tool_result_content(tool_result)
            result = payload.get("result") if isinstance(payload, dict) else None

            if name == "get_sunroof_and_sunshade_position" and isinstance(result, dict):
                state.sunroof.position_checked = True
                state.sunroof.sunroof_position = _safe_float(
                    result.get("sunroof_position"), state.sunroof.sunroof_position
                )
                state.sunroof.sunshade_position = _safe_float(
                    result.get("sunshade_position"), state.sunroof.sunshade_position
                )
            elif name == "get_weather" and isinstance(result, dict):
                state.sunroof.weather_checked = True
                current_slot = result.get("current_slot") or {}
                condition = current_slot.get("condition")
                if isinstance(condition, str):
                    state.sunroof.weather_condition = condition
                    state.fog_lights.weather_condition = condition
                if state.fog_lights.active:
                    state.fog_lights.weather_checked = True
                if state.email.active and state.email.weather_requested:
                    state.email.weather_checked = True
                    weather_condition, weather_temperature = _extract_weather_summary(
                        result
                    )
                    if weather_condition is not None:
                        state.email.weather_condition = weather_condition
                    if weather_temperature is not None:
                        state.email.weather_temperature = weather_temperature
            elif name == "get_weather" and state.email.active:
                state.email.failure_message = (
                    "I couldn't check the meeting weather, so I can't send a reliable weather update."
                )
            elif name == "get_user_preferences" and isinstance(result, dict):
                state.sunroof.preferences_checked = True
                preferred = _extract_preferred_sunroof_percentage(result)
                if preferred is not None and state.sunroof.target_percentage is None:
                    state.sunroof.target_percentage = preferred

                state.ambient_light.preferences_checked = True
                preferred_color = _extract_preferred_ambient_color(result)
                if (
                    preferred_color is not None
                    and state.ambient_light.target_color is None
                ):
                    state.ambient_light.target_color = preferred_color
                    state.ambient_light.on = preferred_color != "NONE"

                state.air_circulation.preferences_checked = True
                preferred_mode = _extract_preferred_air_circulation_mode(result)
                if preferred_mode is not None and state.air_circulation.mode is None:
                    state.air_circulation.mode = preferred_mode

                state.fan_speed.preferences_checked = True
                preferred_fan_level = _extract_preferred_fan_speed_level(result)
                if preferred_fan_level is not None and state.fan_speed.level is None:
                    state.fan_speed.level = preferred_fan_level

                state.steering_wheel_heating.preferences_checked = True
                preferred_heating_level = _extract_preferred_steering_heating_level(
                    result
                )
                if (
                    preferred_heating_level is not None
                    and state.steering_wheel_heating.level is None
                ):
                    state.steering_wheel_heating.level = preferred_heating_level
                state.driver_comfort_temperature.preferences_checked = True
                preferred_temperature = _extract_preferred_temperature(result)
                if (
                    preferred_temperature is not None
                    and state.driver_comfort_temperature.target_temperature is None
                ):
                    state.driver_comfort_temperature.target_temperature = (
                        preferred_temperature
                    )
                state.occupancy_comfort.preferences_checked = True
                if (
                    preferred_temperature is not None
                    and state.occupancy_comfort.target_temperature is None
                ):
                    state.occupancy_comfort.target_temperature = preferred_temperature
                state.business_email_extra_recipients = (
                    state.business_email_extra_recipients
                    + [
                        email
                        for email in _extract_business_email_extra_recipients(
                            json.dumps(result, ensure_ascii=False)
                        )
                        if email not in state.business_email_extra_recipients
                    ]
                )
                if state.email.active:
                    state.email.preferences_checked = True
            elif name == "get_car_color" and isinstance(result, dict):
                if state.ambient_light.active:
                    state.ambient_light.car_color_checked = True
                    car_color = result.get("car_color")
                    if isinstance(car_color, str):
                        color = car_color.upper()
                        if color in AMBIENT_COLORS:
                            state.ambient_light.target_color = color
            elif name == "open_close_sunshade" and isinstance(result, dict):
                state.sunroof.sunshade_position = _safe_float(
                    result.get("percentage"), state.sunroof.sunshade_position
                )
                if state.sunshade.active:
                    state.sunshade.completed = True
                    state.sunshade.target_percentage = _safe_int(
                        result.get("percentage"), state.sunshade.target_percentage
                    )
            elif name == "open_close_sunroof" and isinstance(result, dict):
                state.sunroof.sunroof_position = _safe_float(
                    result.get("percentage"), state.sunroof.sunroof_position
                )
                state.sunroof.completed = True
            elif name == "get_climate_settings" and isinstance(result, dict):
                state.window.climate_checked = True
                ac_value = result.get("air_conditioning")
                if isinstance(ac_value, bool):
                    state.window.ac_on = ac_value
                    state.air_conditioning.air_conditioning = ac_value
                    state.recent_air_conditioning = ac_value

                state.defrost.climate_checked = True
                fan_speed = _safe_int(result.get("fan_speed"), state.defrost.fan_speed)
                if fan_speed is not None:
                    state.defrost.fan_speed = fan_speed
                    state.air_conditioning.fan_speed = fan_speed
                airflow = result.get("fan_airflow_direction")
                if isinstance(airflow, str):
                    state.defrost.fan_airflow_direction = airflow
                if isinstance(ac_value, bool):
                    state.defrost.air_conditioning = ac_value
                if state.air_conditioning.active:
                    state.air_conditioning.climate_checked = True
                if state.air_quality.active:
                    state.air_quality.climate_checked = True
                    state.air_quality.fan_speed = fan_speed
                if state.fan_speed.active and state.fan_speed.query_current:
                    state.fan_speed.current_checked = True
                    state.fan_speed.current_level = fan_speed
                air_circulation = result.get("air_circulation")
                if isinstance(air_circulation, str):
                    state.recent_air_circulation = air_circulation
                if state.climate_energy_inspection.active:
                    climate_energy = state.climate_energy_inspection
                    climate_energy.climate_checked = True
                    climate_energy.fan_speed = fan_speed
                    if isinstance(airflow, str):
                        climate_energy.fan_airflow_direction = airflow
                    if isinstance(ac_value, bool):
                        climate_energy.air_conditioning = ac_value
                    if isinstance(air_circulation, str):
                        climate_energy.air_circulation = air_circulation
                    front_defrost = result.get("window_front_defrost")
                    if isinstance(front_defrost, bool):
                        climate_energy.window_front_defrost = front_defrost
                    rear_defrost = result.get("window_rear_defrost")
                    if isinstance(rear_defrost, bool):
                        climate_energy.window_rear_defrost = rear_defrost
            elif name == "get_temperature_inside_car" and isinstance(result, dict):
                climate = state.climate_inspection
                climate.temperature_checked = True
                climate.driver_temperature = _safe_float(
                    result.get("climate_temperature_driver"),
                    climate.driver_temperature,
                )
                climate.passenger_temperature = _safe_float(
                    result.get("climate_temperature_passenger"),
                    climate.passenger_temperature,
                )
                state.recent_driver_temperature = climate.driver_temperature
                state.recent_passenger_temperature = climate.passenger_temperature
                occupancy = state.occupancy_comfort
                occupancy.temperature_checked = True
                occupancy.driver_temperature = _safe_float(
                    result.get("climate_temperature_driver"),
                    occupancy.driver_temperature,
                )
                occupancy.passenger_temperature = _safe_float(
                    result.get("climate_temperature_passenger"),
                    occupancy.passenger_temperature,
                )
                unit = result.get("temperature_unit")
                if isinstance(unit, str) and unit.strip():
                    climate.temperature_unit = unit.strip()
            elif name == "get_seat_heating_level" and isinstance(result, dict):
                climate = state.climate_inspection
                climate.seat_heating_checked = True
                climate.seat_heating_driver = _safe_int(
                    result.get("seat_heating_driver"),
                    climate.seat_heating_driver,
                )
                climate.seat_heating_passenger = _safe_int(
                    result.get("seat_heating_passenger"),
                    climate.seat_heating_passenger,
                )
                occupancy = state.occupancy_comfort
                occupancy.seat_heating_checked = True
                occupancy.seat_heating_driver = _safe_int(
                    result.get("seat_heating_driver"),
                    occupancy.seat_heating_driver,
                )
                occupancy.seat_heating_passenger = _safe_int(
                    result.get("seat_heating_passenger"),
                    occupancy.seat_heating_passenger,
                )
                passenger_match = state.passenger_comfort_match
                passenger_match.seat_heating_checked = True
                passenger_match.driver_heating_level = _safe_int(
                    result.get("seat_heating_driver"),
                    passenger_match.driver_heating_level,
                )
                passenger_match.passenger_heating_level = _safe_int(
                    result.get("seat_heating_passenger"),
                    passenger_match.passenger_heating_level,
                )
            elif name == "open_close_window" and isinstance(result, dict):
                state.window.completed = True
                state.window.target_percentage = _safe_int(
                    result.get("percentage"), state.window.target_percentage
                )
                window = result.get("window")
                if isinstance(window, str):
                    state.window.window = window
                    percentage = _safe_int(result.get("percentage"))
                    if percentage is not None:
                        _record_recent_window_position(state, window, percentage)
                        _record_defrost_window_position(
                            state.defrost, window, percentage
                        )
                        _record_air_conditioning_window_position(
                            state.air_conditioning, window, percentage
                        )
                        if state.air_conditioning.active and percentage == 0:
                            state.air_conditioning.closed_windows_for_ac = True
                        _record_window_match_position(
                            state.window_match, window, percentage
                        )
            elif name == "set_ambient_lights" and isinstance(result, dict):
                state.ambient_light.completed = True
                color = result.get("lightcolor")
                if isinstance(color, str):
                    state.ambient_light.target_color = color
                on_value = result.get("on")
                if isinstance(on_value, bool):
                    state.ambient_light.on = on_value
            elif name == "open_close_trunk_door" and isinstance(result, dict):
                state.trunk.completed = True
                action = result.get("action")
                if action in {"OPEN", "CLOSE"}:
                    state.trunk.action = action
            elif name == "set_air_circulation" and isinstance(result, dict):
                state.air_circulation.completed = True
                mode = result.get("mode")
                if mode in {"FRESH_AIR", "RECIRCULATION", "AUTO"}:
                    state.air_circulation.mode = mode
                    state.recent_air_circulation = mode
            elif name == "get_exterior_lights_status" and isinstance(result, dict):
                state.high_beam.exterior_lights_checked = True
                fog_lights = result.get("fog_lights")
                if isinstance(fog_lights, bool):
                    state.high_beam.fog_lights_on = fog_lights
                    state.fog_lights.fog_lights_on = fog_lights
                if state.fog_lights.active:
                    state.fog_lights.exterior_lights_checked = True
                    low_beams = result.get("head_lights_low_beams")
                    if isinstance(low_beams, bool):
                        state.fog_lights.low_beams_on = low_beams
                    high_beams = result.get("head_lights_high_beams")
                    if isinstance(high_beams, bool):
                        state.fog_lights.high_beams_on = high_beams
            elif name == "set_head_lights_low_beams" and isinstance(result, dict):
                on_value = result.get("on")
                if isinstance(on_value, bool):
                    state.fog_lights.low_beams_on = on_value
            elif name == "set_head_lights_high_beams" and isinstance(result, dict):
                state.high_beam.completed = True
                on_value = result.get("on")
                if isinstance(on_value, bool):
                    state.high_beam.on = on_value
                    state.fog_lights.high_beams_on = on_value
            elif name == "set_fog_lights" and isinstance(result, dict):
                state.fog_lights.completed = True
                on_value = result.get("on")
                if isinstance(on_value, bool):
                    state.fog_lights.on = on_value
            elif name == "get_vehicle_window_positions" and isinstance(result, dict):
                state.defrost.windows_checked = True
                state.defrost.window_driver_position = _safe_int(
                    result.get("window_driver_position"),
                    state.defrost.window_driver_position,
                )
                state.window_match.window_driver_position = _safe_int(
                    result.get("window_driver_position"),
                    state.window_match.window_driver_position,
                )
                state.air_conditioning.window_driver_position = _safe_int(
                    result.get("window_driver_position"),
                    state.air_conditioning.window_driver_position,
                )
                state.defrost.window_passenger_position = _safe_int(
                    result.get("window_passenger_position"),
                    state.defrost.window_passenger_position,
                )
                state.window_match.window_passenger_position = _safe_int(
                    result.get("window_passenger_position"),
                    state.window_match.window_passenger_position,
                )
                state.air_conditioning.window_passenger_position = _safe_int(
                    result.get("window_passenger_position"),
                    state.air_conditioning.window_passenger_position,
                )
                state.defrost.window_driver_rear_position = _safe_int(
                    result.get("window_driver_rear_position"),
                    state.defrost.window_driver_rear_position,
                )
                state.window_match.window_driver_rear_position = _safe_int(
                    result.get("window_driver_rear_position"),
                    state.window_match.window_driver_rear_position,
                )
                state.air_conditioning.window_driver_rear_position = _safe_int(
                    result.get("window_driver_rear_position"),
                    state.air_conditioning.window_driver_rear_position,
                )
                state.defrost.window_passenger_rear_position = _safe_int(
                    result.get("window_passenger_rear_position"),
                    state.defrost.window_passenger_rear_position,
                )
                state.window_match.window_passenger_rear_position = _safe_int(
                    result.get("window_passenger_rear_position"),
                    state.window_match.window_passenger_rear_position,
                )
                state.air_conditioning.window_passenger_rear_position = _safe_int(
                    result.get("window_passenger_rear_position"),
                    state.air_conditioning.window_passenger_rear_position,
                )
                if state.window_match.active:
                    state.window_match.windows_checked = True
                if state.air_conditioning.active:
                    state.air_conditioning.windows_checked = True
            elif name == "set_fan_speed" and isinstance(result, dict):
                level = _safe_int(result.get("level"), state.defrost.fan_speed)
                if level is not None:
                    state.defrost.fan_speed = level
                    state.fan_speed.level = level
                    state.air_conditioning.fan_speed = level
                    if (
                        state.air_circulation.active
                        and state.air_circulation.combined_with_air_conditioning
                    ):
                        state.air_circulation.combined_fan_level = level
                if state.fan_speed.active:
                    state.fan_speed.completed = True
            elif name == "set_fan_airflow_direction" and isinstance(result, dict):
                direction = result.get("direction")
                if isinstance(direction, str):
                    state.defrost.fan_airflow_direction = direction
            elif name == "set_air_conditioning" and isinstance(result, dict):
                on_value = result.get("on")
                if isinstance(on_value, bool):
                    state.defrost.air_conditioning = on_value
                    state.air_conditioning.air_conditioning = on_value
                    state.air_conditioning.on = on_value
                    state.recent_air_conditioning = on_value
                if state.air_conditioning.active:
                    state.air_conditioning.completed = True
            elif name == "set_window_defrost" and isinstance(result, dict):
                state.defrost.completed = True
                on_value = result.get("on")
                if isinstance(on_value, bool):
                    state.defrost.on = on_value
                defrost_window = result.get("defrost_window")
                if defrost_window in {"ALL", "FRONT", "REAR"}:
                    state.defrost.defrost_window = defrost_window
            elif name == "set_steering_wheel_heating" and isinstance(result, dict):
                if state.steering_wheel_heating.active:
                    state.steering_wheel_heating.completed = True
                if state.occupancy_comfort.active:
                    state.occupancy_comfort.steering_wheel_set = True
                if state.passenger_comfort_match.active:
                    state.passenger_comfort_match.steering_wheel_set = True
                level = _safe_int(
                    result.get("level"), state.steering_wheel_heating.level
                )
                if level is not None:
                    state.steering_wheel_heating.level = level
                    if state.occupancy_comfort.active:
                        state.occupancy_comfort.steering_wheel_level = level
                    if state.passenger_comfort_match.active:
                        state.passenger_comfort_match.passenger_heating_level = (
                            state.passenger_comfort_match.passenger_heating_level
                            if state.passenger_comfort_match.passenger_heating_level
                            is not None
                            else level
                        )
            elif name == "set_climate_temperature" and isinstance(result, dict):
                if state.cabin_temperature.active:
                    state.cabin_temperature.completed = True
                    temperature = _safe_float(
                        result.get("temperature"),
                        state.cabin_temperature.target_temperature,
                    )
                    if temperature is not None:
                        state.cabin_temperature.target_temperature = temperature
                if state.occupancy_comfort.active:
                    state.occupancy_comfort.temperature_set = True
                    temperature = _safe_float(
                        result.get("temperature"),
                        state.occupancy_comfort.target_temperature,
                    )
                    if temperature is not None:
                        state.occupancy_comfort.target_temperature = temperature
                if state.driver_comfort_temperature.active:
                    state.driver_comfort_temperature.temperature_set = True
                    temperature = _safe_float(
                        result.get("temperature"),
                        state.driver_comfort_temperature.target_temperature,
                    )
                    if temperature is not None:
                        state.driver_comfort_temperature.target_temperature = (
                            temperature
                        )
                if state.passenger_comfort_match.active:
                    state.passenger_comfort_match.temperature_set = True
                    temperature = _safe_float(
                        result.get("temperature"),
                        state.passenger_comfort_match.target_temperature,
                    )
                    if temperature is not None:
                        state.passenger_comfort_match.target_temperature = temperature
            elif name == "set_seat_heating" and isinstance(result, dict):
                if state.occupancy_comfort.active:
                    seat_zone = result.get("seat_zone")
                    level = _safe_int(
                        result.get("level"),
                        state.occupancy_comfort.target_heating_level,
                    )
                    unoccupied_adjustment = (
                        level == 0
                        and _seat_zone_is_unoccupied_front(
                            state.occupancy_comfort, seat_zone
                        )
                    )
                    if unoccupied_adjustment:
                        state.occupancy_comfort.unoccupied_heating_set = True
                    else:
                        state.occupancy_comfort.seat_heating_set = True
                    if level is not None and not unoccupied_adjustment:
                        state.occupancy_comfort.target_heating_level = level
                if state.driver_comfort_temperature.active:
                    seat_zone = result.get("seat_zone")
                    level = _safe_int(result.get("level"))
                    if seat_zone == "PASSENGER" and level == 0:
                        state.driver_comfort_temperature.passenger_heating_off = True
                if state.passenger_comfort_match.active:
                    seat_zone = result.get("seat_zone")
                    level = _safe_int(
                        result.get("level"),
                        state.passenger_comfort_match.driver_heating_level,
                    )
                    if seat_zone == "PASSENGER" or seat_zone is None:
                        state.passenger_comfort_match.passenger_heating_set = True
                    if level is not None:
                        state.passenger_comfort_match.passenger_heating_level = level
            elif name == "set_reading_light" and isinstance(result, dict):
                if state.reading_light_occupancy.active:
                    if state.reading_light_occupancy.pending_actions:
                        state.reading_light_occupancy.pending_actions.pop(0)
                    if not state.reading_light_occupancy.pending_actions:
                        state.reading_light_occupancy.completed = True
                if state.reading_light.active:
                    state.reading_light.completed = True
                position = result.get("position")
                if isinstance(position, str):
                    state.reading_light.position = position
            elif name == "get_seats_occupancy" and isinstance(result, dict):
                seats = result.get("seats_occupied")
                if state.reading_light_occupancy.active:
                    state.reading_light_occupancy.seats_checked = True
                    if isinstance(seats, dict):
                        state.reading_light_occupancy.seats_occupied = {
                            str(seat): bool(occupied)
                            for seat, occupied in seats.items()
                        }
                if state.occupancy_comfort.active:
                    state.occupancy_comfort.seats_checked = True
                    if isinstance(seats, dict):
                        state.occupancy_comfort.seats_occupied = {
                            str(seat): bool(occupied)
                            for seat, occupied in seats.items()
                        }
                if state.climate_energy_inspection.active:
                    state.climate_energy_inspection.seats_checked = True
                    if isinstance(seats, dict):
                        state.climate_energy_inspection.seats_occupied = {
                            str(seat): bool(occupied)
                            for seat, occupied in seats.items()
                        }
            elif name == "get_current_navigation_state":
                if state.route_energy.active:
                    route_energy = state.route_energy
                    route_energy.current_navigation_checked = True
                    if isinstance(result, dict):
                        active = result.get("navigation_active")
                        if isinstance(active, bool):
                            route_energy.navigation_active = active
                        waypoints = result.get("waypoints_id")
                        if isinstance(waypoints, list):
                            route_energy.waypoints_id = [
                                str(waypoint)
                                for waypoint in waypoints
                                if isinstance(waypoint, str)
                            ]
                        routes = result.get("routes_to_final_destination_id")
                        if isinstance(routes, list):
                            route_energy.routes_to_final_destination_id = [
                                str(route) for route in routes if isinstance(route, str)
                            ]
                        details = result.get("details")
                        if isinstance(details, dict):
                            route_details = details.get("routes")
                            if isinstance(route_details, list):
                                route_energy.route_details = [
                                    route
                                    for route in route_details
                                    if isinstance(route, dict)
                                ]
                if state.navigation.active:
                    state.navigation.current_navigation_checked = True
                    if isinstance(result, dict):
                        active = result.get("navigation_active")
                        if isinstance(active, bool):
                            state.navigation.navigation_active = active
                        waypoints = result.get("waypoints_id")
                        if isinstance(waypoints, list):
                            state.navigation.waypoints_id = [
                                str(waypoint)
                                for waypoint in waypoints
                                if isinstance(waypoint, str)
                            ]
                        routes = result.get("routes_to_final_destination_id")
                        if isinstance(routes, list):
                            state.navigation.routes_to_final_destination_id = [
                                str(route) for route in routes if isinstance(route, str)
                            ]
                        details = result.get("details")
                        if isinstance(details, dict):
                            waypoint_details = details.get("waypoints")
                            if isinstance(waypoint_details, list):
                                state.navigation.waypoint_details = [
                                    waypoint
                                    for waypoint in waypoint_details
                                    if isinstance(waypoint, dict)
                                ]
                        _resolve_navigation_waypoint_from_state(state.navigation)
                if state.poi.active and state.poi.search_along_route:
                    state.poi.current_navigation_checked = True
                    if isinstance(result, dict):
                        waypoints = result.get("waypoints_id")
                        waypoint_ids = (
                            [
                                str(waypoint)
                                for waypoint in waypoints
                                if isinstance(waypoint, str)
                            ]
                            if isinstance(waypoints, list)
                            else []
                        )
                        routes = result.get("routes_to_final_destination_id")
                        if isinstance(routes, list):
                            route_ids = [
                                str(route) for route in routes if isinstance(route, str)
                            ]
                            if route_ids:
                                state.poi.route_id = route_ids[-1]
                                if len(waypoint_ids) >= len(route_ids) + 1:
                                    index = len(route_ids) - 1
                                    state.poi.route_start_id = waypoint_ids[index]
                                    state.poi.route_end_id = waypoint_ids[index + 1]
            elif name == "get_entries_from_calendar":
                if isinstance(result, dict):
                    meetings = result.get("meetings")
                    if isinstance(meetings, list):
                        state.recent_calendar_meetings = [
                            meeting for meeting in meetings if isinstance(meeting, dict)
                        ]
                    if state.email.active and state.email.mode == "meeting_delay":
                        state.email.calendar_checked = True
                        _record_email_meeting_status(state)
                    elif (
                        state.email.active
                        and state.email.mode == "meeting_attendees"
                    ):
                        state.email.calendar_checked = True
            elif name == "get_location_id_by_location_name":
                if isinstance(result, dict) and isinstance(result.get("id"), str):
                    if state.pending_location_lookup_name:
                        state.recent_location_lookup_name = (
                            state.pending_location_lookup_name
                        )
                        state.recent_location_lookup_id = result["id"]
                        if (
                            state.email.active
                            and state.email.weather_requested
                            and _locations_match(
                                state.email.weather_location_name,
                                state.pending_location_lookup_name,
                            )
                        ):
                            state.email.weather_location_id = result["id"]
                        state.pending_location_lookup_name = None
                    if state.multi_stop_navigation.active:
                        multi_stop = state.multi_stop_navigation
                        if multi_stop.first_destination_id is None:
                            multi_stop.first_destination_id = result["id"]
                        elif multi_stop.final_destination_id is None:
                            multi_stop.final_destination_id = result["id"]
                    if state.poi.active and state.poi.location_id is None:
                        state.poi.location_id = result["id"]
                    if (
                        state.route_energy.active
                        and state.route_energy.destination_id is None
                    ):
                        state.route_energy.destination_id = result["id"]
                    if state.navigation.active:
                        if state.navigation.mode == "replace_one_waypoint":
                            state.navigation.new_waypoint_id = result["id"]
                        else:
                            state.navigation.destination_id = result["id"]
                elif (
                    state.email.active
                    and state.email.weather_requested
                    and state.pending_location_lookup_name
                ):
                    state.pending_location_lookup_name = None
                elif state.multi_stop_navigation.active:
                    multi_stop = state.multi_stop_navigation
                    destination = (
                        multi_stop.first_destination_name
                        if multi_stop.first_destination_id is None
                        else multi_stop.final_destination_name
                    ) or "that destination"
                    multi_stop.failure_message = (
                        f"I couldn't find a location ID for {destination}."
                    )
                elif state.poi.active and state.poi.location_id is None:
                    location = state.poi.location_name or "that location"
                    state.poi.failure_message = f"I couldn't find {location}."
                elif (
                    state.route_energy.active
                    and state.route_energy.destination_id is None
                ):
                    destination = state.route_energy.destination_name or "that destination"
                    state.route_energy.failure_message = (
                        f"I couldn't find a location ID for {destination}."
                    )
                elif state.navigation.active:
                    destination = (
                        state.navigation.new_waypoint_name
                        if state.navigation.mode == "replace_one_waypoint"
                        else state.navigation.destination_name
                    ) or "that destination"
                    state.navigation.failure_message = (
                        f"I couldn't find a location ID for {destination}."
                    )
            elif name == "search_poi_at_location":
                if state.route_energy.active and state.route_energy.search_mode == "nearby":
                    state.route_energy.pois_checked = True
                    if isinstance(result, dict) and isinstance(
                        result.get("pois_found"), list
                    ):
                        state.route_energy.pois = [
                            poi
                            for poi in result["pois_found"]
                            if isinstance(poi, dict)
                        ]
                        state.recent_charging_pois = list(state.route_energy.pois)
                        state.recent_charging_route_id = None
                        state.recent_charging_at_kilometer = None
                    if not state.route_energy.pois:
                        state.route_energy.failure_message = (
                            "I couldn't find matching charging stations there."
                        )
                elif state.poi.active:
                    state.poi.pois_checked = True
                    if isinstance(result, dict) and isinstance(
                        result.get("pois_found"), list
                    ):
                        state.poi.pois = [
                            poi
                            for poi in result["pois_found"]
                            if isinstance(poi, dict)
                        ]
                        if state.poi.required_open_at_minutes is not None:
                            state.poi.pois = [
                                poi
                                for poi in state.poi.pois
                                if _poi_open_at_minutes(
                                    poi, state.poi.required_open_at_minutes
                                )
                            ]
                    if not state.poi.pois:
                        state.poi.failure_message = (
                            "I couldn't find matching places there."
                        )
            elif name == "search_poi_along_the_route" and state.poi.active:
                state.poi.pois_checked = True
                if isinstance(result, dict) and isinstance(
                    result.get("pois_found_along_route"), list
                ):
                    state.poi.pois = [
                        poi
                        for poi in result["pois_found_along_route"]
                        if isinstance(poi, dict)
                    ]
                if not state.poi.pois:
                    state.poi.failure_message = (
                        "I couldn't find matching places along the route."
                    )
            elif name == "get_contact_id_by_contact_name":
                if state.email.active:
                    _record_email_contact_matches(state.email, result)
            elif name == "get_contact_information":
                if state.email.active:
                    _record_email_contact_information(state.email, result)
            elif name == "get_routes_from_start_to_destination":
                if isinstance(result, dict) and isinstance(result.get("routes"), list):
                    routes = [
                        route for route in result["routes"] if isinstance(route, dict)
                    ]
                    if state.route_energy.active:
                        route_energy = state.route_energy
                        if route_energy.route_lookup == "setup_to_poi":
                            route_energy.setup_to_poi_checked = True
                            route_energy.routes_to_poi = routes
                        elif route_energy.route_lookup == "setup_from_poi":
                            route_energy.setup_from_poi_checked = True
                            route_energy.routes_from_poi = routes
                        else:
                            route_energy.routes_checked = True
                            route_energy.routes = routes
                    elif state.multi_stop_navigation.active:
                        multi_stop = state.multi_stop_navigation
                        if not multi_stop.first_routes_checked:
                            multi_stop.first_routes_checked = True
                            multi_stop.first_routes = routes
                        else:
                            multi_stop.final_routes_checked = True
                            multi_stop.final_routes = routes
                    elif state.poi.active and state.poi.selected_poi_id is not None:
                        if state.poi.route_lookup == "from_poi":
                            state.poi.routes_from_poi_checked = True
                            state.poi.routes_from_poi = routes
                        else:
                            state.poi.routes_checked = True
                            state.poi.routes = routes
                    else:
                        _record_navigation_routes(state.navigation, routes)
                    if not routes:
                        if state.poi.active and state.poi.selected_poi_id is not None:
                            state.poi.failure_message = (
                                "I couldn't find a route to that destination."
                            )
                        elif state.route_energy.active:
                            state.route_energy.failure_message = (
                                "I couldn't find a route to that destination."
                            )
                        elif state.multi_stop_navigation.active:
                            state.multi_stop_navigation.failure_message = (
                                "I couldn't find a route to that destination."
                            )
                        else:
                            state.navigation.failure_message = (
                                "I couldn't find a route to that destination."
                            )
                elif state.poi.active and state.poi.selected_poi_id is not None:
                    if state.poi.route_lookup == "from_poi":
                        state.poi.routes_from_poi_checked = True
                        state.poi.routes_from_poi = []
                    else:
                        state.poi.routes_checked = True
                        state.poi.routes = []
                    state.poi.failure_message = (
                        "I can't determine the route options from the available route result."
                    )
                elif state.route_energy.active:
                    route_energy = state.route_energy
                    if route_energy.route_lookup == "setup_to_poi":
                        route_energy.setup_to_poi_checked = True
                        route_energy.routes_to_poi = []
                    elif route_energy.route_lookup == "setup_from_poi":
                        route_energy.setup_from_poi_checked = True
                        route_energy.routes_from_poi = []
                    else:
                        route_energy.routes_checked = True
                        route_energy.routes = []
                    route_energy.failure_message = (
                        "I can't determine the route options from the available route result."
                    )
                elif state.multi_stop_navigation.active:
                    multi_stop = state.multi_stop_navigation
                    if not multi_stop.first_routes_checked:
                        multi_stop.first_routes_checked = True
                        multi_stop.first_routes = []
                    else:
                        multi_stop.final_routes_checked = True
                        multi_stop.final_routes = []
                    multi_stop.failure_message = (
                        "I can't determine the route options from the available route result."
                    )
                elif state.navigation.active:
                    _record_navigation_routes(state.navigation, [])
                    state.navigation.failure_message = (
                        "I can't determine the route options from the available route result."
                    )
            elif name == "get_charging_specs_and_status":
                if state.route_energy.active:
                    charging = state.route_energy
                    charging.charging_status_checked = True
                    if isinstance(result, dict):
                        state_of_charge = _safe_float(result.get("state_of_charge"))
                        if state_of_charge is not None:
                            charging.state_of_charge = state_of_charge
                            if charging.initial_soc is None:
                                charging.initial_soc = state_of_charge
                        remaining_range = _extract_km_value(result.get("remaining_range"))
                        if remaining_range is not None:
                            charging.remaining_range_km = remaining_range
            elif name == "get_distance_by_soc":
                if state.route_energy.active:
                    charging = state.route_energy
                    charging.distance_checked = True
                    if isinstance(result, dict):
                        charging.distance_km = _extract_first_km_value(result)
            elif name == "search_poi_along_the_route":
                if state.route_energy.active:
                    charging = state.route_energy
                    poi_category = charging.poi_search_category
                    if isinstance(result, dict) and isinstance(
                        result.get("pois_found_along_route"), list
                    ):
                        pois = [
                            poi
                            for poi in result["pois_found_along_route"]
                            if isinstance(poi, dict)
                        ]
                    else:
                        pois = []

                    if (
                        poi_category is not None
                        and poi_category == charging.companion_poi_category
                    ):
                        charging.companion_pois_checked = True
                        charging.companion_pois = (
                            _filter_companion_pois_for_route_constraints(
                                charging, state, pois
                            )
                        )
                        if (
                            not charging.companion_pois
                            and not _advance_route_energy_search_kilometer(charging)
                        ):
                            label = _friendly_poi_category(poi_category)
                            charging.failure_message = (
                                f"I couldn't find matching {label} near a charging stop on the route."
                            )
                    else:
                        charging.pois_checked = True
                        charging.pois = pois
                        state.recent_charging_pois = list(charging.pois)
                        state.recent_charging_route_id = charging.route_id
                        state.recent_charging_at_kilometer = charging.at_kilometer
                        if not charging.pois:
                            if not _advance_route_energy_search_kilometer(charging):
                                charging.failure_message = (
                                    "I couldn't find matching charging stations along the route."
                                )
                    charging.poi_search_category = None
            elif name == "calculate_charging_time_by_soc":
                if state.route_energy.active:
                    charging = state.route_energy
                    charging.charging_time_checked = True
                    if isinstance(result, dict):
                        charging.charging_minutes = _extract_minutes_value(result)
            elif name == "call_phone_by_number":
                if state.route_energy.active:
                    state.route_energy.call_completed = True
            elif name in {
                "set_new_navigation",
                "navigation_add_one_waypoint",
                "navigation_replace_final_destination",
                "navigation_delete_destination",
                "navigation_delete_waypoint",
                "navigation_replace_one_waypoint",
            } and isinstance(result, dict):
                if state.route_energy.active and name == "set_new_navigation":
                    state.route_energy.completed = True
                elif (
                    state.multi_stop_navigation.active and name == "set_new_navigation"
                ):
                    state.multi_stop_navigation.completed = True
                elif state.poi.active and name in {
                    "set_new_navigation",
                    "navigation_add_one_waypoint",
                    "navigation_replace_final_destination",
                }:
                    state.poi.completed = True
                else:
                    state.navigation.completed = True
                if state.poi.active and state.poi.replace_final_destination:
                    state.poi.completed = True
                elif state.poi.active and state.poi.completed:
                    state.planned_navigation_route_ids = []
                    state.planned_navigation_waypoints = []
                    state.planned_navigation_selected_routes = []
                    state.planned_navigation_considered_routes = []
                waypoints = result.get("new_waypoints")
                if not isinstance(waypoints, list):
                    waypoints = result.get("new_waypoints_id")
                if not isinstance(waypoints, list):
                    waypoints = result.get("waypoints")
                if isinstance(waypoints, list):
                    state.navigation.waypoints_id = [
                        str(waypoint)
                        for waypoint in waypoints
                        if isinstance(waypoint, str)
                    ]
                    state.recent_navigation_waypoints = list(
                        state.navigation.waypoints_id
                    )
                routes = result.get("new_routes")
                if not isinstance(routes, list):
                    routes = result.get("new_routes_id")
                if isinstance(routes, list):
                    state.navigation.routes_to_final_destination_id = [
                        str(route) for route in routes if isinstance(route, str)
                    ]
                    state.recent_navigation_route_ids = list(
                        state.navigation.routes_to_final_destination_id
                    )
            elif name == "send_email" and state.email.active:
                if isinstance(payload, dict) and payload.get("status") == "SUCCESS":
                    state.email.completed = True

    def _next_sunroof_action(
        self, state: ControllerState, tool_index: ToolIndex
    ) -> NextAction | None:
        sunroof = state.sunroof

        if sunroof.completed:
            target = _format_percentage(sunroof.sunroof_position)
            shade = _format_percentage(sunroof.sunshade_position)
            return NextAction.respond(
                f"Done, the sunroof is open to {target} and the sunshade is at {shade}.",
                reason="sunroof_done",
            )

        if sunroof.target_percentage is None:
            if not sunroof.preferences_checked and tool_index.has(
                "get_user_preferences"
            ):
                return NextAction.tool_call(
                    "get_user_preferences",
                    {
                        "preference_categories": {
                            "vehicle_settings": {"vehicle_settings": True}
                        }
                    },
                    reason="sunroof_internal_disambiguation_preferences",
                )
            return NextAction.respond(
                "How far would you like me to open the sunroof?",
                reason="sunroof_user_disambiguation",
            )

        if not sunroof.position_checked and tool_index.has(
            "get_sunroof_and_sunshade_position"
        ):
            return NextAction.tool_call(
                "get_sunroof_and_sunshade_position",
                reason="sunroof_precondition_state_check",
            )

        if sunroof.target_percentage > 0 and not sunroof.weather_checked:
            if not tool_index.has("get_weather"):
                return NextAction.respond(
                    "I can't safely open the sunroof because the weather check is unavailable right now.",
                    reason="sunroof_missing_weather_tool",
                )
            weather_args = _weather_arguments(state.runtime)
            if weather_args is None:
                return NextAction.respond(
                    "I need the current location and time before I can safely open the sunroof.",
                    reason="sunroof_missing_weather_context",
                )
            return NextAction.tool_call(
                "get_weather",
                weather_args,
                reason="sunroof_weather_check",
            )

        if (
            sunroof.target_percentage > 0
            and _requires_weather_confirmation(sunroof.weather_condition)
            and not sunroof.weather_confirmed
        ):
            sunroof.weather_confirmation_requested = True
            condition = _friendly_weather(sunroof.weather_condition)
            return NextAction.respond(
                f"It's {condition} right now, so opening the sunroof needs your confirmation. Do you still want me to open it?",
                reason="sunroof_bad_weather_confirmation",
            )

        if sunroof.target_percentage > 0 and (sunroof.sunshade_position or 0) < 100:
            if not tool_index.has("open_close_sunshade"):
                return NextAction.respond(
                    "I can't open the sunroof safely because the sunshade control is unavailable, and the sunshade needs to be fully open first.",
                    reason="sunroof_missing_sunshade_tool",
                )
            return NextAction.tool_call(
                "open_close_sunshade",
                {"percentage": 100},
                reason="sunroof_open_sunshade_first",
            )

        if not tool_index.has("open_close_sunroof"):
            return NextAction.respond(
                "I can't open the sunroof because the sunroof control is unavailable right now.",
                reason="sunroof_missing_sunroof_tool",
            )

        return NextAction.tool_call(
            "open_close_sunroof",
            {"percentage": sunroof.target_percentage},
            reason="sunroof_open",
        )

    def _next_sunshade_action(
        self, state: ControllerState, tool_index: ToolIndex
    ) -> NextAction | None:
        sunshade = state.sunshade

        if sunshade.completed:
            target = _format_percentage(sunshade.target_percentage)
            state.sunshade = SunshadeFlow()
            return NextAction.respond(
                f"Done, the sunshade is set to {target}.",
                reason="sunshade_done",
            )

        if not tool_index.has("open_close_sunshade"):
            return NextAction.respond(
                "I can't adjust the sunshade because that control is unavailable right now.",
                reason="sunshade_missing_tool",
            )

        if not _tool_argument_available(
            tool_index, "open_close_sunshade", "percentage"
        ):
            return NextAction.respond(
                "I can't adjust the sunshade because the required position control is unavailable right now.",
                reason="sunshade_missing_percentage_parameter",
            )

        if sunshade.target_percentage is None:
            return NextAction.respond(
                "How far would you like me to set the sunshade?",
                reason="sunshade_user_disambiguation",
            )

        return NextAction.tool_call(
            "open_close_sunshade",
            {"percentage": sunshade.target_percentage},
            reason="sunshade_set",
        )

    def _next_window_action(
        self, state: ControllerState, tool_index: ToolIndex
    ) -> NextAction | None:
        window = state.window

        if window.completed:
            verb = "are" if window.window == "ALL" else "is"
            return NextAction.respond(
                f"Done, {friendly_window_name(window.window)} {verb} open to {_format_percentage(window.target_percentage)}.",
                reason="window_done",
            )

        if not tool_index.has("open_close_window"):
            return NextAction.respond(
                "I can't move the windows because the window control is unavailable right now.",
                reason="window_missing_tool",
            )

        window_args = tool_index.arg_names("open_close_window")
        if "window" not in window_args or "percentage" not in window_args:
            return NextAction.respond(
                "I can't set the windows safely because the required window position controls are unavailable right now.",
                reason="window_missing_required_parameter",
            )

        if window.target_percentage is None:
            return NextAction.respond(
                "How far should I open the windows?",
                reason="window_user_disambiguation",
            )

        if window.target_percentage > 25 and not window.climate_checked:
            if tool_index.has("get_climate_settings"):
                return NextAction.tool_call(
                    "get_climate_settings",
                    reason="window_ac_policy_check",
                )

        if (
            window.target_percentage > 25
            and window.ac_on is True
            and not window.ac_confirmed
        ):
            window.ac_confirmation_requested = True
            return NextAction.respond(
                "The air conditioning is on, so opening the windows that far can waste energy. Do you still want me to open them?",
                reason="window_ac_confirmation",
            )

        return NextAction.tool_call(
            "open_close_window",
            {"window": window.window, "percentage": window.target_percentage},
            reason="window_open",
        )

    def _next_window_match_action(
        self, state: ControllerState, tool_index: ToolIndex
    ) -> NextAction | None:
        match = state.window_match

        if match.completed:
            state.window_match = WindowMatchFlow()
            return NextAction.respond(
                _window_match_completion_message(match),
                reason="window_match_done",
            )

        if not tool_index.has("open_close_window"):
            return NextAction.respond(
                "I can't adjust the windows because the window control is unavailable right now.",
                reason="window_match_missing_window_tool",
            )

        if not match.windows_checked:
            if not tool_index.has("get_vehicle_window_positions"):
                return NextAction.respond(
                    "I can't match the windows because the window position check is unavailable right now.",
                    reason="window_match_missing_position_tool",
                )
            return NextAction.tool_call(
                "get_vehicle_window_positions",
                reason="window_match_position_check",
            )

        if (
            match.window_driver_position is None
            or match.window_passenger_position is None
            or match.window_driver_rear_position is None
            or match.window_passenger_rear_position is None
        ):
            return NextAction.respond(
                "I can't match the rear windows because I couldn't determine all current window positions.",
                reason="window_match_unknown_positions",
            )

        if match.reference_window == "PASSENGER_REAR":
            target = match.window_passenger_rear_position
            if match.followup_defrost_window is not None and not state.defrost.climate_checked:
                if not tool_index.has("get_climate_settings"):
                    return NextAction.respond(
                        "I can't safely continue to defrost because the climate settings check is unavailable right now.",
                        reason="window_match_defrost_missing_climate_tool",
                    )
                return NextAction.tool_call(
                    "get_climate_settings",
                    reason="window_match_defrost_climate_check",
                )
            if not _tool_argument_available(tool_index, "open_close_window", "window"):
                return NextAction.respond(
                    "I can't match the windows because the required window selector is unavailable right now.",
                    reason="window_match_missing_window_parameter",
                )
            if not _tool_argument_available(tool_index, "open_close_window", "percentage"):
                return NextAction.respond(
                    "I can't match the windows because the required window position control is unavailable right now.",
                    reason="window_match_missing_percentage_parameter",
                )
            if any(
                position != target
                for position in (
                    match.window_driver_position,
                    match.window_passenger_position,
                    match.window_driver_rear_position,
                    match.window_passenger_rear_position,
                )
            ):
                return NextAction.tool_call(
                    "open_close_window",
                    {"window": "ALL", "percentage": target},
                    reason="window_match_all_to_passenger_rear",
                )
            if match.followup_defrost_window is not None:
                state.defrost.active = True
                state.defrost.defrost_window = match.followup_defrost_window
                state.defrost.windows_checked = True
                state.window_match = WindowMatchFlow()
                return self._next_defrost_action(state, tool_index)
            match.completed = True
            return self._next_window_match_action(state, tool_index)

        if match.window_driver_position != match.window_passenger_position:
            return NextAction.respond(
                "The two front windows are at different positions. Which front window should the rear windows match?",
                reason="window_match_ambiguous_front_reference",
            )

        if not _tool_argument_available(tool_index, "open_close_window", "window"):
            return NextAction.respond(
                "I can't match the windows because the required window selector is unavailable right now.",
                reason="window_match_missing_window_parameter",
            )

        if not _tool_argument_available(tool_index, "open_close_window", "percentage"):
            return NextAction.respond(
                "I can't match the windows because the required window position control is unavailable right now.",
                reason="window_match_missing_percentage_parameter",
            )

        target = match.window_driver_position
        if match.window_driver_rear_position != target:
            return NextAction.tool_call(
                "open_close_window",
                {"window": "DRIVER_REAR", "percentage": target},
                reason="window_match_driver_rear",
            )

        if match.window_passenger_rear_position != target:
            return NextAction.tool_call(
                "open_close_window",
                {"window": "PASSENGER_REAR", "percentage": target},
                reason="window_match_passenger_rear",
            )

        match.completed = True
        return self._next_window_match_action(state, tool_index)

    def _next_ambient_light_action(
        self, state: ControllerState, tool_index: ToolIndex
    ) -> NextAction | None:
        ambient = state.ambient_light

        if ambient.completed:
            if ambient.on:
                return NextAction.respond(
                    f"Done, the ambient lights are set to {ambient.target_color}.",
                    reason="ambient_light_done",
                )
            return NextAction.respond(
                "Done, the ambient lights are off.",
                reason="ambient_light_done",
            )

        if not tool_index.has("set_ambient_lights"):
            return NextAction.respond(
                "I can't change the ambient lights because that control is unavailable right now.",
                reason="ambient_light_missing_tool",
            )

        ambient_args = tool_index.arg_names("set_ambient_lights")
        if "on" not in ambient_args or "lightcolor" not in ambient_args:
            return NextAction.respond(
                "I can't set the ambient light color because that required control is unavailable right now.",
                reason="ambient_light_missing_required_parameter",
            )

        if not ambient.on:
            return NextAction.tool_call(
                "set_ambient_lights",
                {"on": False, "lightcolor": "NONE"},
                reason="ambient_light_off",
            )

        if ambient.match_car_color and ambient.target_color is None:
            if not ambient.car_color_checked:
                if not tool_index.has("get_car_color"):
                    return NextAction.respond(
                        "I can't match the ambient lights to the car color because the car color check is unavailable right now.",
                        reason="ambient_light_missing_car_color_tool",
                    )
                return NextAction.tool_call(
                    "get_car_color",
                    reason="ambient_light_car_color_check",
                )
            return NextAction.respond(
                "I couldn't determine a supported ambient light color from the car color.",
                reason="ambient_light_unknown_car_color",
            )

        if ambient.target_color is None:
            if not ambient.preferences_checked and tool_index.has(
                "get_user_preferences"
            ):
                return NextAction.tool_call(
                    "get_user_preferences",
                    {
                        "preference_categories": {
                            "vehicle_settings": {"vehicle_settings": True}
                        }
                    },
                    reason="ambient_light_internal_disambiguation_preferences",
                )
            return NextAction.respond(
                "Which ambient light color would you like?",
                reason="ambient_light_user_disambiguation",
            )

        return NextAction.tool_call(
            "set_ambient_lights",
            {"on": True, "lightcolor": ambient.target_color},
            reason="ambient_light_set",
        )

    def _next_trunk_action(
        self, state: ControllerState, tool_index: ToolIndex
    ) -> NextAction | None:
        trunk = state.trunk

        if trunk.completed:
            return NextAction.respond(
                "Done, the trunk is open."
                if trunk.action == "OPEN"
                else "Done, the trunk is closed.",
                reason="trunk_done",
            )

        if not tool_index.has("open_close_trunk_door"):
            return NextAction.respond(
                "I can't move the trunk door because that control is unavailable right now.",
                reason="trunk_missing_tool",
            )

        if not trunk.confirmed:
            trunk.confirmation_requested = True
            return NextAction.respond(
                "Please confirm if you want me to open the trunk."
                if trunk.action == "OPEN"
                else "Please confirm if you want me to close the trunk.",
                reason="trunk_confirmation",
            )

        return NextAction.tool_call(
            "open_close_trunk_door",
            {"action": trunk.action},
            reason="trunk_action",
        )

    def _next_air_circulation_action(
        self, state: ControllerState, tool_index: ToolIndex
    ) -> NextAction | None:
        air = state.air_circulation

        if air.completed:
            if air.combined_with_air_conditioning:
                completed = []
                window_summary = _format_combined_window_completion(
                    air.combined_window,
                    air.combined_window_percentage,
                )
                if window_summary:
                    completed.append(window_summary)
                completed.append("turned on the air conditioning")
                if air.combined_fan_level is not None:
                    completed.append(
                        f"set the fan speed to level {air.combined_fan_level}"
                    )
                completed.append(
                    "set air circulation to "
                    f"{_friendly_air_mode(air.mode)}"
                )
                return NextAction.respond(
                    f"Done, I {_join_completed_actions(completed)}.",
                    reason="air_circulation_done",
                )
            return NextAction.respond(
                f"Done, air circulation is set to {_friendly_air_mode(air.mode)}.",
                reason="air_circulation_done",
            )

        if not tool_index.has("set_air_circulation"):
            return NextAction.respond(
                "I can't change the air circulation because that control is unavailable right now.",
                reason="air_circulation_missing_tool",
            )

        if "mode" not in tool_index.arg_names("set_air_circulation"):
            return NextAction.respond(
                "I can't change the air circulation mode because that required control is unavailable right now.",
                reason="air_circulation_missing_mode_parameter",
            )

        if air.mode is None:
            if not air.preferences_checked and tool_index.has("get_user_preferences"):
                return NextAction.tool_call(
                    "get_user_preferences",
                    {
                        "preference_categories": {
                            "vehicle_settings": {"climate_control": True}
                        }
                    },
                    reason="air_circulation_internal_disambiguation_preferences",
                )
            return NextAction.respond(
                "Which air circulation mode would you like: fresh air, recirculation, or auto?",
                reason="air_circulation_user_disambiguation",
            )

        return NextAction.tool_call(
            "set_air_circulation",
            {"mode": air.mode},
            reason="air_circulation_set",
        )

    def _next_air_conditioning_action(
        self, state: ControllerState, tool_index: ToolIndex
    ) -> NextAction | None:
        ac = state.air_conditioning

        if ac.completed:
            on = ac.on
            content = _format_air_conditioning_completion(ac)
            state.air_conditioning = AirConditioningFlow()
            return NextAction.respond(
                content
                if on
                else "Done, the air conditioning is off.",
                reason="air_conditioning_done",
            )

        if not tool_index.has("set_air_conditioning"):
            return NextAction.respond(
                "I can't change the air conditioning because that control is unavailable right now.",
                reason="air_conditioning_missing_tool",
            )

        if not _tool_argument_available(tool_index, "set_air_conditioning", "on"):
            return NextAction.respond(
                "I can't change the air conditioning because the required on/off control is unavailable right now.",
                reason="air_conditioning_missing_on_parameter",
            )

        if not ac.on:
            return NextAction.tool_call(
                "set_air_conditioning",
                {"on": False},
                reason="air_conditioning_off",
            )

        if not ac.climate_checked:
            if not tool_index.has("get_climate_settings"):
                return NextAction.respond(
                    "I can't safely turn on the air conditioning because the climate settings check is unavailable right now.",
                    reason="air_conditioning_missing_climate_tool",
                )
            return NextAction.tool_call(
                "get_climate_settings",
                reason="air_conditioning_climate_check",
            )

        if not ac.preserve_open_windows and not ac.windows_checked:
            if not tool_index.has("get_vehicle_window_positions"):
                return NextAction.respond(
                    "I can't safely turn on the air conditioning because the window position check is unavailable right now.",
                    reason="air_conditioning_missing_window_positions_tool",
                )
            return NextAction.tool_call(
                "get_vehicle_window_positions",
                reason="air_conditioning_window_position_check",
            )

        if not ac.preserve_open_windows and _air_conditioning_window_status_unknown(ac):
            return NextAction.respond(
                "I can't safely turn on the air conditioning because I couldn't determine all window positions first.",
                reason="air_conditioning_unknown_window_positions",
            )

        open_windows = (
            [] if ac.preserve_open_windows else _air_conditioning_open_windows(ac)
        )
        if open_windows:
            if not tool_index.has("open_close_window"):
                return NextAction.respond(
                    "I can't safely turn on the air conditioning because open windows need to be closed first and the window control is unavailable.",
                    reason="air_conditioning_missing_window_control",
                )
            if not _tool_argument_available(tool_index, "open_close_window", "window"):
                return NextAction.respond(
                    "I can't safely close the open windows because the required window selector is unavailable right now.",
                    reason="air_conditioning_missing_window_parameter",
                )
            if not _tool_argument_available(
                tool_index, "open_close_window", "percentage"
            ):
                return NextAction.respond(
                    "I can't safely close the open windows because the required window position control is unavailable right now.",
                    reason="air_conditioning_missing_percentage_parameter",
                )
            window = "ALL" if len(open_windows) == 4 else open_windows[0][0]
            return NextAction.tool_call(
                "open_close_window",
                {"window": window, "percentage": 0},
                reason="air_conditioning_close_open_window",
            )

        target_level = ac.fan_target_level
        if target_level is None and ac.fan_speed == 0:
            target_level = 1

        if target_level is not None and ac.fan_speed != target_level:
            if not tool_index.has("set_fan_speed"):
                return NextAction.respond(
                    "I can't safely turn on the air conditioning because the fan speed control is unavailable right now.",
                    reason="air_conditioning_missing_fan_speed_tool",
                )
            if not _tool_argument_available(tool_index, "set_fan_speed", "level"):
                return NextAction.respond(
                    "I can't safely turn on the air conditioning because the required fan speed level control is unavailable right now.",
                    reason="air_conditioning_missing_fan_level_parameter",
                )
            return NextAction.tool_call(
                "set_fan_speed",
                {"level": target_level},
                reason="air_conditioning_set_fan_speed",
            )

        if ac.air_conditioning is True:
            content = _format_air_conditioning_completion(ac)
            state.air_conditioning = AirConditioningFlow()
            return NextAction.respond(
                content,
                reason="air_conditioning_already_on",
            )

        return NextAction.tool_call(
            "set_air_conditioning",
            {"on": True},
            reason="air_conditioning_on",
        )

    def _next_air_quality_action(
        self, state: ControllerState, tool_index: ToolIndex
    ) -> NextAction | None:
        air_quality = state.air_quality

        if not air_quality.climate_checked and tool_index.has("get_climate_settings"):
            return NextAction.tool_call(
                "get_climate_settings",
                reason="air_quality_climate_check",
            )

        fan_status = (
            "currently off"
            if air_quality.fan_speed == 0
            else (
                f"currently at level {air_quality.fan_speed}"
                if air_quality.fan_speed is not None
                else "not currently known"
            )
        )
        return NextAction.respond(
            "The fan is "
            f"{fan_status}. I can turn on the fan, change air circulation, or turn on the air conditioning. Which would you like?",
            reason="air_quality_user_disambiguation",
        )

    def _next_fan_speed_action(
        self, state: ControllerState, tool_index: ToolIndex
    ) -> NextAction | None:
        fan = state.fan_speed

        if fan.query_current:
            if not fan.current_checked:
                if not tool_index.has("get_climate_settings"):
                    state.fan_speed = FanSpeedFlow()
                    return NextAction.respond(
                        "I can't look up the current fan speed because the climate settings check is unavailable right now.",
                        reason="fan_speed_missing_climate_tool",
                    )
                return NextAction.tool_call(
                    "get_climate_settings",
                    reason="fan_speed_current_check",
                )
            level = fan.current_level
            state.fan_speed = FanSpeedFlow()
            if level is None:
                return NextAction.respond(
                    "I can't determine the current fan speed from the available climate settings.",
                    reason="fan_speed_current_unknown",
                )
            return NextAction.respond(
                f"The current fan speed is level {level}.",
                reason="fan_speed_current",
            )

        if fan.completed:
            level = fan.level if fan.level is not None else "the requested level"
            state.fan_speed = FanSpeedFlow()
            return NextAction.respond(
                f"Done, the fan speed is set to level {level}.",
                reason="fan_speed_done",
            )

        if not tool_index.has("set_fan_speed"):
            return NextAction.respond(
                "I can't adjust the fan speed because that control is unavailable right now.",
                reason="fan_speed_missing_tool",
            )

        if not _tool_argument_available(tool_index, "set_fan_speed", "level"):
            return NextAction.respond(
                "I can't adjust the fan speed because the required level control is unavailable right now.",
                reason="fan_speed_missing_level_parameter",
            )

        if fan.level is None:
            if not fan.preferences_checked and tool_index.has("get_user_preferences"):
                return NextAction.tool_call(
                    "get_user_preferences",
                    {
                        "preference_categories": {
                            "vehicle_settings": {"climate_control": True}
                        }
                    },
                    reason="fan_speed_internal_disambiguation_preferences",
                )
            return NextAction.respond(
                "What fan speed level should I set?",
                reason="fan_speed_user_disambiguation",
            )

        return NextAction.tool_call(
            "set_fan_speed",
            {"level": fan.level},
            reason="fan_speed_set",
        )

    def _next_steering_wheel_heating_action(
        self, state: ControllerState, tool_index: ToolIndex
    ) -> NextAction | None:
        heating = state.steering_wheel_heating

        if heating.completed:
            level = (
                heating.level if heating.level is not None else "the requested level"
            )
            state.steering_wheel_heating = SteeringWheelHeatingFlow()
            return NextAction.respond(
                f"Done, the steering wheel heating is set to level {level}.",
                reason="steering_wheel_heating_done",
            )

        if not tool_index.has("set_steering_wheel_heating"):
            return NextAction.respond(
                "I can't adjust the steering wheel heating because that control is unavailable right now.",
                reason="steering_wheel_heating_missing_tool",
            )

        if not _tool_argument_available(
            tool_index, "set_steering_wheel_heating", "level"
        ):
            return NextAction.respond(
                "I can't adjust the steering wheel heating because the required level control is unavailable right now.",
                reason="steering_wheel_heating_missing_level_parameter",
            )

        if heating.level is None:
            if not heating.preferences_checked and tool_index.has(
                "get_user_preferences"
            ):
                return NextAction.tool_call(
                    "get_user_preferences",
                    {
                        "preference_categories": {
                            "vehicle_settings": {
                                "climate_control": True,
                                "vehicle_settings": True,
                            }
                        }
                    },
                    reason="steering_wheel_heating_internal_disambiguation_preferences",
                )
            return NextAction.respond(
                "What steering wheel heating level should I set?",
                reason="steering_wheel_heating_user_disambiguation",
            )

        return NextAction.tool_call(
            "set_steering_wheel_heating",
            {"level": heating.level},
            reason="steering_wheel_heating_set",
        )

    def _next_cabin_temperature_action(
        self, state: ControllerState, tool_index: ToolIndex
    ) -> NextAction | None:
        temperature = state.cabin_temperature

        if temperature.completed:
            state.cabin_temperature = CabinTemperatureFlow()
            return NextAction.respond(
                "Done, I set the cabin temperature to "
                f"{_format_temperature_target(temperature.target_temperature)}.",
                reason="cabin_temperature_done",
            )

        if temperature.target_temperature is None:
            return NextAction.respond(
                "What cabin temperature should I set?",
                reason="cabin_temperature_disambiguation",
            )

        if not tool_index.has("set_climate_temperature"):
            temperature.completed = True
            return NextAction.respond(
                "I can't set the cabin temperature because that control is unavailable right now.",
                reason="cabin_temperature_missing_tool",
            )

        temperature_args = tool_index.arg_names("set_climate_temperature")
        if "seat_zone" not in temperature_args or "temperature" not in temperature_args:
            temperature.completed = True
            return NextAction.respond(
                "I can't set the cabin temperature because the required temperature controls are unavailable right now.",
                reason="cabin_temperature_missing_parameter",
            )

        return NextAction.tool_call(
            "set_climate_temperature",
            {
                "seat_zone": temperature.seat_zone,
                "temperature": temperature.target_temperature,
            },
            reason="cabin_temperature_set",
        )

    def _next_climate_energy_inspection_action(
        self, state: ControllerState, tool_index: ToolIndex
    ) -> NextAction | None:
        inspection = state.climate_energy_inspection

        if inspection.completed:
            state.climate_energy_inspection = ClimateEnergyInspectionFlow()
            return None

        if not inspection.climate_checked:
            if not tool_index.has("get_climate_settings"):
                inspection.completed = True
                return NextAction.respond(
                    "I can't check the current climate settings right now.",
                    reason="climate_energy_inspection_missing_climate_tool",
                )
            return NextAction.tool_call(
                "get_climate_settings",
                reason="climate_energy_inspection_climate_check",
            )

        if not inspection.seats_checked:
            if not tool_index.has("get_seats_occupancy"):
                inspection.completed = True
                return NextAction.respond(
                    "I can't check which seats are occupied right now.",
                    reason="climate_energy_inspection_missing_occupancy_tool",
                )
            return NextAction.tool_call(
                "get_seats_occupancy",
                reason="climate_energy_inspection_occupancy_check",
            )

        inspection.completed = True
        return NextAction.respond(
            _format_climate_energy_inspection_summary(inspection),
            reason="climate_energy_inspection_done",
        )

    def _next_climate_inspection_action(
        self, state: ControllerState, tool_index: ToolIndex
    ) -> NextAction | None:
        climate = state.climate_inspection

        if climate.needs_temperature and not climate.temperature_checked:
            if not tool_index.has("get_temperature_inside_car"):
                climate.completed = True
                return NextAction.respond(
                    "I can't check the current cabin temperature because that information is unavailable right now.",
                    reason="climate_inspection_missing_temperature_tool",
                )
            return NextAction.tool_call(
                "get_temperature_inside_car",
                reason="climate_inspection_temperature_check",
            )

        if climate.needs_seat_heating and not climate.seat_heating_checked:
            if not tool_index.has("get_seat_heating_level"):
                climate.completed = True
                return NextAction.respond(
                    "I can't check the current seat heating levels because that information is unavailable right now.",
                    reason="climate_inspection_missing_seat_heating_tool",
                )
            return NextAction.tool_call(
                "get_seat_heating_level",
                reason="climate_inspection_seat_heating_check",
            )

        if climate.needs_temperature and (
            climate.driver_temperature is None
            or climate.passenger_temperature is None
        ):
            climate.completed = True
            return NextAction.respond(
                "I can't determine the current cabin temperatures from the available tool result.",
                reason="climate_inspection_missing_temperature_result",
            )

        if climate.needs_seat_heating and (
            climate.seat_heating_driver is None
            or climate.seat_heating_passenger is None
        ):
            climate.completed = True
            return NextAction.respond(
                "I can't determine the current seat heating levels from the available tool result.",
                reason="climate_inspection_missing_seat_heating_result",
            )

        climate.completed = True
        return NextAction.respond(
            _format_climate_inspection_summary(climate),
            reason="climate_inspection_done",
        )

    def _next_occupancy_comfort_action(
        self, state: ControllerState, tool_index: ToolIndex
    ) -> NextAction | None:
        comfort = state.occupancy_comfort

        if comfort.completed:
            state.occupancy_comfort = OccupancyComfortFlow()
            return None

        if comfort.heating_delta is not None and not comfort.seat_heating_checked:
            if not tool_index.has("get_seat_heating_level"):
                comfort.completed = True
                return NextAction.respond(
                    "I can't check the current seat heating levels, so I can't adjust seat heating by occupancy reliably.",
                    reason="occupancy_comfort_missing_seat_heating_check",
                )
            return NextAction.tool_call(
                "get_seat_heating_level",
                reason="occupancy_comfort_seat_heating_check",
            )

        if not comfort.seats_checked:
            if not tool_index.has("get_seats_occupancy"):
                comfort.completed = True
                return NextAction.respond(
                    "I can't tell which seats are occupied, so I can't optimize comfort by occupied seats right now.",
                    reason="occupancy_comfort_missing_occupancy_tool",
                )
            return NextAction.tool_call(
                "get_seats_occupancy",
                reason="occupancy_comfort_occupancy_check",
            )

        if not comfort.seats_occupied:
            comfort.completed = True
            return NextAction.respond(
                "I can't tell which seats are occupied, so I can't optimize seat heating by occupancy.",
                reason="occupancy_comfort_missing_occupancy_result",
            )

        if (
            comfort.match_driver_to_passenger_temperature
            and not comfort.temperature_checked
        ):
            if not tool_index.has("get_temperature_inside_car"):
                comfort.completed = True
                return NextAction.respond(
                    "I can't check the current passenger temperature, so I can't match the driver temperature to it.",
                    reason="occupancy_comfort_missing_temperature_check",
                )
            return NextAction.tool_call(
                "get_temperature_inside_car",
                reason="occupancy_comfort_temperature_check",
            )

        if (
            comfort.match_driver_to_passenger_temperature
            and comfort.target_temperature is None
        ):
            if comfort.passenger_temperature is None:
                comfort.completed = True
                return NextAction.respond(
                    "I can't determine the current passenger temperature from the available tool result.",
                    reason="occupancy_comfort_missing_passenger_temperature_result",
                )
            comfort.needs_temperature = True
            comfort.target_temperature_zone = "DRIVER"
            comfort.target_temperature = comfort.passenger_temperature

        if (
            comfort.query_unoccupied_heating_status
            and not comfort.seat_heating_checked
        ):
            if not tool_index.has("get_seat_heating_level"):
                comfort.completed = True
                return NextAction.respond(
                    "I can't check the current seat heating levels right now.",
                    reason="occupancy_comfort_missing_seat_heating_check",
                )
            return NextAction.tool_call(
                "get_seat_heating_level",
                reason="occupancy_comfort_unoccupied_heating_status_check",
            )

        if comfort.query_unoccupied_heating_status:
            comfort.completed = True
            return NextAction.respond(
                _format_unoccupied_seat_heating_status(comfort),
                reason="occupancy_comfort_unoccupied_heating_status",
            )

        if comfort.turn_off_unoccupied and not comfort.seat_heating_checked:
            if not tool_index.has("get_seat_heating_level"):
                comfort.completed = True
                return NextAction.respond(
                    "I can't check the current seat heating levels, so I can't adjust seat heating by occupancy reliably.",
                    reason="occupancy_comfort_missing_seat_heating_check",
                )
            return NextAction.tool_call(
                "get_seat_heating_level",
                reason="occupancy_comfort_seat_heating_check",
            )

        if comfort.needs_temperature and comfort.target_temperature is None:
            if not comfort.preferences_checked and tool_index.has(
                "get_user_preferences"
            ):
                return NextAction.tool_call(
                    "get_user_preferences",
                    {
                        "preference_categories": {
                            "vehicle_settings": {"climate_control": True}
                        }
                    },
                    reason="occupancy_comfort_temperature_preferences",
                )
            return NextAction.respond(
                "What cabin temperature should I set?",
                reason="occupancy_comfort_temperature_disambiguation",
            )

        if comfort.target_heating_level is None and comfort.heating_delta is not None:
            target_level = _occupied_front_seat_heating_target(comfort)
            if target_level is None:
                comfort.completed = True
                return NextAction.respond(
                    "I can't determine the current occupied-seat heating levels from the available tool result.",
                    reason="occupancy_comfort_missing_current_heating_result",
                )
            comfort.target_heating_level = target_level

        if comfort.target_heating_level is None:
            if comfort.turn_off_unoccupied:
                comfort.seat_heating_set = True
            else:
                return NextAction.respond(
                    "What seat heating level should I use for the occupied seats?",
                    reason="occupancy_comfort_heating_level_disambiguation",
                )

        if (
            comfort.needs_temperature
            and not comfort.turn_off_unoccupied
            and not comfort.temperature_set
        ):
            if not tool_index.has("set_climate_temperature"):
                comfort.completed = True
                return NextAction.respond(
                    "I can't set the cabin temperature because that control is unavailable right now.",
                    reason="occupancy_comfort_missing_temperature_tool",
                )
            temperature_args = tool_index.arg_names("set_climate_temperature")
            if "seat_zone" not in temperature_args or "temperature" not in temperature_args:
                comfort.completed = True
                return NextAction.respond(
                    "I can't set the cabin temperature because the required temperature controls are unavailable right now.",
                    reason="occupancy_comfort_missing_temperature_parameter",
                )
            return NextAction.tool_call(
                "set_climate_temperature",
                {
                    "seat_zone": comfort.target_temperature_zone,
                    "temperature": comfort.target_temperature,
                },
                reason="occupancy_comfort_set_temperature",
            )

        if comfort.turn_off_unoccupied and not comfort.unoccupied_heating_set:
            if not tool_index.has("set_seat_heating"):
                comfort.completed = True
                return NextAction.respond(
                    "I can't turn off seat heating for unoccupied seats because that control is unavailable right now.",
                    reason="occupancy_comfort_missing_seat_heating_tool",
                )
            seat_args = tool_index.arg_names("set_seat_heating")
            if "seat_zone" not in seat_args or "level" not in seat_args:
                comfort.completed = True
                return NextAction.respond(
                    "I can't adjust seat heating by occupied seat because the required seat heating controls are unavailable right now.",
                    reason="occupancy_comfort_missing_seat_heating_parameter",
                )
            unoccupied_zone = _unoccupied_front_seat_heating_zone(comfort)
            if unoccupied_zone is None:
                comfort.unoccupied_heating_set = True
            else:
                return NextAction.tool_call(
                    "set_seat_heating",
                    {"seat_zone": unoccupied_zone, "level": 0},
                    reason="occupancy_comfort_turn_off_unoccupied_heating",
                )

        if not comfort.seat_heating_set:
            if not tool_index.has("set_seat_heating"):
                comfort.completed = True
                return NextAction.respond(
                    "I can't set seat heating because that control is unavailable right now.",
                    reason="occupancy_comfort_missing_seat_heating_tool",
                )
            seat_args = tool_index.arg_names("set_seat_heating")
            if "seat_zone" not in seat_args or "level" not in seat_args:
                comfort.completed = True
                return NextAction.respond(
                    "I can't set seat heating by occupied seat because the required seat heating controls are unavailable right now.",
                    reason="occupancy_comfort_missing_seat_heating_parameter",
                )
            seat_zone = _seat_heating_zone_for_occupied_front_seats(
                comfort.seats_occupied
            )
            if seat_zone is None:
                comfort.seat_heating_set = True
            else:
                return NextAction.tool_call(
                    "set_seat_heating",
                    {
                        "seat_zone": seat_zone,
                        "level": comfort.target_heating_level,
                    },
                    reason="occupancy_comfort_set_seat_heating",
                )

        if comfort.needs_temperature and not comfort.temperature_set:
            if not tool_index.has("set_climate_temperature"):
                comfort.completed = True
                return NextAction.respond(
                    "I can't set the cabin temperature because that control is unavailable right now.",
                    reason="occupancy_comfort_missing_temperature_tool",
                )
            temperature_args = tool_index.arg_names("set_climate_temperature")
            if "seat_zone" not in temperature_args or "temperature" not in temperature_args:
                comfort.completed = True
                return NextAction.respond(
                    "I can't set the cabin temperature because the required temperature controls are unavailable right now.",
                    reason="occupancy_comfort_missing_temperature_parameter",
                )
            return NextAction.tool_call(
                "set_climate_temperature",
                {
                    "seat_zone": comfort.target_temperature_zone,
                    "temperature": comfort.target_temperature,
                },
                reason="occupancy_comfort_set_temperature",
            )

        if comfort.steering_wheel_requested and not comfort.steering_wheel_set:
            if comfort.steering_wheel_level is None:
                comfort.steering_wheel_level = comfort.target_heating_level
            if not tool_index.has("set_steering_wheel_heating"):
                comfort.completed = True
                return NextAction.respond(
                    "I can't adjust the steering wheel heating because that control is unavailable right now.",
                    reason="occupancy_comfort_missing_steering_heating_tool",
                )
            if not _tool_argument_available(
                tool_index, "set_steering_wheel_heating", "level"
            ):
                comfort.completed = True
                return NextAction.respond(
                    "I can't adjust the steering wheel heating because the required level control is unavailable right now.",
                    reason="occupancy_comfort_missing_steering_heating_parameter",
                )
            return NextAction.tool_call(
                "set_steering_wheel_heating",
                {"level": comfort.steering_wheel_level},
                reason="occupancy_comfort_set_steering_wheel_heating",
            )

        comfort.completed = True
        completed = ["adjusted seat heating for the occupied seats"]
        if comfort.turn_off_unoccupied:
            completed.insert(0, "turned off seat heating for unoccupied front seats")
        if comfort.needs_temperature:
            target = (
                "set the driver temperature to match the passenger temperature"
                if comfort.match_driver_to_passenger_temperature
                else "set the cabin temperature to "
                f"{_format_temperature_target(comfort.target_temperature)}"
            )
            completed.insert(
                0,
                target,
            )
        if comfort.steering_wheel_requested:
            level = (
                comfort.steering_wheel_level
                if comfort.steering_wheel_level is not None
                else comfort.target_heating_level
            )
            completed.append(f"set steering wheel heating to level {level}")
        return NextAction.respond(
            f"Done, I {_join_completed_actions(completed)}.",
            reason="occupancy_comfort_done",
        )

    def _next_passenger_comfort_match_action(
        self, state: ControllerState, tool_index: ToolIndex
    ) -> NextAction | None:
        comfort = state.passenger_comfort_match

        if comfort.completed:
            state.passenger_comfort_match = PassengerComfortMatchFlow()
            return NextAction.respond(
                "Done, I matched the passenger comfort settings to the driver side.",
                reason="passenger_comfort_match_done",
            )

        if comfort.target_temperature is None:
            return NextAction.respond(
                "What passenger temperature should I set?",
                reason="passenger_comfort_match_temperature_disambiguation",
            )

        if not comfort.temperature_set:
            if not tool_index.has("set_climate_temperature"):
                comfort.completed = True
                return NextAction.respond(
                    "I can't set the passenger temperature because that control is unavailable right now.",
                    reason="passenger_comfort_match_missing_temperature_tool",
                )
            temperature_args = tool_index.arg_names("set_climate_temperature")
            if "seat_zone" not in temperature_args or "temperature" not in temperature_args:
                comfort.completed = True
                return NextAction.respond(
                    "I can't set the passenger temperature because the required temperature controls are unavailable right now.",
                    reason="passenger_comfort_match_missing_temperature_parameter",
                )
            return NextAction.tool_call(
                "set_climate_temperature",
                {
                    "seat_zone": "PASSENGER",
                    "temperature": comfort.target_temperature,
                },
                reason="passenger_comfort_match_set_temperature",
            )

        if not comfort.seat_heating_checked:
            if not tool_index.has("get_seat_heating_level"):
                comfort.completed = True
                return NextAction.respond(
                    "I can't check the driver's current seat heating level, so I can't match the passenger seat heating to it.",
                    reason="passenger_comfort_match_missing_seat_heating_check",
                )
            return NextAction.tool_call(
                "get_seat_heating_level",
                reason="passenger_comfort_match_seat_heating_check",
            )

        if comfort.driver_heating_level is None:
            comfort.completed = True
            return NextAction.respond(
                "I can't determine the driver's current seat heating level from the available tool result.",
                reason="passenger_comfort_match_missing_driver_heating_result",
            )

        if not comfort.passenger_heating_set:
            if not tool_index.has("set_seat_heating"):
                comfort.completed = True
                return NextAction.respond(
                    "I can't set the passenger seat heating because that control is unavailable right now.",
                    reason="passenger_comfort_match_missing_seat_heating_tool",
                )
            seat_args = tool_index.arg_names("set_seat_heating")
            if "seat_zone" not in seat_args:
                comfort.completed = True
                return NextAction.respond(
                    "I can't set the passenger seat heating because the seat selector is unavailable right now, so I can't safely complete matching the steering wheel heating to that applied level.",
                    reason="passenger_comfort_match_missing_seat_zone_parameter",
                )
            if "level" not in seat_args:
                comfort.completed = True
                return NextAction.respond(
                    "I can't set the passenger seat heating because the heating level control is unavailable right now.",
                    reason="passenger_comfort_match_missing_level_parameter",
                )
            return NextAction.tool_call(
                "set_seat_heating",
                {"seat_zone": "PASSENGER", "level": comfort.driver_heating_level},
                reason="passenger_comfort_match_set_passenger_heating",
            )

        if not comfort.steering_wheel_set:
            if not tool_index.has("set_steering_wheel_heating"):
                comfort.completed = True
                return NextAction.respond(
                    "I can't adjust the steering wheel heating because that control is unavailable right now.",
                    reason="passenger_comfort_match_missing_steering_tool",
                )
            if not _tool_argument_available(
                tool_index, "set_steering_wheel_heating", "level"
            ):
                comfort.completed = True
                return NextAction.respond(
                    "I can't adjust the steering wheel heating because the required level control is unavailable right now.",
                    reason="passenger_comfort_match_missing_steering_level_parameter",
                )
            return NextAction.tool_call(
                "set_steering_wheel_heating",
                {"level": comfort.driver_heating_level},
                reason="passenger_comfort_match_set_steering_wheel_heating",
            )

        comfort.completed = True
        return NextAction.respond(
            "Done, I set the passenger temperature, matched passenger seat heating to the driver side, and matched the steering wheel heating to that level.",
            reason="passenger_comfort_match_done",
        )

    def _next_driver_comfort_temperature_action(
        self, state: ControllerState, tool_index: ToolIndex
    ) -> NextAction | None:
        comfort = state.driver_comfort_temperature

        if comfort.passenger_heating_required and not comfort.passenger_heating_off:
            if not tool_index.has("set_seat_heating"):
                comfort.completed = True
                return NextAction.respond(
                    "I can't turn off the passenger seat heating because that control is unavailable right now.",
                    reason="driver_comfort_missing_seat_heating_tool",
                )
            seat_args = tool_index.arg_names("set_seat_heating")
            if "seat_zone" not in seat_args or "level" not in seat_args:
                comfort.completed = True
                return NextAction.respond(
                    "I can't turn off the passenger seat heating because the required seat heating controls are unavailable right now.",
                    reason="driver_comfort_missing_seat_heating_parameter",
                )
            return NextAction.tool_call(
                "set_seat_heating",
                {"seat_zone": "PASSENGER", "level": 0},
                reason="driver_comfort_passenger_heating_off",
            )

        if comfort.target_temperature is None:
            if not comfort.preferences_checked and tool_index.has(
                "get_user_preferences"
            ):
                return NextAction.tool_call(
                    "get_user_preferences",
                    {
                        "preference_categories": {
                            "vehicle_settings": {"climate_control": True}
                        }
                    },
                    reason="driver_comfort_temperature_preferences",
                )
            return NextAction.respond(
                "What driver temperature should I set?",
                reason="driver_comfort_temperature_disambiguation",
            )

        if not comfort.temperature_set:
            if not tool_index.has("set_climate_temperature"):
                comfort.completed = True
                return NextAction.respond(
                    "I can't set the driver temperature because that control is unavailable right now.",
                    reason="driver_comfort_missing_temperature_tool",
                )
            temperature_args = tool_index.arg_names("set_climate_temperature")
            if "seat_zone" not in temperature_args or "temperature" not in temperature_args:
                comfort.completed = True
                return NextAction.respond(
                    "I can't set the driver temperature because the required temperature controls are unavailable right now.",
                    reason="driver_comfort_missing_temperature_parameter",
                )
            return NextAction.tool_call(
                "set_climate_temperature",
                {"seat_zone": "DRIVER", "temperature": comfort.target_temperature},
                reason="driver_comfort_set_temperature",
            )

        comfort.completed = True
        if comfort.passenger_heating_required:
            return NextAction.respond(
                "Done, I turned off passenger seat heating and set the driver temperature to "
                f"{_format_temperature_target(comfort.target_temperature)}.",
                reason="driver_comfort_done",
            )
        return NextAction.respond(
            "Done, I set the driver temperature to "
            f"{_format_temperature_target(comfort.target_temperature)}.",
            reason="driver_comfort_done",
        )

    def _next_reading_light_occupancy_action(
        self, state: ControllerState, tool_index: ToolIndex
    ) -> NextAction | None:
        occupancy = state.reading_light_occupancy

        if occupancy.completed:
            state.reading_light_occupancy = ReadingLightOccupancyFlow()
            return NextAction.respond(
                "Done, the reading lights now match occupied seats.",
                reason="reading_light_occupancy_done",
            )

        if not tool_index.has("set_reading_light"):
            return NextAction.respond(
                "I can't change the reading lights because that control is unavailable right now.",
                reason="reading_light_occupancy_missing_tool",
            )

        if not _tool_argument_available(tool_index, "set_reading_light", "position"):
            return NextAction.respond(
                "I can't change the reading lights because the required position control is unavailable right now.",
                reason="reading_light_occupancy_missing_position_parameter",
            )

        if not _tool_argument_available(tool_index, "set_reading_light", "on"):
            return NextAction.respond(
                "I can't change the reading lights because the required on/off control is unavailable right now.",
                reason="reading_light_occupancy_missing_on_parameter",
            )

        if not occupancy.seats_checked:
            if not tool_index.has("get_seats_occupancy"):
                return NextAction.respond(
                    "I can't optimize the reading lights by occupied seats because seat occupancy is unavailable right now.",
                    reason="reading_light_occupancy_missing_seats_tool",
                )
            return NextAction.tool_call(
                "get_seats_occupancy",
                reason="reading_light_occupancy_check",
            )

        if not occupancy.pending_actions:
            occupancy.pending_actions = _reading_light_actions_for_occupancy(
                occupancy.seats_occupied
            )

        if not occupancy.pending_actions:
            occupancy.completed = True
            return self._next_reading_light_occupancy_action(state, tool_index)

        position, on = occupancy.pending_actions[0]
        return NextAction.tool_call(
            "set_reading_light",
            {"position": position, "on": on},
            reason="reading_light_occupancy_set",
        )

    def _next_reading_light_action(
        self, state: ControllerState, tool_index: ToolIndex
    ) -> NextAction | None:
        reading_light = state.reading_light

        if reading_light.completed:
            position = _friendly_reading_light_position(reading_light.position)
            state.reading_light = ReadingLightFlow()
            return NextAction.respond(
                f"Done, {position} reading light is on."
                if reading_light.on
                else f"Done, {position} reading light is off.",
                reason="reading_light_done",
            )

        if not tool_index.has("set_reading_light"):
            return NextAction.respond(
                "I can't change the reading lights because that control is unavailable right now.",
                reason="reading_light_missing_tool",
            )

        if not _tool_argument_available(tool_index, "set_reading_light", "position"):
            return NextAction.respond(
                "I can't change the reading lights because the required position control is unavailable right now.",
                reason="reading_light_missing_position_parameter",
            )

        if not _tool_argument_available(tool_index, "set_reading_light", "on"):
            return NextAction.respond(
                "I can't change the reading lights because the required on/off control is unavailable right now.",
                reason="reading_light_missing_on_parameter",
            )

        if reading_light.position is None:
            return NextAction.respond(
                "Which reading light should I turn on: driver, passenger, driver rear, or passenger rear?",
                reason="reading_light_user_disambiguation",
            )

        return NextAction.tool_call(
            "set_reading_light",
            {"position": reading_light.position, "on": reading_light.on},
            reason="reading_light_set",
        )

    def _next_email_action(
        self, state: ControllerState, tool_index: ToolIndex
    ) -> NextAction | None:
        email = state.email

        if email.completed:
            state.email = EmailFlow()
            return NextAction.respond(
                "Done, the email has been sent.",
                reason="email_done",
            )

        if email.failure_message:
            state.email = EmailFlow()
            return NextAction.respond(email.failure_message, reason="email_failed")

        if not tool_index.has("send_email"):
            return NextAction.respond(
                "I can't send email because the email tool is unavailable right now.",
                reason="email_missing_send_tool",
            )

        if email.pending_contact_matches:
            return NextAction.respond(
                _format_contact_choice_prompt(email.pending_contact_matches),
                reason="email_contact_disambiguation",
            )

        if email.mode == "meeting_attendees":
            if not email.calendar_checked:
                if state.recent_calendar_meetings:
                    email.calendar_checked = True
                elif not tool_index.has("get_entries_from_calendar"):
                    return NextAction.respond(
                        "I can't look up the meeting attendees because calendar lookup is unavailable right now.",
                        reason="email_missing_calendar_tool",
                    )
                else:
                    calendar_args = _calendar_arguments(state.runtime)
                    if calendar_args is None:
                        return NextAction.respond(
                            "I need today's date before I can look up the meeting attendees.",
                            reason="email_missing_calendar_context",
                        )
                    return NextAction.tool_call(
                        "get_entries_from_calendar",
                        calendar_args,
                        reason="email_calendar_lookup",
                    )

            if not email.attendee_contact_ids:
                selected = _select_meeting(
                    state.recent_calendar_meetings, email.meeting_topic
                )
                if selected is None:
                    return NextAction.respond(
                        _format_meeting_choice_prompt(state.recent_calendar_meetings),
                        reason="email_meeting_disambiguation",
                    )
                _record_email_attendee_meeting(state, selected)

            if not email.attendee_contact_ids:
                return NextAction.respond(
                    "I can't send the email because I couldn't find attendees for that meeting.",
                    reason="email_missing_attendees",
                )

        if email.mode != "meeting_attendees" and email.recipient_contact_id is None:
            return self._next_email_contact_lookup(
                email,
                tool_index,
                role="recipient",
            )

        if email.mode == "share_contact" and email.subject_contact_id is None:
            if email.subject_name is None:
                return NextAction.respond(
                    "Whose contact details should I include in the email?",
                    reason="email_missing_contact_to_share",
                )
            return self._next_email_contact_lookup(
                email,
                tool_index,
                role="subject",
            )

        if email.mode == "meeting_delay":
            if email.calendar_checked and email.meeting_started is None:
                _record_email_meeting_status(state)

            if not email.calendar_checked:
                if state.recent_calendar_meetings:
                    email.calendar_checked = True
                    _record_email_meeting_status(state)
                elif not tool_index.has("get_entries_from_calendar"):
                    return NextAction.respond(
                        "I can't verify the meeting status because calendar lookup is unavailable right now.",
                        reason="email_missing_calendar_tool",
                    )
                else:
                    calendar_args = _calendar_arguments(state.runtime)
                    if calendar_args is None:
                        return NextAction.respond(
                            "I need today's date before I can verify the meeting status.",
                            reason="email_missing_calendar_context",
                        )
                    return NextAction.tool_call(
                        "get_entries_from_calendar",
                        calendar_args,
                        reason="email_calendar_lookup",
                    )

            if email.meeting_started is False and not email.user_claimed_late:
                return NextAction.respond(
                    "That meeting has not started yet, so I will not send a late-arrival email.",
                    reason="email_meeting_not_started",
                )

        missing_contact_ids = _email_missing_contact_info_ids(email)
        if missing_contact_ids:
            if not tool_index.has("get_contact_information"):
                return NextAction.respond(
                    "I can't send the email because contact details lookup is unavailable right now.",
                    reason="email_missing_contact_info_tool",
                )
            return NextAction.tool_call(
                "get_contact_information",
                {"contact_ids": missing_contact_ids},
                reason="email_contact_information_lookup",
            )

        if email.mode == "meeting_attendees":
            if not _email_attendee_addresses(email):
                return NextAction.respond(
                    "I can't send the email because I couldn't find email addresses for the meeting attendees.",
                    reason="email_missing_attendee_addresses",
                )
        elif email.recipient_email is None:
            return NextAction.respond(
                "I can't send the email because I couldn't find the recipient's email address.",
                reason="email_missing_recipient_address",
            )

        if email.weather_requested and not email.weather_checked:
            if not tool_index.has("get_weather"):
                return NextAction.respond(
                    "I can't check the weather for the meeting because weather lookup is unavailable right now.",
                    reason="email_missing_weather_tool",
                )
            if _email_weather_needs_location_lookup(state, email, tool_index):
                state.pending_location_lookup_name = email.weather_location_name
                email.weather_location_lookup_attempted = True
                return NextAction.tool_call(
                    "get_location_id_by_location_name",
                    {"location": email.weather_location_name},
                    reason="email_weather_location_lookup",
                )
            weather_args = _email_weather_arguments(state, email, tool_index)
            if weather_args is None:
                return NextAction.respond(
                    "I need the meeting location and time before I can check the weather.",
                    reason="email_missing_weather_context",
                )
            return NextAction.tool_call(
                "get_weather",
                weather_args,
                reason="email_weather_lookup",
            )

        if (
            email.mode == "meeting_delay"
            and not email.preferences_checked
            and not state.business_email_extra_recipients
            and tool_index.has("get_user_preferences")
        ):
            return NextAction.tool_call(
                "get_user_preferences",
                {
                    "preference_categories": {
                        "productivity_and_communication": {"email": True}
                    }
                },
                reason="email_preferences_lookup",
            )

        if email.content_message is None:
            email.content_message = _build_email_content(state)
            if email.content_message is None:
                return NextAction.respond(
                    "What should the email say?",
                    reason="email_missing_content",
                )

        if not email.confirmed:
            email.confirmation_requested = True
            return NextAction.respond(
                _format_email_confirmation(
                    email, state.business_email_extra_recipients
                ),
                reason="email_confirmation",
            )

        return NextAction.tool_call(
            "send_email",
            {
                "email_addresses": _email_recipient_addresses(
                    email, state.business_email_extra_recipients
                ),
                "content_message": email.content_message,
            },
            reason="email_send",
        )

    def _next_email_contact_lookup(
        self,
        email: EmailFlow,
        tool_index: ToolIndex,
        *,
        role: Literal["recipient", "subject"],
    ) -> NextAction | None:
        if not tool_index.has("get_contact_id_by_contact_name"):
            return NextAction.respond(
                "I can't look up contacts because contact search is unavailable right now.",
                reason="email_missing_contact_lookup_tool",
            )

        first_name = email.recipient_first_name
        last_name = email.recipient_last_name
        if role == "subject":
            first_name = email.subject_first_name
            last_name = email.subject_last_name

        if not first_name:
            return NextAction.respond(
                "Which contact should I look up?",
                reason="email_missing_contact_name",
            )

        arguments = {"contact_first_name": first_name}
        if last_name:
            arguments["contact_last_name"] = last_name

        email.pending_lookup_role = role
        return NextAction.tool_call(
            "get_contact_id_by_contact_name",
            arguments,
            reason=f"email_{role}_contact_lookup",
        )

    def _next_high_beam_action(
        self, state: ControllerState, tool_index: ToolIndex
    ) -> NextAction | None:
        high_beam = state.high_beam

        if high_beam.declined:
            target = "on" if high_beam.on else "off"
            state.high_beam = HighBeamFlow()
            return NextAction.respond(
                f"Okay, I won't set the high beam headlights to {target}.",
                reason="high_beam_declined",
            )

        if high_beam.completed:
            on = high_beam.on
            state.high_beam = HighBeamFlow()
            return NextAction.respond(
                "Done, the high beam headlights are on."
                if on
                else "Done, the high beam headlights are off.",
                reason="high_beam_done",
            )

        if not tool_index.has("set_head_lights_high_beams"):
            return NextAction.respond(
                "I can't change the high beam headlights because that control is unavailable right now.",
                reason="high_beam_missing_tool",
            )

        if high_beam.on and not high_beam.exterior_lights_checked:
            if not tool_index.has("get_exterior_lights_status"):
                return NextAction.respond(
                    "I can't safely turn on the high beams because the exterior light status check is unavailable right now.",
                    reason="high_beam_missing_status_tool",
                )
            return NextAction.tool_call(
                "get_exterior_lights_status",
                reason="high_beam_status_check",
            )

        if high_beam.on and high_beam.fog_lights_on is True:
            return NextAction.respond(
                "I can't turn on the high beams while the fog lights are on, because that combination reduces visibility.",
                reason="high_beam_fog_light_conflict",
            )

        if high_beam.on and high_beam.fog_lights_on is None:
            return NextAction.respond(
                "I can't safely turn on the high beams because I couldn't determine the current fog light status.",
                reason="high_beam_unknown_fog_light_status",
            )

        if not high_beam.confirmed:
            high_beam.confirmation_requested = True
            target = "on" if high_beam.on else "off"
            return NextAction.respond(
                f"Please confirm: I will set the high beam headlights to {target}.",
                reason="high_beam_confirmation",
            )

        return NextAction.tool_call(
            "set_head_lights_high_beams",
            {"on": high_beam.on},
            reason="high_beam_set",
        )

    def _next_fog_lights_action(
        self, state: ControllerState, tool_index: ToolIndex
    ) -> NextAction | None:
        fog_lights = state.fog_lights

        if fog_lights.declined:
            state.fog_lights = FogLightFlow()
            return NextAction.respond(
                "Okay, I won't change the fog lights.",
                reason="fog_lights_declined",
            )

        if fog_lights.completed:
            on = fog_lights.on
            state.fog_lights = FogLightFlow()
            return NextAction.respond(
                "Done, the fog lights are on."
                if on
                else "Done, the fog lights are off.",
                reason="fog_lights_done",
            )

        if not tool_index.has("set_fog_lights"):
            return NextAction.respond(
                "I can't change the fog lights because that control is unavailable right now.",
                reason="fog_lights_missing_tool",
            )

        if not fog_lights.on:
            return NextAction.tool_call(
                "set_fog_lights",
                {"on": False},
                reason="fog_lights_off",
            )

        if not fog_lights.weather_checked:
            if not tool_index.has("get_weather"):
                return NextAction.respond(
                    "I can't safely turn on the fog lights because the weather check is unavailable right now.",
                    reason="fog_lights_missing_weather_tool",
                )
            weather_args = _weather_arguments(state.runtime)
            if weather_args is None:
                return NextAction.respond(
                    "I need the current location and time before I can safely turn on the fog lights.",
                    reason="fog_lights_missing_weather_context",
                )
            return NextAction.tool_call(
                "get_weather",
                weather_args,
                reason="fog_lights_weather_check",
            )

        if (
            _requires_fog_light_weather_confirmation(fog_lights.weather_condition)
            and not fog_lights.weather_confirmed
        ):
            fog_lights.weather_confirmation_requested = True
            condition = _friendly_weather(fog_lights.weather_condition)
            return NextAction.respond(
                f"It's {condition} right now, so turning on the fog lights needs your confirmation. Do you still want me to turn them on?",
                reason="fog_lights_weather_confirmation",
            )

        if not fog_lights.exterior_lights_checked:
            if not tool_index.has("get_exterior_lights_status"):
                return NextAction.respond(
                    "I can't safely turn on the fog lights because the exterior light status check is unavailable right now.",
                    reason="fog_lights_missing_exterior_status_tool",
                )
            return NextAction.tool_call(
                "get_exterior_lights_status",
                reason="fog_lights_exterior_status_check",
            )

        if fog_lights.low_beams_on is None or fog_lights.high_beams_on is None:
            return NextAction.respond(
                "I can't safely turn on the fog lights because I couldn't determine the current headlight status.",
                reason="fog_lights_unknown_headlight_status",
            )

        if fog_lights.low_beams_on is False:
            if not tool_index.has("set_head_lights_low_beams"):
                return NextAction.respond(
                    "I can't safely turn on the fog lights because the low beam control is unavailable right now.",
                    reason="fog_lights_missing_low_beam_tool",
                )
            return NextAction.tool_call(
                "set_head_lights_low_beams",
                {"on": True},
                reason="fog_lights_enable_low_beams",
            )

        if fog_lights.high_beams_on is True:
            if not tool_index.has("set_head_lights_high_beams"):
                return NextAction.respond(
                    "I can't safely turn on the fog lights because the high beam control is unavailable right now.",
                    reason="fog_lights_missing_high_beam_tool",
                )
            if not fog_lights.high_beam_confirmed:
                fog_lights.high_beam_confirmation_requested = True
                return NextAction.respond(
                    "The high beams are on, and I need to turn them off before turning on the fog lights. Please confirm if you want me to continue.",
                    reason="fog_lights_high_beam_confirmation",
                )
            return NextAction.tool_call(
                "set_head_lights_high_beams",
                {"on": False},
                reason="fog_lights_disable_high_beams",
            )

        return NextAction.tool_call(
            "set_fog_lights",
            {"on": True},
            reason="fog_lights_on",
        )

    def _next_defrost_action(
        self, state: ControllerState, tool_index: ToolIndex
    ) -> NextAction | None:
        defrost = state.defrost

        if defrost.completed:
            target = _friendly_defrost_window(defrost.defrost_window)
            on = defrost.on
            state.defrost = DefrostFlow()
            return NextAction.respond(
                f"Done, {target} defrost is on."
                if on
                else f"Done, {target} defrost is off.",
                reason="defrost_done",
            )

        if not tool_index.has("set_window_defrost"):
            return NextAction.respond(
                "I can't change window defrost because that control is unavailable right now.",
                reason="defrost_missing_tool",
            )

        if defrost.defrost_window is None:
            return NextAction.respond(
                "Which window defrost should I set: front, rear, or all?",
                reason="defrost_user_disambiguation",
            )

        if not defrost.on or defrost.defrost_window == "REAR":
            return NextAction.tool_call(
                "set_window_defrost",
                {"on": defrost.on, "defrost_window": defrost.defrost_window},
                reason="defrost_set_simple",
            )

        if not defrost.climate_checked:
            if not tool_index.has("get_climate_settings"):
                return NextAction.respond(
                    "I can't safely turn on front defrost because the climate settings check is unavailable right now.",
                    reason="defrost_missing_climate_tool",
                )
            return NextAction.tool_call(
                "get_climate_settings",
                reason="defrost_climate_check",
            )

        needs_ac = defrost.air_conditioning is not True

        if needs_ac and not defrost.windows_checked:
            if not tool_index.has("get_vehicle_window_positions"):
                return NextAction.respond(
                    "I can't safely turn on front defrost because the window position check is unavailable right now.",
                    reason="defrost_missing_window_positions_tool",
                )
            return NextAction.tool_call(
                "get_vehicle_window_positions",
                reason="defrost_window_position_check",
            )

        if needs_ac and _defrost_window_status_unknown(defrost):
            return NextAction.respond(
                "I can't safely turn on front defrost because I couldn't determine all window positions before enabling air conditioning.",
                reason="defrost_unknown_window_positions",
            )

        if needs_ac and _any_defrost_window_open_over(defrost, 20):
            if not tool_index.has("open_close_window"):
                return NextAction.respond(
                    "I can't safely turn on front defrost because open windows need to be closed first and the window control is unavailable.",
                    reason="defrost_missing_window_control",
                )
            return NextAction.tool_call(
                "open_close_window",
                {"window": "ALL", "percentage": 0},
                reason="defrost_close_windows_before_ac",
            )

        if (defrost.fan_speed or 0) < 2:
            if not tool_index.has("set_fan_speed"):
                return NextAction.respond(
                    "I can't safely turn on front defrost because the fan speed control is unavailable right now.",
                    reason="defrost_missing_fan_speed_tool",
                )
            return NextAction.tool_call(
                "set_fan_speed",
                {"level": 2},
                reason="defrost_raise_fan_speed",
            )

        if not _airflow_includes_windshield(defrost.fan_airflow_direction):
            if not tool_index.has("set_fan_airflow_direction"):
                return NextAction.respond(
                    "I can't safely turn on front defrost because the airflow direction control is unavailable right now.",
                    reason="defrost_missing_airflow_tool",
                )
            return NextAction.tool_call(
                "set_fan_airflow_direction",
                {"direction": "WINDSHIELD"},
                reason="defrost_set_windshield_airflow",
            )

        if needs_ac:
            if not tool_index.has("set_air_conditioning"):
                return NextAction.respond(
                    "I can't safely turn on front defrost because the air conditioning control is unavailable right now.",
                    reason="defrost_missing_ac_tool",
                )
            return NextAction.tool_call(
                "set_air_conditioning",
                {"on": True},
                reason="defrost_enable_ac",
            )

        return NextAction.tool_call(
            "set_window_defrost",
            {"on": True, "defrost_window": defrost.defrost_window},
            reason="defrost_set",
        )

    def _next_route_energy_action(
        self, state: ControllerState, tool_index: ToolIndex
    ) -> NextAction | None:
        charging = state.route_energy

        if charging.completed:
            _remember_route_energy_selection(state, charging)
            message = _format_route_energy_completion(charging)
            state.route_energy = RouteEnergyFlow()
            return NextAction.respond(message, reason="route_energy_done")

        if charging.failure_message:
            message = charging.failure_message
            state.route_energy = RouteEnergyFlow()
            return NextAction.respond(message, reason="route_energy_failed")

        if charging.route_choice_requested and charging.route_detail_requested:
            charging.route_detail_requested = False
            return NextAction.respond(
                _format_route_alternative_detail_prompt(charging.routes),
                reason="route_energy_route_alternative_details",
            )

        if charging.needs_current_navigation and not charging.current_navigation_checked:
            if not tool_index.has("get_current_navigation_state"):
                return NextAction.respond(
                    "I need the current navigation state before I can check charging needs along the route.",
                    reason="route_energy_missing_current_navigation_tool",
                )
            return NextAction.tool_call(
                "get_current_navigation_state",
                {"detailed_information": True},
                reason="route_energy_current_navigation_check",
            )

        if charging.destination_id is None and charging.destination_name is not None:
            if not tool_index.has("get_location_id_by_location_name"):
                return NextAction.respond(
                    "I can't look up that destination because location search is unavailable right now.",
                    reason="route_energy_missing_location_tool",
                )
            return NextAction.tool_call(
                "get_location_id_by_location_name",
                {"location": charging.destination_name},
                reason="route_energy_destination_lookup",
            )

        if charging.wants_navigation_setup:
            setup_action = self._next_route_energy_navigation_setup_action(
                charging, state, tool_index
            )
            if setup_action is not None:
                return setup_action

        if (
            charging.destination_id is not None
            and charging.route_id is None
            and (charging.wants_current_range or charging.wants_battery_sufficiency)
        ):
            route_action = self._prepare_route_energy_route(charging, state, tool_index)
            if route_action is not None:
                return route_action

        if charging.needs_charging_status and not charging.charging_status_checked:
            if not tool_index.has("get_charging_specs_and_status"):
                return NextAction.respond(
                    "I can't check the vehicle charging status because that information is unavailable right now.",
                    reason="route_energy_missing_charging_status_tool",
                )
            return NextAction.tool_call(
                "get_charging_specs_and_status",
                reason="route_energy_charging_status",
            )

        if charging.wants_stop_count and (
            charging.wants_charger_search or charging.destination_id is not None
        ):
            route_action = self._prepare_route_energy_route(charging, state, tool_index)
            if route_action is not None:
                return route_action

        if charging.wants_range and not charging.distance_checked:
            if charging.initial_soc is None and charging.state_of_charge is not None:
                charging.initial_soc = charging.state_of_charge
            if charging.initial_soc is None or charging.final_soc is None:
                return NextAction.respond(
                    "What battery percentage range should I calculate?",
                    reason="route_energy_missing_soc_range",
                )
            if not tool_index.has("get_distance_by_soc"):
                return NextAction.respond(
                    "I can't calculate battery range because that range calculation tool is unavailable right now.",
                    reason="route_energy_missing_distance_tool",
                )
            return NextAction.tool_call(
                "get_distance_by_soc",
                {
                    "initial_state_of_charge": _json_number(charging.initial_soc),
                    "final_state_of_charge": _json_number(charging.final_soc),
                },
                reason="route_energy_distance_by_soc",
            )

        if charging.wants_charger_search or charging.destination_id is not None:
            route_action = self._prepare_route_energy_route(charging, state, tool_index)
            if route_action is not None:
                return route_action

        if charging.wants_charger_search and not charging.pois_checked:
            if charging.search_mode == "nearby":
                if state.runtime.location_id is None:
                    return NextAction.respond(
                        "I need the current location before I can search for charging stations nearby.",
                        reason="route_energy_missing_current_location",
                    )
                if not tool_index.has("search_poi_at_location"):
                    return NextAction.respond(
                        "I can't search for charging stations because place search is unavailable right now.",
                        reason="route_energy_missing_nearby_poi_tool",
                    )
                arguments: dict[str, Any] = {
                    "category_poi": "charging_stations",
                    "location_id": state.runtime.location_id,
                }
                if charging.filters and _tool_argument_available(
                    tool_index, "search_poi_at_location", "filters"
                ):
                    arguments["filters"] = charging.filters
                return NextAction.tool_call(
                    "search_poi_at_location",
                    arguments,
                    reason="route_energy_nearby_charger_search",
                )

            if charging.route_id is None:
                if not charging.current_navigation_checked and tool_index.has(
                    "get_current_navigation_state"
                ):
                    charging.needs_current_navigation = True
                    return NextAction.tool_call(
                        "get_current_navigation_state",
                        {"detailed_information": True},
                        reason="route_energy_current_navigation_for_route",
                    )
                return NextAction.respond(
                    "I need a route before I can search for charging stations along it.",
                    reason="route_energy_missing_route_id",
                )

            if charging.at_kilometer is None:
                return NextAction.respond(
                    "At about what distance along the route should I search for charging stations?",
                    reason="route_energy_missing_route_position",
                )

            if not tool_index.has("search_poi_along_the_route"):
                return NextAction.respond(
                    "I can't search for charging stations along the route because that search capability is unavailable right now.",
                    reason="route_energy_missing_along_route_poi_tool",
                )
            arguments = {
                "category_poi": "charging_stations",
                "route_id": charging.route_id,
                "at_kilometer": _json_number(charging.at_kilometer),
            }
            if charging.filters and _tool_argument_available(
                tool_index, "search_poi_along_the_route", "filters"
            ):
                arguments["filters"] = charging.filters
            return NextAction.tool_call(
                "search_poi_along_the_route",
                arguments,
                reason="route_energy_along_route_charger_search",
            )

        if charging.wants_charging_time and not charging.charging_time_checked:
            if charging.selected_poi_id is None or charging.selected_plug_id is None:
                selected = _select_route_energy_charging_option(charging)
                if selected is None:
                    return NextAction.respond(
                        "Which charging station should I use for the charging time calculation?",
                        reason="route_energy_missing_charging_station_choice",
                    )
                _record_route_energy_selected_poi(charging, selected)

            if charging.target_soc is None:
                return NextAction.respond(
                    "What target battery percentage should I calculate charging time for?",
                    reason="route_energy_missing_target_soc",
                )

            if (
                not charging.charging_status_checked
                and tool_index.has("get_charging_specs_and_status")
            ):
                charging.needs_charging_status = True
                return NextAction.tool_call(
                    "get_charging_specs_and_status",
                    reason="route_energy_charging_status_for_time",
                )

            if charging.start_soc_for_charging is None:
                charging.start_soc_for_charging = _route_energy_start_soc(charging)
            if charging.start_soc_for_charging is None:
                if not charging.charging_status_checked and tool_index.has(
                    "get_charging_specs_and_status"
                ):
                    charging.needs_charging_status = True
                    return NextAction.tool_call(
                        "get_charging_specs_and_status",
                        reason="route_energy_charging_status_for_time",
                    )
                return NextAction.respond(
                    "I need the starting battery percentage before I can calculate charging time.",
                    reason="route_energy_missing_start_soc",
                )

            if not tool_index.has("calculate_charging_time_by_soc"):
                return NextAction.respond(
                    "I can't calculate charging time because that calculation tool is unavailable right now.",
                    reason="route_energy_missing_charging_time_tool",
                )
            return NextAction.tool_call(
                "calculate_charging_time_by_soc",
                {
                    "charging_station_id": charging.selected_poi_id,
                    "charging_station_plug_id": charging.selected_plug_id,
                    "start_state_of_charge": _json_number(
                        charging.start_soc_for_charging
                    ),
                    "target_state_of_charge": _json_number(charging.target_soc),
                },
                reason="route_energy_charging_time",
            )

        if charging.wants_call and not charging.call_completed:
            if charging.selected_phone_number is None:
                selected = _select_route_energy_charging_option(charging)
                if selected is not None:
                    _record_route_energy_selected_poi(charging, selected)
            if charging.selected_phone_number is None:
                return NextAction.respond(
                    "I can't call the charging station because I don't have a phone number for it.",
                    reason="route_energy_missing_phone_number",
                )
            if not tool_index.has("call_phone_by_number"):
                return NextAction.respond(
                    "I can't call the charging station because phone calling is unavailable right now.",
                    reason="route_energy_missing_phone_tool",
                )
            return NextAction.tool_call(
                "call_phone_by_number",
                {"phone_number": charging.selected_phone_number},
                reason="route_energy_call_charging_station",
            )

        charging.completed = True
        return self._next_route_energy_action(state, tool_index)

    def _next_route_energy_navigation_setup_action(
        self,
        charging: RouteEnergyFlow,
        state: ControllerState,
        tool_index: ToolIndex,
    ) -> NextAction | None:
        if charging.destination_id is None:
            return NextAction.respond(
                "Which destination should I use for the navigation?",
                reason="route_energy_navigation_missing_destination",
            )

        if charging.selected_poi_id is None and (
            charging.wants_charger_search or charging.companion_poi_category
        ):
            route_action = self._prepare_route_energy_route(charging, state, tool_index)
            if route_action is not None:
                return route_action

            if charging.route_id is None:
                return NextAction.respond(
                    "I need a route before I can select a charging stop.",
                    reason="route_energy_navigation_missing_route_for_stop",
                )

            _ensure_route_energy_search_kilometers(charging, state)
            if charging.at_kilometer is None and charging.search_kilometers:
                charging.at_kilometer = charging.search_kilometers[
                    charging.search_kilometer_index
                ]
            if charging.at_kilometer is None:
                return NextAction.respond(
                    "At about what distance along the route should I search for the charging stop?",
                    reason="route_energy_navigation_missing_route_position",
                )

            if (
                charging.companion_poi_category
                and not charging.companion_pois_checked
            ):
                if not tool_index.has("search_poi_along_the_route"):
                    label = _friendly_poi_category(charging.companion_poi_category)
                    return NextAction.respond(
                        f"I can't search for {label} along the route because that search capability is unavailable right now.",
                        reason="route_energy_navigation_missing_companion_poi_tool",
                    )
                arguments: dict[str, Any] = {
                    "category_poi": charging.companion_poi_category,
                    "route_id": charging.route_id,
                    "at_kilometer": _json_number(charging.at_kilometer),
                }
                if (
                    charging.companion_poi_filters
                    and charging.arrival_window_start_minutes is None
                    and _tool_argument_available(
                        tool_index, "search_poi_along_the_route", "filters"
                    )
                ):
                    arguments["filters"] = charging.companion_poi_filters
                charging.poi_search_category = charging.companion_poi_category
                return NextAction.tool_call(
                    "search_poi_along_the_route",
                    arguments,
                    reason="route_energy_navigation_companion_poi_search",
                )

            if (
                charging.companion_poi_category
                and charging.companion_pois_checked
                and not charging.companion_pois
            ):
                if _advance_route_energy_search_kilometer(charging):
                    return self._next_route_energy_navigation_setup_action(
                        charging, state, tool_index
                    )
                label = _friendly_poi_category(charging.companion_poi_category)
                return NextAction.respond(
                    f"I couldn't find matching {label} near a charging stop on the route.",
                    reason="route_energy_navigation_no_companion_poi",
                )

            if not charging.pois_checked:
                if not tool_index.has("search_poi_along_the_route"):
                    return NextAction.respond(
                        "I can't search for charging stations along the route because that search capability is unavailable right now.",
                        reason="route_energy_navigation_missing_along_route_poi_tool",
                    )
                arguments: dict[str, Any] = {
                    "category_poi": "charging_stations",
                    "route_id": charging.route_id,
                    "at_kilometer": _json_number(charging.at_kilometer),
                }
                if charging.filters and _tool_argument_available(
                    tool_index, "search_poi_along_the_route", "filters"
                ):
                    arguments["filters"] = charging.filters
                charging.poi_search_category = "charging_stations"
                return NextAction.tool_call(
                    "search_poi_along_the_route",
                    arguments,
                    reason="route_energy_navigation_charging_stop_search",
                )

            selected = _select_route_energy_charging_option(charging)
            if selected is not None:
                _record_route_energy_selected_poi(charging, selected)

        if charging.selected_poi_id is None:
            selected = _select_route_energy_charging_option(charging)
            if selected is not None:
                _record_route_energy_selected_poi(charging, selected)
        if charging.selected_poi_id is None:
            return NextAction.respond(
                "Which charging station should I add as the stop?",
                reason="route_energy_navigation_missing_charging_stop",
            )

        start_id = state.runtime.location_id
        if start_id is None:
            return NextAction.respond(
                "I need the current location before I can set up that navigation.",
                reason="route_energy_navigation_missing_start",
            )

        if not charging.setup_to_poi_checked:
            if not tool_index.has("get_routes_from_start_to_destination"):
                return NextAction.respond(
                    "I can't find the route to the charging station because route search is unavailable right now.",
                    reason="route_energy_navigation_missing_route_tool",
                )
            charging.route_lookup = "setup_to_poi"
            return NextAction.tool_call(
                "get_routes_from_start_to_destination",
                {
                    "start_id": start_id,
                    "destination_id": charging.selected_poi_id,
                },
                reason="route_energy_navigation_route_to_stop",
            )

        if not charging.setup_from_poi_checked:
            if not tool_index.has("get_routes_from_start_to_destination"):
                return NextAction.respond(
                    "I can't find the route from the charging station because route search is unavailable right now.",
                    reason="route_energy_navigation_missing_route_tool",
                )
            charging.route_lookup = "setup_from_poi"
            return NextAction.tool_call(
                "get_routes_from_start_to_destination",
                {
                    "start_id": charging.selected_poi_id,
                    "destination_id": charging.destination_id,
                },
                reason="route_energy_navigation_route_from_stop",
            )

        route_to_poi = _select_requested_route(
            charging.routes_to_poi,
            charging.station_route_preference or "fastest",
            None,
        )
        if route_to_poi is None and charging.routes_to_poi:
            route_to_poi = charging.routes_to_poi[0]

        route_from_poi_index = (
            _select_route_index_by_via(
                charging.routes_from_poi, charging.final_route_via
            )
            if charging.final_route_via
            else None
        )
        route_from_poi = _select_requested_route(
            charging.routes_from_poi,
            charging.route_preference,
            (
                route_from_poi_index
                if route_from_poi_index is not None
                else charging.selected_route_index
            ),
        )
        if route_from_poi is None and len(charging.routes_from_poi) == 1:
            route_from_poi = charging.routes_from_poi[0]

        if route_to_poi is None or route_from_poi is None:
            return NextAction.respond(
                _format_route_choice_prompt(charging.routes_from_poi),
                reason="route_energy_navigation_route_disambiguation",
            )

        route_to_poi_id = route_to_poi.get("route_id")
        route_from_poi_id = route_from_poi.get("route_id")
        if not isinstance(route_to_poi_id, str) or not isinstance(
            route_from_poi_id, str
        ):
            return NextAction.respond(
                "I can't safely set that navigation because a route ID is missing.",
                reason="route_energy_navigation_missing_route_id",
            )

        if not tool_index.has("set_new_navigation"):
            return NextAction.respond(
                "I can't start navigation because that control is unavailable right now.",
                reason="route_energy_navigation_missing_set_tool",
            )

        charging.completion_message = _format_route_energy_navigation_completion(
            charging, route_to_poi, route_from_poi
        )
        return NextAction.tool_call(
            "set_new_navigation",
            {"route_ids": [route_to_poi_id, route_from_poi_id]},
            reason="route_energy_navigation_set",
        )

    def _prepare_route_energy_route(
        self,
        charging: RouteEnergyFlow,
        state: ControllerState,
        tool_index: ToolIndex,
    ) -> NextAction | None:
        if charging.route_id is not None:
            return None

        _resolve_route_energy_route_id(charging)
        if charging.route_id is not None:
            return None

        if charging.destination_id is None:
            return None

        if charging.route_start_id is None:
            charging.route_start_id = _route_energy_start_id(state, charging)
            if charging.route_start_id is None:
                return NextAction.respond(
                    "I need the current location before I can find a route.",
                    reason="route_energy_missing_route_start",
                )

        if not charging.routes_checked:
            if not tool_index.has("get_routes_from_start_to_destination"):
                return NextAction.respond(
                    "I can't find a route because route search is unavailable right now.",
                    reason="route_energy_missing_route_tool",
                )
            charging.route_lookup = "destination"
            return NextAction.tool_call(
                "get_routes_from_start_to_destination",
                {
                    "start_id": charging.route_start_id,
                    "destination_id": charging.destination_id,
                },
                reason="route_energy_route_lookup",
            )

        selected_route = _select_requested_route(
            charging.routes,
            charging.route_preference,
            charging.selected_route_index,
        )
        if (
            selected_route is None
            and charging.route_preference is None
            and (charging.wants_charging_time or charging.wants_charger_search)
        ):
            selected_route = _select_route(charging.routes, "fastest")
            if selected_route is not None:
                charging.route_preference = "fastest"
        if selected_route is None:
            if len(charging.routes) == 1:
                selected_route = charging.routes[0]
            else:
                charging.route_choice_requested = True
                return NextAction.respond(
                    _format_route_choice_prompt(charging.routes),
                    reason="route_energy_route_disambiguation",
                )

        route_id = selected_route.get("route_id")
        if not isinstance(route_id, str) or not route_id:
            return NextAction.respond(
                "I can't use that route because the route ID is missing.",
                reason="route_energy_missing_selected_route_id",
            )
        charging.route_id = route_id
        return None

    def _next_pending_navigation_setup_action(
        self, state: ControllerState, tool_index: ToolIndex
    ) -> NextAction | None:
        route_ids = [
            route_id
            for route_id in state.pending_set_navigation_route_ids
            if isinstance(route_id, str) and route_id
        ]
        state.pending_set_navigation_route_ids = []
        if not route_ids:
            return NextAction.respond(
                "I don't have a planned route ready to start.",
                reason="planned_navigation_missing_routes",
            )
        if not tool_index.has("set_new_navigation"):
            return NextAction.respond(
                "I can't start navigation because that control is unavailable right now.",
                reason="planned_navigation_missing_set_tool",
            )
        state.navigation = NavigationFlow(
            active=True,
            routes_to_final_destination_id=list(route_ids),
            completion_message="Done, navigation is started.",
        )
        state.recent_navigation_route_ids = list(route_ids)
        state.planned_navigation_route_ids = []
        state.planned_navigation_waypoints = []
        state.planned_navigation_selected_routes = []
        state.planned_navigation_considered_routes = []
        return NextAction.tool_call(
            "set_new_navigation",
            {"route_ids": route_ids},
            reason="planned_navigation_set",
        )

    def _next_multi_stop_navigation_action(
        self, state: ControllerState, tool_index: ToolIndex
    ) -> NextAction | None:
        multi_stop = state.multi_stop_navigation

        if multi_stop.completed:
            message = multi_stop.completion_message or "Done, navigation is started."
            state.multi_stop_navigation = MultiStopNavigationFlow()
            return NextAction.respond(message, reason="multi_stop_navigation_done")

        if multi_stop.failure_message:
            message = multi_stop.failure_message
            state.multi_stop_navigation = MultiStopNavigationFlow()
            return NextAction.respond(message, reason="multi_stop_navigation_failed")

        if multi_stop.first_destination_id is None:
            if not tool_index.has("get_location_id_by_location_name"):
                return NextAction.respond(
                    "I can't look up that destination because location search is unavailable right now.",
                    reason="multi_stop_navigation_missing_location_tool",
                )
            if not multi_stop.first_destination_name:
                return NextAction.respond(
                    "Which first stop should I use?",
                    reason="multi_stop_navigation_missing_first_destination",
                )
            return NextAction.tool_call(
                "get_location_id_by_location_name",
                {"location": multi_stop.first_destination_name},
                reason="multi_stop_navigation_lookup_first_destination",
            )

        if multi_stop.final_destination_id is None:
            if not tool_index.has("get_location_id_by_location_name"):
                return NextAction.respond(
                    "I can't look up that destination because location search is unavailable right now.",
                    reason="multi_stop_navigation_missing_location_tool",
                )
            if not multi_stop.final_destination_name:
                return NextAction.respond(
                    "Which final destination should I use?",
                    reason="multi_stop_navigation_missing_final_destination",
                )
            return NextAction.tool_call(
                "get_location_id_by_location_name",
                {"location": multi_stop.final_destination_name},
                reason="multi_stop_navigation_lookup_final_destination",
            )

        start_id = state.runtime.location_id
        if start_id is None:
            return NextAction.respond(
                "I need the current location before I can find a route.",
                reason="multi_stop_navigation_missing_start",
            )

        if not tool_index.has("get_routes_from_start_to_destination"):
            return NextAction.respond(
                "I can't find a route because route search is unavailable right now.",
                reason="multi_stop_navigation_missing_route_tool",
            )

        if not multi_stop.first_routes_checked:
            return NextAction.tool_call(
                "get_routes_from_start_to_destination",
                {
                    "start_id": start_id,
                    "destination_id": multi_stop.first_destination_id,
                },
                reason="multi_stop_navigation_first_route_lookup",
            )

        if not multi_stop.final_routes_checked:
            return NextAction.tool_call(
                "get_routes_from_start_to_destination",
                {
                    "start_id": multi_stop.first_destination_id,
                    "destination_id": multi_stop.final_destination_id,
                },
                reason="multi_stop_navigation_final_route_lookup",
            )

        first_route = _select_requested_route(
            multi_stop.first_routes,
            multi_stop.route_preference,
            None,
        )
        if first_route is None and len(multi_stop.first_routes) == 1:
            first_route = multi_stop.first_routes[0]
        elif first_route is None and _route_choice_is_unambiguous(
            multi_stop.first_routes
        ):
            first_route = _select_route(multi_stop.first_routes, "fastest")

        final_route = _select_requested_route(
            multi_stop.final_routes,
            multi_stop.route_preference,
            None,
        )
        if final_route is None and len(multi_stop.final_routes) == 1:
            final_route = multi_stop.final_routes[0]
        elif final_route is None and _route_choice_is_unambiguous(
            multi_stop.final_routes
        ):
            final_route = _select_route(multi_stop.final_routes, "fastest")

        if first_route is None or final_route is None:
            routes = (
                multi_stop.first_routes if first_route is None else multi_stop.final_routes
            )
            return NextAction.respond(
                _format_route_choice_prompt(routes),
                reason="multi_stop_navigation_route_disambiguation",
            )

        first_route_id = first_route.get("route_id")
        final_route_id = final_route.get("route_id")
        if not isinstance(first_route_id, str) or not isinstance(final_route_id, str):
            return NextAction.respond(
                "I can't safely start navigation because a route segment ID is missing.",
                reason="multi_stop_navigation_missing_route_id",
            )

        if not tool_index.has("set_new_navigation"):
            return NextAction.respond(
                "I can't start navigation because that control is unavailable right now.",
                reason="multi_stop_navigation_missing_set_tool",
            )

        route_ids = [first_route_id, final_route_id]
        state.recent_navigation_route_ids = list(route_ids)
        state.recent_navigation_waypoints = [
            start_id,
            multi_stop.first_destination_id,
            multi_stop.final_destination_id,
        ]
        multi_stop.completion_message = _navigation_multi_route_completion_message(
            "navigation is started",
            [first_route, final_route],
            multi_stop.route_preference,
            multi_stop.first_routes + multi_stop.final_routes,
        )
        return NextAction.tool_call(
            "set_new_navigation",
            {"route_ids": route_ids},
            reason="multi_stop_navigation_set",
        )

    def _next_poi_action(
        self, state: ControllerState, tool_index: ToolIndex
    ) -> NextAction | None:
        poi = state.poi

        if poi.completed:
            message = poi.completion_message or "Done, the navigation is updated."
            state.poi = POIFlow()
            return NextAction.respond(message, reason="poi_done")

        if poi.failure_message:
            message = poi.failure_message
            state.poi = POIFlow()
            return NextAction.respond(message, reason="poi_failed")

        if poi.search_along_route:
            if poi.route_id is None and not poi.current_navigation_checked:
                if not tool_index.has("get_current_navigation_state"):
                    return NextAction.respond(
                        "I need the current route before I can search along it.",
                        reason="poi_along_route_missing_current_nav_tool",
                    )
                return NextAction.tool_call(
                    "get_current_navigation_state",
                    {"detailed_information": False},
                    reason="poi_along_route_current_navigation",
                )
            if poi.route_id is None:
                return NextAction.respond(
                    "I need a route before I can search along it.",
                    reason="poi_along_route_missing_route",
                )
            if not poi.pois_checked:
                if not tool_index.has("search_poi_along_the_route"):
                    return NextAction.respond(
                        "I can't search for places along the route because that route search is unavailable right now.",
                        reason="poi_missing_along_route_search_tool",
                    )
                return NextAction.tool_call(
                    "search_poi_along_the_route",
                    {"category_poi": poi.category, "route_id": poi.route_id},
                    reason="poi_along_route_search",
                )

            if poi.selected_poi_id is None:
                if len(poi.pois) == 1:
                    selected = poi.pois[0]
                    poi.selected_poi_id = selected.get("id")
                    poi.selected_poi_name = selected.get("name")
                else:
                    return NextAction.respond(
                        _format_poi_choice_prompt(poi.category, poi.pois),
                        reason="poi_along_route_choice",
                    )

            if not isinstance(poi.selected_poi_id, str) or not poi.selected_poi_id:
                return NextAction.respond(
                    "Which place would you like me to add as a stop?",
                    reason="poi_along_route_missing_choice",
                )

            if poi.do_not_set_navigation:
                stop_name = poi.selected_poi_name or "that place"
                poi.completed = True
                poi.completion_message = f"{stop_name} is available along the route."
                return self._next_poi_action(state, tool_index)

            if poi.route_start_id is None or poi.route_end_id is None:
                return NextAction.respond(
                    "I need the current route waypoints before I can add a stop along it.",
                    reason="poi_along_route_missing_adjacent_waypoints",
                )

            if not tool_index.has("get_routes_from_start_to_destination"):
                return NextAction.respond(
                    "I can't find the route segments for that stop because route search is unavailable right now.",
                    reason="poi_along_route_missing_route_tool",
                )

            if not poi.routes_checked:
                poi.route_lookup = "to_poi"
                return NextAction.tool_call(
                    "get_routes_from_start_to_destination",
                    {
                        "start_id": poi.route_start_id,
                        "destination_id": poi.selected_poi_id,
                    },
                    reason="poi_along_route_to_stop_lookup",
                )

            if not poi.routes_from_poi_checked:
                poi.route_lookup = "from_poi"
                return NextAction.tool_call(
                    "get_routes_from_start_to_destination",
                    {
                        "start_id": poi.selected_poi_id,
                        "destination_id": poi.route_end_id,
                    },
                    reason="poi_along_route_from_stop_lookup",
                )

            route_preference_defaulted = poi.route_preference is None
            route_preference = poi.route_preference or "fastest"
            route_to_poi = _select_requested_route(
                poi.routes,
                route_preference,
                poi.selected_route_index,
            )
            if route_to_poi is None and route_preference_defaulted and poi.routes:
                route_to_poi = _select_route(poi.routes, "fastest") or poi.routes[0]
            elif route_to_poi is None and len(poi.routes) == 1:
                route_to_poi = poi.routes[0]

            route_from_poi = _select_requested_route(
                poi.routes_from_poi,
                route_preference,
                poi.selected_route_index,
            )
            if (
                route_from_poi is None
                and route_preference_defaulted
                and poi.routes_from_poi
            ):
                route_from_poi = (
                    _select_route(poi.routes_from_poi, "fastest")
                    or poi.routes_from_poi[0]
                )
            elif route_from_poi is None and len(poi.routes_from_poi) == 1:
                route_from_poi = poi.routes_from_poi[0]

            if route_to_poi is None or route_from_poi is None:
                poi.route_choice_requested = True
                routes = poi.routes or poi.routes_from_poi
                return NextAction.respond(
                    _format_route_choice_prompt(routes),
                    reason="poi_along_route_route_disambiguation",
                )

            route_to_poi_id = route_to_poi.get("route_id")
            route_from_poi_id = route_from_poi.get("route_id")
            if not isinstance(route_to_poi_id, str) or not isinstance(
                route_from_poi_id, str
            ):
                return NextAction.respond(
                    "I can't safely add that stop because a route segment ID is missing.",
                    reason="poi_along_route_missing_route_id",
                )

            if not tool_index.has("navigation_add_one_waypoint"):
                return NextAction.respond(
                    "I can't add a waypoint because that navigation edit control is unavailable right now.",
                    reason="poi_along_route_missing_add_waypoint_tool",
                )

            poi.completed = True
            stop_name = poi.selected_poi_name or "that place"
            poi.completion_message = f"Done, I added {stop_name} as a stop along the route."
            return NextAction.tool_call(
                "navigation_add_one_waypoint",
                {
                    "waypoint_id_to_add": poi.selected_poi_id,
                    "waypoint_id_before_new_waypoint": poi.route_start_id,
                    "waypoint_id_after_new_waypoint": poi.route_end_id,
                    "route_id_leading_to_new_waypoint": route_to_poi_id,
                    "route_id_leading_away_from_new_waypoint": route_from_poi_id,
                },
                reason="poi_along_route_add_waypoint",
            )

        if poi.location_id is None:
            if poi.location_name is None:
                return NextAction.respond(
                    "Which location should I search in?",
                    reason="poi_missing_location",
                )
            if not tool_index.has("get_location_id_by_location_name"):
                return NextAction.respond(
                    "I can't look up that location because location search is unavailable right now.",
                    reason="poi_missing_location_tool",
                )
            return NextAction.tool_call(
                "get_location_id_by_location_name",
                {"location": poi.location_name},
                reason="poi_location_lookup",
            )

        if not poi.pois_checked:
            if not tool_index.has("search_poi_at_location"):
                return NextAction.respond(
                    "I can't search for places because place search is unavailable right now.",
                    reason="poi_missing_search_tool",
                )
            return NextAction.tool_call(
                "search_poi_at_location",
                {"category_poi": poi.category, "location_id": poi.location_id},
                reason="poi_search",
            )

        if poi.selected_poi_id is None:
            if len(poi.pois) == 1 or poi.route_prefix_ids:
                selected = poi.pois[0]
                poi.selected_poi_id = selected.get("id")
                poi.selected_poi_name = selected.get("name")
            else:
                return NextAction.respond(
                    _format_poi_choice_prompt(poi.category, poi.pois),
                    reason="poi_choice",
                )

        if poi.selected_poi_id is None:
            return NextAction.respond(
                "Which place would you like directions to?",
                reason="poi_missing_choice",
            )

        if not poi.routes_checked:
            if state.runtime.location_id is None:
                return NextAction.respond(
                    "I need the current location before I can find a route.",
                    reason="poi_missing_start",
                )
            if not tool_index.has("get_routes_from_start_to_destination"):
                return NextAction.respond(
                    "I can't find a route because route search is unavailable right now.",
                    reason="poi_missing_route_tool",
                )
            return NextAction.tool_call(
                "get_routes_from_start_to_destination",
                {
                    "start_id": poi.route_start_id or state.runtime.location_id,
                    "destination_id": poi.selected_poi_id,
                },
                reason="poi_route_lookup",
            )

        selected_route = _select_requested_route(
            poi.routes,
            poi.route_preference,
            poi.selected_route_index,
        )
        if poi.do_not_set_navigation and selected_route is None and poi.routes:
            selected_route = _select_route(poi.routes, "fastest") or poi.routes[0]

        if selected_route is None:
            if len(poi.routes) == 1:
                selected_route = poi.routes[0]
            else:
                poi.route_choice_requested = True
                return NextAction.respond(
                    _format_route_choice_prompt(poi.routes),
                    reason="poi_route_disambiguation",
                )

        route_id = selected_route.get("route_id")
        if not isinstance(route_id, str) or not route_id:
            return NextAction.respond(
                "I can't safely set that route because the route ID is missing.",
                reason="poi_missing_route_id",
            )

        if poi.do_not_set_navigation:
            poi.completed = True
            poi.completion_message = (
                f"The selected route to {poi.selected_poi_name or 'that place'} is "
                f"{_route_summary(selected_route)}."
            )
            return self._next_poi_action(state, tool_index)

        if poi.replace_final_destination:
            if not tool_index.has("navigation_replace_final_destination"):
                return NextAction.respond(
                    "I can't replace the destination because that navigation edit control is unavailable right now.",
                    reason="poi_missing_replace_destination_tool",
                )
            poi.completion_message = _navigation_completion_message(
                "the navigation destination is updated",
                selected_route,
                poi.route_preference,
                poi.routes,
            )
            return NextAction.tool_call(
                "navigation_replace_final_destination",
                {
                    "new_destination_id": poi.selected_poi_id,
                    "route_id_leading_to_new_destination": route_id,
                },
                reason="poi_replace_navigation_destination",
            )

        if poi.route_prefix_ids:
            full_route_ids = poi.route_prefix_ids + [route_id]
            selected_routes = [*poi.route_prefix_selected_routes, selected_route]
            considered_routes = [
                *poi.route_prefix_considered_routes,
                *poi.routes,
            ]
            state.planned_navigation_route_ids = list(full_route_ids)
            state.planned_navigation_selected_routes = list(selected_routes)
            state.planned_navigation_considered_routes = list(considered_routes)
            if poi.route_start_id and poi.selected_poi_id:
                state.planned_navigation_waypoints = [
                    *state.planned_navigation_waypoints,
                    poi.selected_poi_id,
                ]
            state.recent_navigation_route_ids = list(full_route_ids)
            if poi.defer_navigation_setup:
                poi.completed = True
                message_parts = ["Done, I planned the multi-stop route."]
                message_parts.extend(
                    _route_toll_notice_parts(
                        selected_routes,
                        considered_routes,
                        selected_text=(
                            "At least one selected route segment includes toll roads."
                        ),
                        alternative_text=(
                            "An available alternative route segment includes toll roads."
                        ),
                        fastest_alternative_text=(
                            "The fastest route segment includes toll roads, "
                            "so I used a toll-free alternative."
                        ),
                    )
                )
                message_parts.append("Do you want me to start navigation?")
                poi.completion_message = " ".join(message_parts)
                return self._next_poi_action(state, tool_index)

            if not tool_index.has("set_new_navigation"):
                return NextAction.respond(
                    "I can't start navigation because that control is unavailable right now.",
                    reason="poi_missing_set_navigation_tool",
                )
            poi.completed = True
            poi.completion_message = "Done, navigation is started with the added stop."
            return NextAction.tool_call(
                "set_new_navigation",
                {"route_ids": full_route_ids},
                reason="poi_set_prefixed_navigation",
            )

        poi.completed = True
        poi.completion_message = (
            f"The route to {poi.selected_poi_name or 'that place'} is "
            f"{_route_summary(selected_route)}."
        )
        return self._next_poi_action(state, tool_index)

    def _next_navigation_action(
        self, state: ControllerState, tool_index: ToolIndex
    ) -> NextAction | None:
        navigation = state.navigation

        if navigation.completed:
            return NextAction.respond(
                navigation.completion_message or "Done, the navigation is updated.",
                reason="navigation_done",
            )

        if navigation.failure_message:
            return NextAction.respond(
                navigation.failure_message,
                reason="navigation_failed",
            )

        if navigation.route_choice_requested and navigation.route_detail_requested:
            navigation.route_detail_requested = False
            return NextAction.respond(
                _format_route_alternative_detail_prompt(
                    _navigation_choice_routes(navigation)
                ),
                reason="navigation_route_alternative_details",
            )

        if navigation.mode == "delete_waypoint":
            return self._next_navigation_delete_waypoint_action(navigation, tool_index)

        if navigation.mode == "delete_final_destination":
            return self._next_navigation_delete_destination_action(
                navigation, tool_index
            )

        if navigation.mode == "replace_one_waypoint":
            return self._next_navigation_replace_waypoint_action(navigation, tool_index)

        if navigation.destination_id is None:
            if not tool_index.has("get_location_id_by_location_name"):
                return NextAction.respond(
                    "I can't look up that destination because location search is unavailable right now.",
                    reason="navigation_missing_location_tool",
                )
            if not navigation.destination_name:
                return NextAction.respond(
                    "Which destination should I use?",
                    reason="navigation_missing_destination",
                )
            return NextAction.tool_call(
                "get_location_id_by_location_name",
                {"location": navigation.destination_name},
                reason="navigation_lookup_destination",
            )

        if (
            navigation.mode == "replace_final_destination"
            and navigation.needs_current_navigation
            and not navigation.current_navigation_checked
        ):
            if not tool_index.has("get_current_navigation_state"):
                return NextAction.respond(
                    "I can't check the current navigation state because that lookup is unavailable right now, so I can't safely edit this route.",
                    reason="navigation_missing_current_state_tool",
                )
            return NextAction.tool_call(
                "get_current_navigation_state",
                {"detailed_information": False},
                reason="navigation_current_state_check",
            )

        if navigation.route_start_id is None:
            navigation.route_start_id = _navigation_route_start_id(state, navigation)
            if navigation.route_start_id is None:
                return NextAction.respond(
                    "I need the current location before I can find a route.",
                    reason="navigation_missing_start",
                )

        if not navigation.routes_checked:
            if not tool_index.has("get_routes_from_start_to_destination"):
                return NextAction.respond(
                    "I can't find a route because route search is unavailable right now.",
                    reason="navigation_missing_route_tool",
                )
            navigation.route_lookup = "destination"
            return NextAction.tool_call(
                "get_routes_from_start_to_destination",
                {
                    "start_id": navigation.route_start_id,
                    "destination_id": navigation.destination_id,
                },
                reason="navigation_route_lookup",
            )

        selected_route = _select_navigation_requested_route(navigation)
        if selected_route is None:
            if len(navigation.routes) == 1:
                selected_route = navigation.routes[0]
            elif _route_choice_is_unambiguous(navigation.routes):
                selected_route = _select_route(navigation.routes, "fastest")
            else:
                navigation.route_choice_requested = True
                return NextAction.respond(
                    _format_route_choice_prompt(navigation.routes),
                    reason="navigation_route_disambiguation",
                )

        route_id = selected_route.get("route_id")
        if not isinstance(route_id, str) or not route_id:
            return NextAction.respond(
                "I can't safely set that route because the route ID is missing.",
                reason="navigation_missing_route_id",
            )

        if navigation.planning_only:
            start_id = navigation.route_start_id
            destination_id = navigation.destination_id
            if isinstance(start_id, str) and isinstance(destination_id, str):
                state.planned_navigation_route_ids = [route_id]
                state.planned_navigation_waypoints = [start_id, destination_id]
                state.planned_navigation_selected_routes = [selected_route]
                state.planned_navigation_considered_routes = list(navigation.routes)
                state.recent_navigation_route_ids = [route_id]
                state.recent_navigation_waypoints = [start_id, destination_id]
                duration = _route_duration_minutes(selected_route)
                if duration is not None and state.runtime.hour is not None:
                    current_minutes = state.runtime.hour * 60 + (
                        state.runtime.minute or 0
                    )
                    state.planned_navigation_arrival_minutes = (
                        current_minutes + duration
                    ) % (24 * 60)
                pending_poi = state.pending_poi_after_navigation
                if pending_poi is not None:
                    state.pending_poi_after_navigation = None
                    pending_poi.route_prefix_ids = [route_id]
                    pending_poi.route_prefix_selected_routes = [selected_route]
                    pending_poi.route_prefix_considered_routes = list(navigation.routes)
                    pending_poi.route_start_id = (
                        start_id if pending_poi.search_along_route else destination_id
                    )
                    pending_poi.route_end_id = (
                        destination_id if pending_poi.search_along_route else None
                    )
                    pending_poi.route_id = (
                        route_id if pending_poi.search_along_route else None
                    )
                    pending_poi.location_id = (
                        None if pending_poi.search_along_route else destination_id
                    )
                    if (
                        state.planned_navigation_arrival_minutes is not None
                        and "open" in state.last_user_text.lower()
                    ):
                        pending_poi.required_open_at_minutes = (
                            state.planned_navigation_arrival_minutes
                        )
                    state.poi = pending_poi
                    state.navigation = NavigationFlow()
                    return self._next_poi_action(state, tool_index)
            navigation.completed = True
            navigation.completion_message = _navigation_completion_message(
                "the first route is planned",
                selected_route,
                (
                    None
                    if navigation.toll_avoidance_tolerance_minutes is not None
                    else navigation.route_preference
                ),
                navigation.routes,
            )
            return self._next_navigation_action(state, tool_index)

        if navigation.mode == "replace_final_destination":
            if not tool_index.has("navigation_replace_final_destination"):
                return NextAction.respond(
                    "I can't replace the destination because that navigation edit control is unavailable right now.",
                    reason="navigation_missing_replace_tool",
                )
            navigation.completion_message = _navigation_completion_message(
                "the navigation destination is updated",
                selected_route,
                navigation.route_preference,
                navigation.routes,
            )
            return NextAction.tool_call(
                "navigation_replace_final_destination",
                {
                    "new_destination_id": navigation.destination_id,
                    "route_id_leading_to_new_destination": route_id,
                },
                reason="navigation_replace_destination",
            )

        if not tool_index.has("set_new_navigation"):
            return NextAction.respond(
                "I can't start navigation because that control is unavailable right now.",
                reason="navigation_missing_set_tool",
            )
        navigation.completion_message = _navigation_completion_message(
            "navigation is started",
            selected_route,
            (
                None
                if navigation.toll_avoidance_tolerance_minutes is not None
                else navigation.route_preference
            ),
            navigation.routes,
        )
        if navigation.route_start_id and navigation.destination_id:
            navigation.routes_to_final_destination_id = [route_id]
            navigation.waypoints_id = [
                navigation.route_start_id,
                navigation.destination_id,
            ]
            state.recent_navigation_route_ids = [route_id]
            state.recent_navigation_waypoints = list(navigation.waypoints_id)
        return NextAction.tool_call(
            "set_new_navigation",
            {"route_ids": [route_id]},
            reason="navigation_set_new",
        )

    def _next_navigation_delete_destination_action(
        self, navigation: NavigationFlow, tool_index: ToolIndex
    ) -> NextAction | None:
        _resolve_navigation_final_destination_from_state(navigation)

        if (
            navigation.destination_id is None
            and not navigation.current_navigation_checked
        ):
            if not tool_index.has("get_current_navigation_state"):
                return NextAction.respond(
                    "I need the current navigation state before I can safely edit this route.",
                    reason="navigation_missing_current_state_tool",
                )
            return NextAction.tool_call(
                "get_current_navigation_state",
                {"detailed_information": True},
                reason="navigation_current_state_check",
            )

        if navigation.navigation_active is False:
            return NextAction.respond(
                "Navigation is not active right now.",
                reason="navigation_not_active",
            )

        _resolve_navigation_final_destination_from_state(navigation)
        if navigation.destination_id is None:
            if navigation.destination_name and tool_index.has(
                "get_location_id_by_location_name"
            ):
                return NextAction.tool_call(
                    "get_location_id_by_location_name",
                    {"location": navigation.destination_name},
                    reason="navigation_lookup_destination_to_delete",
                )
            return NextAction.respond(
                "Which final destination should I remove from the route?",
                reason="navigation_missing_destination_to_delete",
            )

        if not tool_index.has("navigation_delete_destination"):
            return NextAction.respond(
                "I can't delete the destination because that navigation edit control is unavailable right now.",
                reason="navigation_missing_delete_destination_tool",
            )

        navigation.completion_message = (
            "Done, the final destination is removed from the navigation route."
        )
        return NextAction.tool_call(
            "navigation_delete_destination",
            {"destination_id_to_delete": navigation.destination_id},
            reason="navigation_delete_destination",
        )

    def _next_navigation_delete_waypoint_action(
        self, navigation: NavigationFlow, tool_index: ToolIndex
    ) -> NextAction | None:
        if not navigation.current_navigation_checked:
            if not tool_index.has("get_current_navigation_state"):
                return NextAction.respond(
                    "I can't check the current navigation state because that lookup is unavailable right now, so I can't safely edit this route.",
                    reason="navigation_missing_current_state_tool",
                )
            return NextAction.tool_call(
                "get_current_navigation_state",
                {"detailed_information": True},
                reason="navigation_current_state_check",
            )

        if navigation.navigation_active is False:
            return NextAction.respond(
                "Navigation is not active right now.",
                reason="navigation_not_active",
            )

        _resolve_navigation_waypoint_from_state(navigation)
        if navigation.waypoint_id is None:
            if navigation.waypoint_name:
                return NextAction.respond(
                    f"{navigation.waypoint_name} is not currently an intermediate waypoint in this route.",
                    reason="navigation_waypoint_not_found",
                )
            return NextAction.respond(
                "Which waypoint should I remove from the route?",
                reason="navigation_missing_waypoint",
            )

        adjacent = _navigation_adjacent_waypoints(navigation, navigation.waypoint_id)
        if adjacent is None:
            if (
                navigation.waypoints_id
                and navigation.waypoint_id == navigation.waypoints_id[-1]
            ):
                navigation.mode = "delete_final_destination"
                navigation.destination_id = navigation.waypoint_id
                navigation.destination_name = navigation.waypoint_name
                return self._next_navigation_delete_destination_action(
                    navigation, tool_index
                )
            return NextAction.respond(
                "I can only remove an intermediate waypoint from the current route.",
                reason="navigation_waypoint_not_intermediate",
            )
        previous_waypoint_id, next_waypoint_id = adjacent

        if not navigation.route_without_waypoint_checked:
            if not tool_index.has("get_routes_from_start_to_destination"):
                return NextAction.respond(
                    "I can't find a replacement route because route search is unavailable right now.",
                    reason="navigation_missing_route_tool",
                )
            navigation.route_lookup = "delete_without_waypoint"
            return NextAction.tool_call(
                "get_routes_from_start_to_destination",
                {
                    "start_id": previous_waypoint_id,
                    "destination_id": next_waypoint_id,
                },
                reason="navigation_delete_waypoint_route_lookup",
            )

        selected_route = _select_requested_route(
            navigation.routes_without_waypoint,
            navigation.route_preference,
            navigation.selected_route_index,
        )
        if selected_route is None:
            if len(navigation.routes_without_waypoint) == 1:
                selected_route = navigation.routes_without_waypoint[0]
            else:
                navigation.route_choice_requested = True
                return NextAction.respond(
                    _format_route_choice_prompt(navigation.routes_without_waypoint),
                    reason="navigation_route_disambiguation",
                )

        route_id = selected_route.get("route_id")
        if not isinstance(route_id, str) or not route_id:
            return NextAction.respond(
                "I can't safely edit that route because the replacement route ID is missing.",
                reason="navigation_missing_route_id",
            )

        if not tool_index.has("navigation_delete_waypoint"):
            return NextAction.respond(
                "I can't delete that waypoint because that navigation edit control is unavailable right now.",
                reason="navigation_missing_delete_waypoint_tool",
            )
        navigation.completion_message = _navigation_completion_message(
            "the waypoint is removed from the navigation route",
            selected_route,
            navigation.route_preference,
            navigation.routes_without_waypoint,
        )
        return NextAction.tool_call(
            "navigation_delete_waypoint",
            {
                "waypoint_id_to_delete": navigation.waypoint_id,
                "route_id_without_waypoint": route_id,
            },
            reason="navigation_delete_waypoint",
        )

    def _next_navigation_replace_waypoint_action(
        self, navigation: NavigationFlow, tool_index: ToolIndex
    ) -> NextAction | None:
        if not navigation.current_navigation_checked:
            if not tool_index.has("get_current_navigation_state"):
                return NextAction.respond(
                    "I need the current navigation state before I can safely edit this route.",
                    reason="navigation_missing_current_state_tool",
                )
            return NextAction.tool_call(
                "get_current_navigation_state",
                {"detailed_information": True},
                reason="navigation_current_state_check",
            )

        if navigation.navigation_active is False:
            return NextAction.respond(
                "Navigation is not active right now.",
                reason="navigation_not_active",
            )

        _resolve_navigation_waypoint_from_state(navigation)
        if navigation.waypoint_id is None:
            if navigation.waypoint_name:
                return NextAction.respond(
                    f"{navigation.waypoint_name} is not currently an intermediate waypoint in this route.",
                    reason="navigation_waypoint_not_found",
                )
            return NextAction.respond(
                "Which waypoint should I replace?",
                reason="navigation_missing_waypoint",
            )

        if navigation.new_waypoint_id is None:
            if not navigation.new_waypoint_name:
                return NextAction.respond(
                    "Which new waypoint should I use?",
                    reason="navigation_missing_new_waypoint",
                )
            if not tool_index.has("get_location_id_by_location_name"):
                return NextAction.respond(
                    "I can't look up that waypoint because location search is unavailable right now.",
                    reason="navigation_missing_location_tool",
                )
            return NextAction.tool_call(
                "get_location_id_by_location_name",
                {"location": navigation.new_waypoint_name},
                reason="navigation_lookup_new_waypoint",
            )

        adjacent = _navigation_adjacent_waypoints(navigation, navigation.waypoint_id)
        if adjacent is None:
            return NextAction.respond(
                "I can only replace an intermediate waypoint from the current route.",
                reason="navigation_waypoint_not_intermediate",
            )
        previous_waypoint_id, next_waypoint_id = adjacent

        if not navigation.route_to_new_waypoint_checked:
            if not tool_index.has("get_routes_from_start_to_destination"):
                return NextAction.respond(
                    "I can't find a route to that waypoint because route search is unavailable right now.",
                    reason="navigation_missing_route_tool",
                )
            navigation.route_lookup = "to_new_waypoint"
            return NextAction.tool_call(
                "get_routes_from_start_to_destination",
                {
                    "start_id": previous_waypoint_id,
                    "destination_id": navigation.new_waypoint_id,
                },
                reason="navigation_route_to_new_waypoint_lookup",
            )

        if not navigation.route_from_new_waypoint_checked:
            if not tool_index.has("get_routes_from_start_to_destination"):
                return NextAction.respond(
                    "I can't find a route from that waypoint because route search is unavailable right now.",
                    reason="navigation_missing_route_tool",
                )
            navigation.route_lookup = "from_new_waypoint"
            return NextAction.tool_call(
                "get_routes_from_start_to_destination",
                {
                    "start_id": navigation.new_waypoint_id,
                    "destination_id": next_waypoint_id,
                },
                reason="navigation_route_from_new_waypoint_lookup",
            )

        route_preference_defaulted = navigation.route_preference is None
        if route_preference_defaulted:
            navigation.route_preference = "fastest"

        route_to_new_waypoint = _select_requested_route(
            navigation.routes_to_new_waypoint,
            navigation.route_preference,
            navigation.selected_route_index,
        )
        if (
            route_to_new_waypoint is None
            and len(navigation.routes_to_new_waypoint) == 1
        ):
            route_to_new_waypoint = navigation.routes_to_new_waypoint[0]

        route_from_new_waypoint = _select_requested_route(
            navigation.routes_from_new_waypoint,
            navigation.route_preference,
            navigation.selected_route_index,
        )
        if (
            route_from_new_waypoint is None
            and len(navigation.routes_from_new_waypoint) == 1
        ):
            route_from_new_waypoint = navigation.routes_from_new_waypoint[0]

        if route_to_new_waypoint is None or route_from_new_waypoint is None:
            navigation.route_choice_requested = True
            routes = (
                navigation.routes_to_new_waypoint or navigation.routes_from_new_waypoint
            )
            return NextAction.respond(
                _format_route_choice_prompt(routes),
                reason="navigation_route_disambiguation",
            )

        route_to_new_waypoint_id = route_to_new_waypoint.get("route_id")
        route_from_new_waypoint_id = route_from_new_waypoint.get("route_id")
        if not isinstance(route_to_new_waypoint_id, str) or not isinstance(
            route_from_new_waypoint_id, str
        ):
            return NextAction.respond(
                "I can't safely edit that route because a replacement route ID is missing.",
                reason="navigation_missing_route_id",
            )

        if not tool_index.has("navigation_replace_one_waypoint"):
            return NextAction.respond(
                "I can't replace that waypoint because that navigation edit control is unavailable right now.",
                reason="navigation_missing_replace_waypoint_tool",
            )
        navigation.completion_message = _navigation_multi_route_completion_message(
            "the waypoint is replaced in the navigation route",
            [route_to_new_waypoint, route_from_new_waypoint],
            None if route_preference_defaulted else navigation.route_preference,
            [
                *navigation.routes_to_new_waypoint,
                *navigation.routes_from_new_waypoint,
            ],
        )
        return NextAction.tool_call(
            "navigation_replace_one_waypoint",
            {
                "waypoint_id_to_replace": navigation.waypoint_id,
                "new_waypoint_id": navigation.new_waypoint_id,
                "route_id_leading_to_new_waypoint": route_to_new_waypoint_id,
                "route_id_leading_away_from_new_waypoint": route_from_new_waypoint_id,
            },
            reason="navigation_replace_waypoint",
        )


def _extract_json_after_label(text: str, label: str) -> dict[str, Any] | None:
    label_index = text.rfind(label)
    if label_index == -1:
        return None
    start = text.find("{", label_index)
    if start == -1:
        return None
    raw = _balanced_json_object(text, start)
    if raw is None:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return None
    return parsed if isinstance(parsed, dict) else None


def _balanced_json_object(text: str, start: int) -> str | None:
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")


def _extract_business_email_extra_recipients(text: str) -> list[str]:
    lowered = text.lower()
    recipients: list[str] = []
    for match in re.finditer(r"secretary", lowered):
        segment = text[max(0, match.start() - 240) : match.end() + 240]
        if "business" not in segment.lower() and "email" not in segment.lower():
            continue
        for email in EMAIL_RE.findall(segment):
            if email not in recipients:
                recipients.append(email)
    return recipients


def _extract_quoted_meeting_topic(text: str) -> str | None:
    if "meeting" not in text.lower():
        return None
    match = re.search(r"['\"]([^'\"]+)['\"]", text)
    if match:
        return match.group(1).strip()
    return None


def _is_meeting_delay_email_request(text: str) -> bool:
    if "email" not in text or "meeting" not in text:
        return False
    return _is_late_email_request(text)


def _is_late_email_request(text: str) -> bool:
    if "email" not in text:
        return False
    if not any(word in text for word in ("late", "delay", "apologize", "apologise")):
        return False
    return bool(re.search(r"\b(send|write|email)\b", text))


def _is_meeting_attendees_email_request(text: str) -> bool:
    if "email" not in text or "meeting" not in text:
        return False
    if not re.search(r"\b(send|write|email)\b", text):
        return False
    return any(
        word in text
        for word in (
            "attendee",
            "attendees",
            "participant",
            "participants",
            "everyone",
            "all",
        )
    )


def _is_email_composition_request(text: str) -> bool:
    if "email" not in text:
        return False
    return bool(re.search(r"\b(send|write|draft|compose|message)\b", text))


def _new_meeting_attendees_email_flow(
    text: str, state: ControllerState
) -> EmailFlow:
    weather_requested = _is_weather_meeting_attendees_email_request(text)
    weather_hour, weather_minute = _extract_email_weather_time(text)
    return EmailFlow(
        active=True,
        mode="meeting_attendees",
        meeting_topic=_extract_meeting_topic_for_attendees_email(text),
        calendar_checked=bool(state.recent_calendar_meetings),
        content_message=_extract_meeting_attendees_email_content(text, state),
        weather_requested=weather_requested,
        weather_location_name=_extract_email_weather_location(text),
        weather_hour=weather_hour,
        weather_minute=weather_minute,
    )


def _extract_meeting_topic_for_attendees_email(text: str) -> str | None:
    quoted = _extract_quoted_meeting_topic(text)
    if quoted:
        return quoted

    patterns = (
        r"\b(?:attendees|participants)\s+of\s+(?:my\s+|the\s+)?(.+?)\s+meeting\b",
        r"\b(?:my\s+|the\s+)?(.+?)\s+meeting\s+(?:attendees|participants)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        topic = _clean_meeting_topic(match.group(1))
        if topic:
            return topic
    return None


def _clean_meeting_topic(value: str) -> str | None:
    value = re.sub(
        r"^\s*(?:at\s+)?\d{1,2}:\d{2}\s*(?:am|pm)?\s+",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.split(
        r"\b(?:today|tomorrow|please|with|about|at|in|on|for|and|to|from)\b",
        value,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    value = " ".join(value.strip(" .?!;:-'\"").split())
    if not value:
        return None
    if value.casefold() in {"my", "the", "a", "an"}:
        return None
    return value


def _extract_meeting_attendees_email_content(
    text: str, state: ControllerState
) -> str | None:
    lowered = text.lower()
    if not any(
        marker in lowered
        for marker in ("delay", "late", "charging stop", "detour", "charging station")
    ):
        return None

    station_name, detour_minutes = _recent_charging_stop_email_context(text, state)
    if station_name is None and "charging" not in lowered:
        return None

    topic = _extract_meeting_topic_for_attendees_email(text) or "the meeting"
    stop_text = (
        f"a charging stop at {station_name}"
        if station_name
        else "a charging stop"
    )
    delay_text = (
        f"which will add about {detour_minutes} minutes to my journey due to the detour"
        if detour_minutes is not None
        else "which may add a slight delay due to the detour"
    )
    return (
        "Dear team,\n\n"
        f"I wanted to let you know that I need to make {stop_text}, {delay_text}. "
        f"I may be slightly delayed for the {topic} meeting.\n\n"
        "Best regards"
    )


def _is_weather_meeting_attendees_email_request(text: str) -> bool:
    lowered = text.lower()
    return (
        "weather" in lowered
        and "email" in lowered
        and "meeting" in lowered
        and any(word in lowered for word in ("attendees", "participants", "team"))
    )


def _update_email_weather_request_from_text(email: EmailFlow, text: str) -> None:
    if not _is_weather_meeting_attendees_email_request(text):
        return
    email.weather_requested = True
    location = _extract_email_weather_location(text)
    if location is not None:
        email.weather_location_name = location
    hour, minute = _extract_email_weather_time(text)
    if hour is not None:
        email.weather_hour = hour
        email.weather_minute = minute


def _extract_email_weather_location(text: str) -> str | None:
    patterns = (
        r"\bweather\s+(?:for|in|at)\s+([A-Z][A-Za-z .'-]+?)(?:\s+at\b|\s+today\b|\s+tomorrow\b|[,.?!;]|$)",
        r"\bconditions\s+(?:for|in|at)\s+([A-Z][A-Za-z .'-]+?)(?:\s+at\b|\s+today\b|\s+tomorrow\b|[,.?!;]|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            location = _clean_location_query(match.group(1))
            if location:
                return location
    return None


def _extract_email_weather_time(text: str) -> tuple[int | None, int | None]:
    match = re.search(r"\bat\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", text, re.I)
    if not match:
        return None, None
    hour = int(match.group(1))
    minute = int(match.group(2) or "0")
    suffix = match.group(3)
    if suffix:
        suffix = suffix.lower()
        if suffix == "pm" and hour != 12:
            hour += 12
        elif suffix == "am" and hour == 12:
            hour = 0
    return hour, minute


def _recent_charging_stop_email_context(
    text: str, state: ControllerState
) -> tuple[str | None, int | None]:
    if not state.recent_charging_pois:
        return None, None
    selected = _select_poi_from_text(state.recent_charging_pois, text)
    if selected is None:
        selected = state.recent_charging_pois[0]
    name = selected.get("name")
    station_name = name if isinstance(name, str) and name else None
    return station_name, _extract_poi_detour_minutes(selected)


def _extract_poi_detour_minutes(poi: dict[str, Any]) -> int | None:
    raw_time = poi.get("detour_from_route_time")
    if isinstance(raw_time, dict):
        hours = _safe_int(raw_time.get("hour"), 0) or 0
        minutes = _safe_int(raw_time.get("minutes"), 0) or 0
        total = hours * 60 + minutes
        if total > 0:
            return total
    return None


def _is_contact_share_email_start(text: str) -> bool:
    if "email" not in text:
        return False
    if not any(phrase in text for phrase in ("contact information", "contact info")):
        return False
    return bool(
        re.search(r"\blook up\b", text)
        or re.search(r"\bfind\b", text)
        or re.search(r"\bsend\b", text)
    )


def _is_contact_lookup_or_call_request(text: str) -> bool:
    if any(
        phrase in text
        for phrase in (
            "contact info",
            "contact information",
            "contact details",
            "phone number",
            "call contact",
        )
    ):
        return True
    if re.search(r"\b(?:call|phone)\s+[a-z][a-z'-]*(?:\s+[a-z][a-z'-]*)?\b", text):
        return True
    return bool(
        re.fullmatch(
            r"\s*(?:find|look up|lookup|search for)\s+"
            r"[a-z][a-z'-]*(?:\s+[a-z][a-z'-]*)?\.?\s*",
            text,
        )
    )


def _extract_email_recipient_name(text: str) -> str | None:
    location = _location_name_pattern()
    patterns = (
        rf"\b(?i:send|write|email)\s+(?:an?\s+)?(?i:email)?\s*(?:to\s+)?{location}",
        rf"\b(?i:email)\s+{location}",
        rf"\b(?i:look up|find)\s+{location}(?:'s)?\s+(?i:contact)",
    )
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            value = matches[-1]
            if isinstance(value, tuple):
                value = value[-1]
            name = _clean_person_name(value)
            if name:
                return name
    return None


def _extract_contact_share_subject(text: str) -> str | None:
    location = _location_name_pattern()
    patterns = (
        rf"\b(?i:share|include|send)\s+{location}(?:'s)?\s+(?i:contact)",
        rf"\b{location}(?:'s)?\s+(?i:contact details|contact information|contact info)",
    )
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            value = matches[-1]
            if isinstance(value, tuple):
                value = value[-1]
            name = _clean_person_name(value)
            if name:
                return name
    return None


def _clean_person_name(value: str) -> str | None:
    value = re.split(
        r"\b(?:for|with|about|because|so|and|to|from|please|i|me)\b",
        value,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    value = " ".join(value.strip(" .?!;:-'\"").split())
    if not value or value.lower() in {"an", "a", "the", "her", "him", "them"}:
        return None
    return value


def _set_email_flow_name(
    email: EmailFlow, role: Literal["recipient", "subject"], name: str
) -> None:
    parts = name.strip().split()
    first_name = parts[0] if parts else None
    last_name = " ".join(parts[1:]) if len(parts) > 1 else None
    if role == "recipient":
        email.recipient_name = name
        email.recipient_first_name = first_name
        email.recipient_last_name = last_name
    else:
        email.subject_name = name
        email.subject_first_name = first_name
        email.subject_last_name = last_name


def _select_contact_match_from_text(matches: dict[str, str], text: str) -> str | None:
    lowered = text.casefold()
    exact: list[str] = []
    partial: list[str] = []
    for contact_id, name in matches.items():
        normalized = name.casefold()
        if normalized in lowered:
            exact.append(contact_id)
        elif all(part in lowered for part in normalized.split()):
            partial.append(contact_id)
    if len(exact) == 1:
        return exact[0]
    if len(partial) == 1:
        return partial[0]
    return None


def _record_email_meeting_status(state: ControllerState) -> None:
    email = state.email
    meetings = state.recent_calendar_meetings
    if not meetings:
        email.meeting_started = False
        return

    selected = _select_meeting(meetings, email.meeting_topic)
    if selected is None:
        email.meeting_started = False
        return

    topic = selected.get("topic")
    if isinstance(topic, str):
        email.meeting_topic = topic
        state.recent_meeting_topic = topic

    start = selected.get("start")
    if isinstance(start, dict):
        email.meeting_start_hour = _safe_int(start.get("hour"))
        email.meeting_start_minute = _safe_int(start.get("minute"), 0)

    email.meeting_started = _meeting_has_started(
        state.runtime, email.meeting_start_hour, email.meeting_start_minute
    )


def _select_meeting(
    meetings: list[dict[str, Any]], topic: str | None
) -> dict[str, Any] | None:
    if topic:
        normalized_topic = topic.casefold()
        for meeting in meetings:
            meeting_topic = meeting.get("topic")
            if (
                isinstance(meeting_topic, str)
                and meeting_topic.casefold() == normalized_topic
            ):
                return meeting
    return meetings[0] if len(meetings) == 1 else None


def _meeting_has_started(
    runtime: RuntimeContext, start_hour: int | None, start_minute: int | None
) -> bool | None:
    if runtime.hour is None or start_hour is None:
        return None
    current_minutes = runtime.hour * 60 + (runtime.minute or 0)
    start_minutes = start_hour * 60 + (start_minute or 0)
    return current_minutes >= start_minutes


def _record_email_contact_matches(email: EmailFlow, result: Any) -> None:
    if not isinstance(result, dict):
        email.failure_message = "I couldn't look up that contact."
        return

    raw_matches = result.get("matches")
    if not isinstance(raw_matches, dict) or not raw_matches:
        email.failure_message = "I couldn't find that contact."
        return

    matches = {
        str(contact_id): str(name)
        for contact_id, name in raw_matches.items()
        if isinstance(contact_id, str)
    }
    role = email.pending_lookup_role
    if role is None:
        return

    target_name = email.recipient_name if role == "recipient" else email.subject_name
    selected = _select_contact_match_by_target(matches, target_name)
    if selected is None and len(matches) == 1:
        selected = next(iter(matches))

    if selected is None:
        email.pending_contact_matches = matches
        return

    if role == "recipient":
        email.recipient_contact_id = selected
        email.recipient_name = matches[selected]
        _set_email_flow_name(email, "recipient", email.recipient_name)
    else:
        email.subject_contact_id = selected
        email.subject_name = matches[selected]
        _set_email_flow_name(email, "subject", email.subject_name)
    email.pending_lookup_role = None
    email.pending_contact_matches = {}


def _select_contact_match_by_target(
    matches: dict[str, str], target_name: str | None
) -> str | None:
    if not target_name:
        return None
    normalized_target = target_name.casefold()
    for contact_id, name in matches.items():
        if name.casefold() == normalized_target:
            return contact_id
    target_parts = normalized_target.split()
    candidates = []
    for contact_id, name in matches.items():
        normalized_name = name.casefold()
        if all(part in normalized_name.split() for part in target_parts):
            candidates.append(contact_id)
    return candidates[0] if len(candidates) == 1 else None


def _record_email_contact_information(email: EmailFlow, result: Any) -> None:
    if not isinstance(result, dict):
        return
    for contact_id, info in result.items():
        if not isinstance(contact_id, str) or not isinstance(info, dict):
            continue
        if contact_id == email.recipient_contact_id:
            email.recipient_email = _extract_email_from_contact_info(info)
            email.recipient_name = _contact_info_name(info) or email.recipient_name
            if email.recipient_name:
                _set_email_flow_name(email, "recipient", email.recipient_name)
        if contact_id == email.subject_contact_id:
            email.subject_email = _extract_email_from_contact_info(info)
            email.subject_phone = _extract_phone_from_contact_info(info)
            email.subject_name = _contact_info_name(info) or email.subject_name
            if email.subject_name:
                _set_email_flow_name(email, "subject", email.subject_name)
        if contact_id in email.attendee_contact_ids:
            attendee_email = _extract_email_from_contact_info(info)
            if attendee_email:
                email.attendee_emails_by_id[contact_id] = attendee_email


def _extract_email_from_contact_info(info: dict[str, Any]) -> str | None:
    email = info.get("email")
    return email if isinstance(email, str) and email else None


def _extract_phone_from_contact_info(info: dict[str, Any]) -> str | None:
    phone = info.get("phone_number")
    return phone if isinstance(phone, str) and phone else None


def _contact_info_name(info: dict[str, Any]) -> str | None:
    raw_name = info.get("name")
    if isinstance(raw_name, dict):
        first = raw_name.get("first_name")
        last = raw_name.get("last_name")
        parts = [part for part in (first, last) if isinstance(part, str) and part]
        return " ".join(parts) if parts else None
    if isinstance(raw_name, str):
        return raw_name
    return None


def _email_missing_contact_info_ids(email: EmailFlow) -> list[str]:
    ids: list[str] = []
    if email.recipient_contact_id and email.recipient_email is None:
        ids.append(email.recipient_contact_id)
    if (
        email.mode == "share_contact"
        and email.subject_contact_id
        and (email.subject_email is None or email.subject_phone is None)
    ):
        ids.append(email.subject_contact_id)
    if email.mode == "meeting_attendees":
        ids.extend(
            contact_id
            for contact_id in email.attendee_contact_ids
            if contact_id not in email.attendee_emails_by_id
        )
    return ids


def _calendar_arguments(runtime: RuntimeContext) -> dict[str, Any] | None:
    if runtime.month is None or runtime.day is None:
        return None
    return {"month": runtime.month, "day": runtime.day}


def _build_email_content(state: ControllerState) -> str | None:
    email = state.email
    if email.mode == "meeting_delay":
        recipient_first = email.recipient_first_name or _first_name(
            email.recipient_name
        )
        topic = email.meeting_topic or "our meeting"
        started = _format_clock_time(
            email.meeting_start_hour, email.meeting_start_minute
        )
        delay = _meeting_delay_minutes(state.runtime, email)
        delay_text = (
            f"I'm running about {delay} minutes late"
            if delay is not None and delay > 0
            else "I'm running late"
        )
        return (
            f"Hi {recipient_first}, I wanted to reach out regarding our {topic} "
            f"meeting that started at {started} today. {delay_text} and apologize "
            "for the delay. I should be there shortly. Thank you for your patience. "
            "Best regards"
        )

    if email.mode == "share_contact":
        recipient_first = email.recipient_first_name or _first_name(
            email.recipient_name
        )
        subject_name = email.subject_name
        if not subject_name or not email.subject_phone or not email.subject_email:
            return None
        return (
            f"Hi {recipient_first},\n\n"
            f"I wanted to share {subject_name}'s contact information with you:\n\n"
            f"Name: {subject_name}\n"
            f"Phone: {email.subject_phone}\n"
            f"Email: {email.subject_email}\n\n"
            "Best regards"
        )

    if email.mode == "meeting_attendees" and email.weather_requested:
        if not email.weather_checked:
            return None
        return _build_weather_meeting_email_content(email)

    return None


def _build_weather_meeting_email_content(email: EmailFlow) -> str:
    topic = email.meeting_topic or "the meeting"
    location = email.weather_location_name or email.meeting_location_name or "the meeting location"
    time_text = _format_clock_time(email.weather_hour, email.weather_minute)
    condition_text = _format_weather_condition(email.weather_condition)
    temperature_text = ""
    if email.weather_temperature is not None:
        temperature_text = f" The temperature is expected to be around {email.weather_temperature:g} C."

    impact = _weather_travel_impact(email.weather_condition)
    return (
        "Hi team,\n\n"
        f"I wanted to share a weather update for our {topic} meeting in {location} "
        f"today at {time_text}. The forecast shows {condition_text}."
        f"{temperature_text}\n\n"
        f"{impact}\n\n"
        "Best regards"
    )


def _format_weather_condition(condition: str | None) -> str:
    if condition is None:
        return "the available weather conditions"
    return condition.replace("_", " ").strip().lower()


def _weather_travel_impact(condition: str | None) -> str:
    normalized = (condition or "").casefold()
    if any(term in normalized for term in ("rain", "snow", "hail", "storm")):
        return (
            "Please plan accordingly for travel, allow extra time, and bring suitable "
            "weather protection if needed."
        )
    if "fog" in normalized:
        return (
            "Please allow extra travel time because visibility may be reduced on the "
            "way to the meeting."
        )
    if "cloud" in normalized:
        return (
            "Please keep the cloudy conditions in mind while planning your travel to "
            "the meeting."
        )
    return "Please keep the forecast in mind while planning your travel to the meeting."


def _format_email_confirmation(email: EmailFlow, extra_recipients: list[str]) -> str:
    addresses = _email_recipient_addresses(email, extra_recipients)
    address_list = ", ".join(addresses)
    return (
        "Please confirm these email details before I send it.\n"
        f"Recipients: {address_list}\n"
        f"Content: {email.content_message}\n\n"
        "Please confirm if I should send it."
    )


def _email_recipient_addresses(
    email: EmailFlow, extra_recipients: list[str]
) -> list[str]:
    addresses = []
    if email.mode == "meeting_attendees":
        addresses.extend(_email_attendee_addresses(email))
    if email.recipient_email:
        addresses.append(email.recipient_email)
    if email.mode == "meeting_delay":
        for extra in extra_recipients:
            if extra not in addresses:
                addresses.append(extra)
    return addresses


def _email_attendee_addresses(email: EmailFlow) -> list[str]:
    addresses: list[str] = []
    for contact_id in email.attendee_contact_ids:
        address = email.attendee_emails_by_id.get(contact_id)
        if address and address not in addresses:
            addresses.append(address)
    return addresses


def _record_email_attendee_meeting(
    state: ControllerState, meeting: dict[str, Any]
) -> None:
    email = state.email
    topic = meeting.get("topic")
    if isinstance(topic, str):
        email.meeting_topic = topic
        state.recent_meeting_topic = topic

    start = meeting.get("start")
    if isinstance(start, dict):
        email.meeting_start_hour = _safe_int(start.get("hour"))
        email.meeting_start_minute = _safe_int(start.get("minute"), 0)
        if email.weather_requested and email.weather_hour is None:
            email.weather_hour = email.meeting_start_hour
            email.weather_minute = email.meeting_start_minute

    location = meeting.get("location")
    if isinstance(location, str) and location.strip():
        email.meeting_location_name = location.strip()
        if email.weather_requested:
            email.weather_location_name = location.strip()

    attendees = meeting.get("attendees")
    if isinstance(attendees, list):
        email.attendee_contact_ids = [
            attendee for attendee in attendees if isinstance(attendee, str)
        ]


def _select_meeting_from_text(
    meetings: list[dict[str, Any]], text: str
) -> dict[str, Any] | None:
    normalized_text = _normalized_text(text)
    candidates: list[dict[str, Any]] = []
    for meeting in meetings:
        topic = meeting.get("topic")
        if not isinstance(topic, str):
            continue
        normalized_topic = _normalized_text(topic)
        if normalized_topic and normalized_topic in normalized_text:
            candidates.append(meeting)
    return candidates[0] if len(candidates) == 1 else None


def _format_meeting_choice_prompt(meetings: list[dict[str, Any]]) -> str:
    options = []
    for index, meeting in enumerate(meetings[:4], start=1):
        topic = meeting.get("topic")
        if not isinstance(topic, str) or not topic.strip():
            topic = f"Meeting {index}"
        start = meeting.get("start")
        time_text = ""
        if isinstance(start, dict):
            hour = _safe_int(start.get("hour"))
            minute = _safe_int(start.get("minute"), 0)
            if hour is not None:
                time_text = f" at {_format_clock_time(hour, minute)}"
        options.append(f"{index}. {topic}{time_text}")
    if not options:
        return "Which meeting should I use?"
    return f"Which meeting should I use? {'; '.join(options)}"


def _normalize_email_content(text: str) -> str:
    content = text.strip()
    content = re.sub(
        r"^(?:it should be|say|send|write|please send|please write)\s+",
        "",
        content,
        flags=re.IGNORECASE,
    )
    content = content.strip(" .")

    def replace_time(match: re.Match[str]) -> str:
        hour = int(match.group(1))
        minute = int(match.group(2) or "00")
        suffix = match.group(3).lower()
        if suffix == "pm" and hour != 12:
            hour += 12
        if suffix == "am" and hour == 12:
            hour = 0
        return f"{hour:02d}:{minute:02d}"

    content = re.sub(
        r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b",
        replace_time,
        content,
        flags=re.IGNORECASE,
    )
    if content:
        return content[0].upper() + content[1:]
    return text.strip()


def _format_contact_choice_prompt(matches: dict[str, str]) -> str:
    names = ", ".join(name.title() for name in matches.values())
    return f"I found multiple matching contacts: {names}. Which one should I use?"


def _first_name(name: str | None) -> str:
    if not name:
        return "there"
    return name.split()[0]


def _format_clock_time(hour: int | None, minute: int | None) -> str:
    if hour is None:
        return "the scheduled time"
    minute = minute or 0
    return f"{hour:02d}:{minute:02d}"


def _meeting_delay_minutes(runtime: RuntimeContext, email: EmailFlow) -> int | None:
    if runtime.hour is None or email.meeting_start_hour is None:
        return None
    current = runtime.hour * 60 + (runtime.minute or 0)
    start = email.meeting_start_hour * 60 + (email.meeting_start_minute or 0)
    return max(0, current - start)


def _contains_new_action_intent(text: str) -> bool:
    stripped = text.strip().lower()
    if stripped in {"yes", "yeah", "yep", "sure", "ok", "okay", "no", "nope"}:
        return False

    return bool(
        re.search(
            r"\b(open|close|shut|turn|switch|set|change|replace|update|adjust|"
            r"activate|deactivate|navigate|drive|route|find|search|look for|"
            r"delete|remove|cancel|send|email|match|check|call)\b",
            stripped,
        )
        or "battery" in stripped
        or "range" in stripped
        or "charging station" in stripped
        or "what about" in stripped
    )


def _is_sunroof_open_request(text: str) -> bool:
    if "sunroof" not in text:
        return False
    if any(word in text for word in ("close", "shut")):
        return False
    return any(word in text for word in ("open", "fresh air", "vent"))


def _is_sunshade_request(text: str) -> bool:
    if "sunshade" not in text or "sunroof" in text:
        return False
    return any(
        phrase in text
        for phrase in (
            "open",
            "close",
            "shut",
            "adjust",
            "set",
            "help",
            "too bright",
            "sun is",
            "sun's",
        )
    )


def _is_window_open_request(text: str) -> bool:
    if "sunroof" in text:
        return False
    if not re.search(r"\bwindows?\b", text):
        return False
    if _is_close_request(text):
        return False
    return any(
        phrase in text
        for phrase in (
            "open",
            "fresh air",
            "ventilation",
            "ventilate",
            "stuffy",
            "air out",
            "crack",
            "roll down",
        )
    )


def _is_window_match_request(text: str) -> bool:
    if not re.search(r"\bwindows?\b", text):
        return False
    if "rear" not in text and "front" not in text:
        return False
    return any(
        phrase in text
        for phrase in (
            "match",
            "same position",
            "same level",
            "same amount",
            "equal",
            "even",
        )
    )


def _extract_window_match_reference(
    text: str,
) -> Literal["FRONT", "PASSENGER_REAR"]:
    if "passenger rear" in text or "rear passenger" in text or "right rear" in text:
        return "PASSENGER_REAR"
    return "FRONT"


def _window_match_completion_message(match: WindowMatchFlow) -> str:
    if match.reference_window == "PASSENGER_REAR":
        return "Done, all windows match the passenger rear window."
    return "Done, the rear windows match the front windows."


def _is_ambient_light_request(text: str) -> bool:
    if re.search(r"\b(?:not|no)\s+(?:the\s+)?ambient lights?\b", text):
        return False
    return bool(
        ("ambient" in text and ("light" in text or "lighting" in text))
        or "cabin light" in text
        or "mood light" in text
    )


def _is_ambient_light_match_car_color_request(text: str) -> bool:
    if not _is_ambient_light_request(text):
        return False
    if "match" not in text:
        return False
    return any(
        phrase in text
        for phrase in (
            "car color",
            "car's color",
            "color of my car",
            "car colour",
            "car's colour",
            "colour of my car",
            "exterior color",
            "exterior colour",
            "outside color",
            "outside colour",
            "exterior",
            "paint color",
            "paint colour",
        )
    )


def _is_light_off_request(text: str) -> bool:
    return (
        _is_close_request(text)
        or "turn off" in text
        or "switch off" in text
        or "deactivate" in text
    )


def _is_high_beam_request(text: str) -> bool:
    if re.search(r"\blow[- ]beams?\b", text):
        return False
    if not (re.search(r"\bhigh[- ]beams?\b", text) or re.search(r"\bbeams?\b", text)):
        return False
    return any(
        phrase in text
        for phrase in (
            "turn on",
            "turn off",
            "switch on",
            "switch off",
            "activate",
            "deactivate",
            "set",
            "need",
        )
    )


def _is_fog_light_request(text: str) -> bool:
    if not re.search(r"\bfog[- ]lights?\b", text):
        return False
    return any(
        phrase in text
        for phrase in (
            "turn on",
            "turn off",
            "switch on",
            "switch off",
            "activate",
            "deactivate",
            "set",
            "need",
            "want",
        )
    )


def _is_fan_speed_request(text: str) -> bool:
    if not re.search(r"\bfan\b", text):
        return False
    if "airflow direction" in text or "air flow direction" in text:
        return False
    if ("air conditioning" in text or re.search(r"\bac\b", text)) and re.search(
        r"\bwindows?\b", text
    ):
        return False
    return any(
        phrase in text
        for phrase in (
            "turn on",
            "turn off",
            "switch on",
            "switch off",
            "set",
            "increase",
            "decrease",
            "level",
            "speed",
        )
    )


def _is_current_fan_speed_query(text: str) -> bool:
    if not re.search(r"\bfan\b", text) or "speed" not in text:
        return False
    if _extract_level(text) is not None or _is_light_off_request(text):
        return False
    return bool(
        "current" in text
        or "look up" in text
        or "check" in text
        or "what about" in text
        or re.search(r"\bwhat(?:'s| is)?\b", text)
    )


def _is_steering_wheel_heating_request(text: str) -> bool:
    if "steering wheel" not in text:
        return False
    return any(
        phrase in text
        for phrase in (
            "heating",
            "heated",
            "heat",
            "warm",
            "cold",
            "chilly",
            "turn on",
            "turn off",
            "switch on",
            "switch off",
            "set",
        )
    )


def _is_reading_light_request(text: str) -> bool:
    if "reading light" not in text and "reading lights" not in text:
        return False
    return any(
        phrase in text
        for phrase in (
            "turn on",
            "turn off",
            "switch on",
            "switch off",
            "activate",
            "deactivate",
            "set",
            "need",
        )
    )


def _is_reading_light_by_occupancy_request(text: str) -> bool:
    if (
        "reading light" not in text
        and "reading lights" not in text
        and "lights" not in text
    ):
        return False
    return any(
        phrase in text
        for phrase in (
            "occupied seats",
            "unoccupied seats",
            "empty seats",
            "who's actually in the car",
            "who is actually in the car",
            "based on who",
            "waste energy",
        )
    )


def _is_defrost_request(text: str) -> bool:
    if "defrost" not in text:
        return False
    if any(phrase in text for phrase in ("match", "same as", "as much as")):
        return False
    if "passenger rear" in text or "driver rear" in text:
        return False
    return any(
        phrase in text
        for phrase in (
            "turn on",
            "turn off",
            "switch on",
            "switch off",
            "activate",
            "deactivate",
            "set",
            "clear",
            "fog",
            "fogging",
        )
    )


def _is_trunk_request(text: str) -> bool:
    if "trunk" not in text and "boot" not in text:
        return False
    return any(word in text for word in ("open", "close", "shut", "access"))


def _is_air_circulation_request(text: str) -> bool:
    if "air circulation" not in text and "recirculation" not in text:
        return False
    return any(
        phrase in text
        for phrase in (
            "change",
            "set",
            "switch",
            "turn",
            "what about",
            "fresh air",
            "recirculation",
            "preferred",
            "don't like",
            "do not like",
        )
    )


def _is_air_conditioning_request(text: str) -> bool:
    mentions_ac = "air conditioning" in text or re.search(r"\bac\b", text)
    if not mentions_ac:
        return bool(
            re.search(r"\b(?:cool down|cool off)\b", text)
            and any(word in text for word in ("warm", "hot", "heat", "heating"))
            and any(word in text for word in ("car", "cabin", "interior", "inside"))
        )
    return any(
        phrase in text
        for phrase in (
            "turn on",
            "turn off",
            "switch on",
            "switch off",
            "activate",
            "deactivate",
            "set",
            "cool",
            "what about",
        )
    )


def _is_air_conditioning_off_request(text: str) -> bool:
    return bool(
        "deactivate" in text
        or re.search(r"\bturn off (?:the )?(?:air conditioning|ac)\b", text)
        or re.search(r"\bswitch off (?:the )?(?:air conditioning|ac)\b", text)
        or re.search(r"\b(?:air conditioning|ac) off\b", text)
        or re.search(r"\bturn (?:the )?(?:air conditioning|ac) off\b", text)
        or re.search(r"\bswitch (?:the )?(?:air conditioning|ac) off\b", text)
    )


def _mentions_temperature_setting(text: str) -> bool:
    return bool(
        "temperature" in text
        or "temperatures" in text
        or re.search(r"\b\d+(?:\.\d+)?\s*(?:degree|degrees|celsius|c)\b", text)
    )


def _extract_temperature_setting(text: str) -> float | None:
    final_match = re.search(
        r"\b(?:would be|should be|to|at)\s*([1-3]?\d(?:\.\d+)?)\s*"
        r"(?:degree|degrees|celsius|c)\b",
        text,
    )
    if final_match:
        return float(final_match.group(1))

    degree_match = _first_absolute_temperature_match(text)
    if degree_match is not None:
        return float(degree_match.group(1))

    if _extract_temperature_delta(text) is not None:
        return None

    temperature_match = re.search(
        r"\b(?:temperature|temp)[^.?!,;]*?\b([1-3]?\d(?:\.\d+)?)\b",
        text,
    )
    if temperature_match:
        return float(temperature_match.group(1))

    return None


def _first_absolute_temperature_match(text: str) -> re.Match[str] | None:
    for degree_match in re.finditer(
        r"\b([1-3]?\d(?:\.\d+)?)\s*(?:degree|degrees|celsius|c)\b",
        text,
    ):
        prefix = text[max(0, degree_match.start() - 24) : degree_match.start()]
        if re.search(
            r"\b(?:by|raise|raised|increase|increased|lower|lowered|decrease|"
            r"decreased|up|down|current|currently|already|now|is)\s*$",
            prefix,
        ):
            continue
        return degree_match
    return None


def _extract_temperature_delta(text: str) -> float | None:
    increase_match = re.search(
        r"\b(?:raise|increase|turn up|bump up|warm up)[^.?!;]*?\bby\s+"
        r"([1-3]?\d(?:\.\d+)?)\s*(?:degree|degrees|celsius|c)\b",
        text,
    )
    if increase_match:
        return float(increase_match.group(1))

    decrease_match = re.search(
        r"\b(?:lower|decrease|turn down|cool down)[^.?!;]*?\bby\s+"
        r"([1-3]?\d(?:\.\d+)?)\s*(?:degree|degrees|celsius|c)\b",
        text,
    )
    if decrease_match:
        return -float(decrease_match.group(1))
    return None


def _resolve_temperature_setting(
    text: str,
    state: ControllerState,
    seat_zone: Literal["ALL_ZONES", "DRIVER", "PASSENGER"] = "ALL_ZONES",
) -> float | None:
    target = _extract_temperature_setting(text)
    if target is not None:
        return target

    delta = _extract_temperature_delta(text)
    if delta is None:
        return None
    base = _recent_temperature_for_zone(state, seat_zone)
    if base is None:
        return None
    return base + delta


def _recent_temperature_for_zone(
    state: ControllerState,
    seat_zone: Literal["ALL_ZONES", "DRIVER", "PASSENGER"],
) -> float | None:
    if seat_zone == "DRIVER":
        return state.recent_driver_temperature
    if seat_zone == "PASSENGER":
        return state.recent_passenger_temperature
    temperatures = [
        temperature
        for temperature in (
            state.recent_driver_temperature,
            state.recent_passenger_temperature,
        )
        if temperature is not None
    ]
    if not temperatures:
        return None
    return sum(temperatures) / len(temperatures)


def _is_cabin_temperature_request(text: str) -> bool:
    if _extract_temperature_setting(text) is None:
        return False
    if _is_climate_inspection_request(text):
        return False
    if re.search(r"\b(what|tell me|check|current)\b", text):
        return False
    return bool(
        "cabin temperature" in text
        or "climate temperature" in text
        or "all zones" in text
        or "driver temperature" in text
        or "passenger temperature" in text
        or "driver side temperature" in text
        or "passenger side temperature" in text
    )


def _extract_temperature_seat_zone(
    text: str,
) -> Literal["ALL_ZONES", "DRIVER", "PASSENGER"]:
    if "passenger" in text and "all zones" not in text:
        return "PASSENGER"
    if "driver" in text and "all zones" not in text:
        return "DRIVER"
    return "ALL_ZONES"


def _mentions_seat_heating(text: str) -> bool:
    return bool(
        "seat heating" in text
        or "heated seat" in text
        or "heated seats" in text
        or "seat heater" in text
        or "seat heaters" in text
    )


def _is_occupancy_comfort_request(text: str) -> bool:
    if not any(word in text for word in ("warm", "heat", "cold", "chilly")):
        return False
    if not any(
        phrase in text
        for phrase in (
            "occupied seat",
            "occupied seats",
            "seats that are occupied",
            "seats currently occupied",
            "currently occupied seats",
            "seat zones that are currently occupied",
            "seat zones currently occupied",
            "unoccupied seat",
            "unoccupied seats",
            "empty seat",
            "empty seats",
            "where people are sitting",
            "where people are seated",
            "where someone is sitting",
            "where someone is seated",
            "where anyone is sitting",
            "where anyone is seated",
            "people are sitting",
            "people are seated",
        )
    ):
        return False
    return any(
        phrase in text
        for phrase in (
            "warm up",
            "heat up",
            "efficient",
            "comfort",
            "seat heating",
            "heated seats",
        )
    )


def _occupancy_comfort_needs_temperature(text: str) -> bool:
    if _mentions_temperature_setting(text):
        return True
    if any(phrase in text for phrase in ("cabin temperature", "climate temperature")):
        return True
    if "usual comfortable temperature" in text or "comfort temperature" in text:
        return True
    if "comprehensively" in text or "comprehensive" in text:
        return True
    if _mentions_seat_heating(text) and not any(
        phrase in text
        for phrase in (
            "warm up the car",
            "heat up the car",
            "overall comfort",
            "comprehensive warming",
        )
    ):
        return False
    return "warm up" in text or "heat up" in text


def _requests_unoccupied_seat_heating_off(text: str) -> bool:
    return bool(
        _mentions_seat_heating(text)
        and any(
            phrase in text
            for phrase in (
                "unoccupied seat",
                "unoccupied seats",
                "empty seat",
                "empty seats",
                "seats that are not occupied",
                "not occupied",
            )
        )
        and _is_light_off_request(text)
    )


def _is_unoccupied_seat_heating_status_query(text: str) -> bool:
    if not _mentions_seat_heating(text):
        return False
    if not any(
        phrase in text
        for phrase in (
            "unoccupied seat",
            "unoccupied seats",
            "empty seat",
            "empty seats",
            "seats that are empty",
            "seats that are currently empty",
        )
    ):
        return False
    if _is_light_off_request(text):
        return False
    return bool(
        "?" in text
        or "currently being heated" in text
        or "heating on" in text
        or "is it on" in text
        or "are any" in text
        or "tell me" in text
        or "need to know" in text
    )


def _is_child_seat_heating_status_query(text: str) -> bool:
    if "child" not in text and "kid" not in text:
        return False
    if not _mentions_seat_heating(text):
        return False
    return bool(
        "activated" in text
        or re.search(r"\bon\b", text)
        or "enabled" in text
        or "working" in text
    )


def _is_passenger_comfort_match_request(text: str) -> bool:
    if _is_driver_temperature_match_passenger_request(text):
        return False
    if "passenger" not in text or "driver" not in text:
        return False
    if not _mentions_seat_heating(text):
        return False
    if not _mentions_temperature_setting(text):
        return False
    return bool(
        "match" in text
        or "same level" in text
        or "same as" in text
        or "driver side comfort" in text
    )


def _is_driver_temperature_match_passenger_request(text: str) -> bool:
    if "driver" not in text or "passenger" not in text:
        return False
    if "temperature" not in text and "temp" not in text:
        return False
    return bool(
        re.search(
            r"\bdriver(?:'s)?(?: zone)? temperature\b.*\b(?:same as|match|matching)\b.*\bpassenger",
            text,
        )
        or re.search(
            r"\b(?:same as|match|matching)\b.*\bpassenger(?:'s)?(?: zone)? temperature\b",
            text,
        )
    )


def _is_driver_comfort_temperature_request(text: str) -> bool:
    if _requests_passenger_heating_off(text) and any(
        phrase in text
        for phrase in (
            "my temperature",
            "my comfort level",
            "comfort temperature",
            "driver comfort",
            "just me",
        )
    ):
        return True
    if not (
        "driver zone temperature" in text
        or "driver temperature" in text
        or "driver's temperature" in text
    ):
        return False
    if "comfort" in text:
        return True
    if re.search(r"\bby\s+[1-3]?\d(?:\.\d+)?\s*(?:degree|degrees|celsius|c)\b", text):
        return False
    return _extract_temperature_setting(text) is not None


def _requests_passenger_heating_off(text: str) -> bool:
    return bool(
        _mentions_seat_heating(text)
        and "passenger" in text
        and _is_light_off_request(text)
    )


def _is_climate_inspection_request(text: str) -> bool:
    if not (_mentions_temperature_setting(text) or _mentions_seat_heating(text)):
        return False
    return any(
        phrase in text
        for phrase in (
            "current climate",
            "current temperature",
            "current temperatures",
            "current seat heating",
            "climate settings",
            "seat heating levels",
            "heating levels",
            "tell me what",
            "check the current",
            "what are the current",
            "what the current",
        )
    )


def _is_climate_energy_inspection_request(text: str) -> bool:
    if (
        any(
            phrase in text
            for phrase in (
                "climate settings",
                "current climate",
                "climate situation",
            )
        )
        and any(
            phrase in text
            for phrase in (
                "who's in the car",
                "who is in the car",
                "who's inside",
                "who is inside",
                "seating situation",
                "which seats are occupied",
                "seat occupancy",
            )
        )
    ):
        return True

    has_energy_goal = bool(
        re.search(r"\b(?:save|saving)\b[^.?!,;]{0,40}\benergy\b", text)
        or re.search(r"\benergy\b[^.?!,;]{0,40}\b(?:save|saving)\b", text)
    ) or any(
        phrase in text
        for phrase in (
            "energy efficiency",
            "energy efficient",
            "waste energy",
            "wasting energy",
            "inefficient",
            "inefficiencies",
            "optimize",
            "optimise",
        )
    )
    if not has_energy_goal:
        return False

    has_climate_context = any(
        phrase in text
        for phrase in (
            "climate",
            "temperature",
            "seat heating",
            "heated seats",
        )
    )
    if not has_climate_context:
        return False

    has_occupancy_context = any(
        phrase in text
        for phrase in (
            "who's in the car",
            "who is in the car",
            "who's sitting",
            "who is sitting",
            "sitting where",
            "seating situation",
            "actually in the car",
            "occupied",
            "unoccupied",
            "empty seat",
            "empty seats",
        )
    )
    if not has_occupancy_context:
        return False

    return bool(
        "current" in text
        or "check" in text
        or "tell me" in text
        or "what" in text
        or "who" in text
    )


def _is_air_quality_question(text: str) -> bool:
    if not any(word in text for word in ("stagnant", "stale", "stuffy")):
        return False
    if re.search(
        r"\b(turn on|turn off|switch on|switch off|set|activate|deactivate)\b",
        text,
    ):
        return False
    return (
        "what can" in text
        or "what should" in text
        or "current climate settings" in text
        or "climate settings" in text
        or "?" in text
    )


def _is_simple_navigation_request(text: str) -> bool:
    lowered = text.lower()
    if _is_complex_navigation_request(lowered):
        return False
    if _extract_navigation_destination(text) is None:
        return False
    return bool(
        re.search(
            r"\b(navigate|navigation|directions?|route|drive|go|travel)\b",
            lowered,
        )
        or _is_replace_destination_request(lowered)
    )


def _is_navigation_destination_change_followup(
    previous_user_text: str, text: str
) -> bool:
    previous = previous_user_text.lower()
    if not previous:
        return False
    return bool(
        _is_replace_destination_request(previous)
        and re.search(r"\b(go|drive|travel|route|navigate|destination)\b", text)
    )


def _navigation_replace_needs_current_state(text: str) -> bool:
    return any(
        marker in text
        for marker in (
            "final destination",
            "multi-stop",
            "multistop",
            "waypoint",
            "intermediate stop",
            " stops ",
            " stop ",
        )
    )


def _is_navigation_delete_waypoint_request(text: str) -> bool:
    if not re.search(r"\b(route|navigation|waypoint|stop)\b", text):
        return False
    return bool(
        re.search(r"\b(remove|delete|skip|drop)\b", text)
        or "no longer need" in text
        or "no longer necessary" in text
        or "without the intermediate stop" in text
    )


def _is_navigation_delete_destination_request(text: str) -> bool:
    if not re.search(r"\b(remove|delete|drop|cancel)\b", text) and not any(
        marker in text
        for marker in (
            "no longer need",
            "don't need",
            "do not need",
            "end my trip at",
            "end the trip at",
        )
    ):
        return False
    return bool(
        "final destination" in text
        or "last stop" in text
        or "final stop" in text
        or re.search(r"\bdelete\b[^.?!]{0,60}\bdestination\b", text)
        or re.search(r"\bcancel\b[^.?!]{0,60}\bdestination\b", text)
        or re.search(r"\bdestination\b[^.?!]{0,60}\b(delete|remove|cancel)\b", text)
    )


def _is_navigation_replace_waypoint_request(text: str) -> bool:
    if "final destination" in text or re.search(r"\bdestination\b", text):
        return False
    if not (
        re.search(r"\b(intermediate stop|waypoint|stop)\b", text)
        or (
            re.search(r"\b(route|navigation)\b", text)
            and re.search(r"\breplace\b[^.?!]{0,80}\bwith\b", text)
        )
    ):
        return False
    return bool(re.search(r"\b(replace|change|switch|swap)\b", text))


def _is_vague_navigation_destination_request(text: str) -> bool:
    if not (
        _is_replace_destination_request(text)
        or re.search(r"\b(navigate|directions?|route|drive|go|travel)\b", text)
    ):
        return False
    vague_markers = (
        "known for",
        "famous for",
        "a city",
        "which city",
        "some city",
        "somewhere",
        "automotive",
    )
    return any(marker in text for marker in vague_markers)


def _location_name_pattern() -> str:
    return r"([A-Z][A-Za-z]*(?:\s+(?:[A-Z][A-Za-z]*|la|de|del|di|and)){0,4})"


def _extract_navigation_clarified_destination(text: str) -> str | None:
    location = _location_name_pattern()
    patterns = (
        rf"\b(?i:meant|mean|choose|use|destination is|go to|to)\s+{location}",
        rf"^\s*{location}\s*[.?!]?\s*$",
    )
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            value = matches[-1]
            if isinstance(value, tuple):
                value = value[-1]
            destination = _clean_location_query(value)
            if destination:
                return destination
    return _extract_navigation_destination(text)


def _extract_navigation_new_waypoint(text: str) -> str | None:
    location = _location_name_pattern()
    patterns = (
        rf"\b(?i:with|to)\s+{location}",
        rf"\b(?i:instead|rather)\s+{location}",
    )
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            value = matches[-1]
            if isinstance(value, tuple):
                value = value[-1]
            waypoint = _clean_location_query(value)
            if waypoint:
                return waypoint
    return None


def _extract_navigation_waypoint_to_replace(text: str) -> str | None:
    location = _location_name_pattern()
    patterns = (
        rf"\b(?i:replace|swap|change|switch)\s+{location}\s+(?i:with|to)\b",
        rf"\b(?i:from)\s+{location}\s+(?i:to)\b",
    )
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            value = matches[-1]
            if isinstance(value, tuple):
                value = value[-1]
            waypoint = _clean_location_query(value)
            if waypoint and waypoint.lower() not in {"current", "intermediate"}:
                return waypoint
    return None


def _extract_navigation_waypoint_to_delete(text: str) -> str | None:
    location = _location_name_pattern()
    patterns = (
        rf"\b(?i:remove|delete|skip|drop)\s+{location}",
        rf"\b(?i:no longer need)(?: to)? (?i:stop) (?i:in|at)\s+{location}",
        rf"\b{location}\s+(?i:stop|waypoint)\s+(?:(?i:is)\s+)?(?i:no longer)",
    )
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            value = matches[-1]
            if isinstance(value, tuple):
                value = value[-1]
            waypoint = _clean_location_query(value)
            if waypoint:
                return waypoint
    return None


def _extract_navigation_destination_to_delete(text: str) -> str | None:
    location = _location_name_pattern()
    patterns = (
        rf"\b(?i:remove|delete|drop|cancel)\s+(?:my\s+|the\s+|current\s+)?(?:final\s+destination|destination|final\s+stop|last\s+stop)?\s*(?:\(\s*)?{location}",
        rf"\b(?i:no longer need|don't need|do not need)(?:\s+to\s+(?:go|travel|drive|navigate))?(?:\s+to)?\s+{location}",
        rf"\b{location}\s+(?i:is|as)?\s*(?:my\s+|the\s+)?(?i:final destination|last stop|final stop)[^.?!]{0, 80}\b(?i:remove|delete|drop|cancel|no longer)",
    )
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            value = matches[-1]
            if isinstance(value, tuple):
                value = value[-1]
            destination = _clean_location_query(value)
            if destination:
                return destination
    return None


def _new_navigation_delete_destination_flow(
    text: str, state: ControllerState
) -> NavigationFlow:
    previous_navigation = state.navigation
    destination_name = _extract_navigation_destination_to_delete(text)
    destination_id = _matching_recent_location_id(
        destination_name,
        previous_navigation.destination_name,
        previous_navigation.destination_id,
    ) or _matching_recent_location_id(
        destination_name,
        state.recent_location_lookup_name,
        state.recent_location_lookup_id,
    )
    return NavigationFlow(
        active=True,
        mode="delete_final_destination",
        destination_name=destination_name,
        destination_id=destination_id,
        needs_current_navigation=destination_id is None,
    )


def _new_navigation_delete_waypoint_flow(text: str) -> NavigationFlow:
    return NavigationFlow(
        active=True,
        mode="delete_waypoint",
        waypoint_name=_extract_navigation_waypoint_to_delete(text),
        route_preference=_extract_route_preference(text.lower()),
        needs_current_navigation=True,
    )


def _is_complex_navigation_request(text: str) -> bool:
    complex_markers = (
        "charging station",
        "charging stop",
        "charge",
        "restaurant",
        "fast food",
        "parking",
        "airport",
        "toilet",
        "supermarket",
        "bakery",
        "point of interest",
        "poi",
        "weather",
        "rain",
        "calendar",
        "multi-stop",
        "multistop",
        "waypoint",
        "along the route",
        "stop in",
        "stop at",
        "stops in",
        "stops at",
    )
    return any(marker in text for marker in complex_markers)


def _is_route_energy_request(text: str) -> bool:
    has_energy_term = any(
        term in text
        for term in (
            "battery",
            "range",
            "state of charge",
            "soc",
            "charge",
            "charging",
            "charger",
        )
    )
    if not has_energy_term:
        return False

    if (
        ("calendar" in text or "meeting" in text)
        and "charging station" not in text
        and "charger" not in text
    ):
        return False

    if (
        "charging station stop" in text
        and re.search(r"\b(set up|start)\s+navigation\b", text)
        and not any(
            term in text
            for term in (
                "battery",
                "range",
                "along",
                "around",
                "on the way",
                "fast food",
                "eat",
                "open",
                "dinner",
                "lunch",
                "between",
            )
        )
    ):
        return False

    return bool(
        "battery" in text
        or "range" in text
        or "state of charge" in text
        or "soc" in text
        or "how far" in text
        or "how long" in text
        or "charging time" in text
        or "charging stop" in text
        or "charging station" in text
        or "charger" in text
    )


def _companion_poi_category_for_charging_stop(text: str) -> str | None:
    if not (
        any(
            phrase in text
            for phrase in ("charging station", "charging stop", "charger")
        )
        and re.search(r"\b(navigation|route|navigate|set up|start)\b", text)
    ):
        return None

    explicit_categories: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("fast_food", ("fast food", "burger", "drive-through", "drive through")),
        ("restaurants", ("restaurant", "restaurants")),
        ("supermarkets", ("supermarket", "grocery", "groceries")),
        ("public_toilets", ("toilet", "toilets", "restroom", "bathroom")),
        ("bakery", ("bakery", "bakeries")),
        ("parking", ("parking", "car park")),
        ("airports", ("airport", "airports")),
    )
    for category, terms in explicit_categories:
        if any(term in text for term in terms):
            return category

    if any(term in text for term in ("eat", "meal", "dinner", "lunch")):
        return "restaurants"
    return None


def _extract_time_window_minutes(text: str) -> tuple[int, int] | None:
    match = re.search(
        r"\bbetween\s+"
        r"(\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)?)"
        r"\s+(?:and|to|-)\s+"
        r"(\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)?)",
        text,
    )
    if not match:
        return None

    end = _parse_clock_time_minutes(match.group(2))
    start = _parse_clock_time_minutes(match.group(1), reference=end)
    if start is None or end is None:
        return None
    return start, end


def _parse_clock_time_minutes(value: str, reference: int | None = None) -> int | None:
    text = value.strip().lower().replace(".", "")
    match = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", text)
    if not match:
        return None

    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    if minute > 59:
        return None

    suffix = match.group(3)
    if suffix:
        if hour < 1 or hour > 12:
            return None
        if suffix == "pm" and hour != 12:
            hour += 12
        elif suffix == "am" and hour == 12:
            hour = 0
    elif hour <= 12 and reference is not None:
        ref_hour = reference // 60
        if ref_hour >= 12 and hour < 12:
            hour += 12
    elif hour > 23:
        return None

    return hour * 60 + minute


def _is_recent_charging_stop_navigation_request(
    text: str, state: ControllerState
) -> bool:
    station_name = state.recent_selected_charging_poi_name
    if not station_name:
        return False
    return bool(
        "navigation" in text
        and "then to" in text
        and _normalized_text(station_name) in _normalized_text(text)
    )


def _new_route_energy_flow(text: str) -> RouteEnergyFlow:
    flow = RouteEnergyFlow(active=True)
    _update_route_energy_from_text(flow, text)
    return flow


def _hydrate_route_energy_from_recent(
    flow: RouteEnergyFlow, state: ControllerState
) -> None:
    if (
        not flow.wants_charging_time
        and not flow.wants_call
        and not flow.wants_navigation_setup
    ):
        return
    if not state.recent_charging_pois:
        return
    if not _recent_charging_context_matches(flow, state):
        return
    flow.pois = list(state.recent_charging_pois)
    flow.pois_checked = True
    if flow.route_id is None:
        flow.route_id = state.recent_charging_route_id
    if flow.at_kilometer is None:
        flow.at_kilometer = state.recent_charging_at_kilometer
    if flow.search_mode is None and flow.route_id is not None:
        flow.search_mode = "along_route"
    if state.recent_selected_charging_poi_id is not None:
        flow.selected_poi_id = state.recent_selected_charging_poi_id
        flow.selected_poi_name = state.recent_selected_charging_poi_name
        flow.selected_plug_id = state.recent_selected_charging_plug_id
        flow.selected_phone_number = state.recent_selected_charging_phone_number
        selected = _select_route_energy_charging_option(flow)
        if selected is not None:
            _record_route_energy_selected_poi(flow, selected)


def _remember_route_energy_selection(
    state: ControllerState, flow: RouteEnergyFlow
) -> None:
    if flow.selected_poi_id is None:
        return
    state.recent_selected_charging_poi_id = flow.selected_poi_id
    state.recent_selected_charging_poi_name = flow.selected_poi_name
    state.recent_selected_charging_plug_id = flow.selected_plug_id
    state.recent_selected_charging_phone_number = flow.selected_phone_number


def _recent_charging_context_matches(
    flow: RouteEnergyFlow, state: ControllerState
) -> bool:
    if not flow.wants_charger_search:
        return True
    if flow.search_mode == "along_route":
        if state.recent_charging_route_id is None:
            return False
        if (
            flow.at_kilometer is not None
            and state.recent_charging_at_kilometer is not None
            and abs(flow.at_kilometer - state.recent_charging_at_kilometer) > 5
        ):
            return False
    if flow.search_mode == "nearby" and state.recent_charging_route_id is not None:
        return False
    return True


def _update_route_energy_from_text(flow: RouteEnergyFlow, text: str) -> None:
    lowered = text.lower()
    mentions_current_navigation = _route_energy_mentions_current_navigation(lowered)

    destination = _extract_route_energy_destination(text)
    if (
        destination is not None
        and flow.destination_name is None
        and not mentions_current_navigation
    ):
        flow.destination_name = destination
        flow.route_selection = "planned_route"

    route_preference = _extract_route_preference(lowered)
    if route_preference is not None:
        flow.route_preference = route_preference

    station_route_preference = _extract_station_route_preference(lowered)
    if station_route_preference is not None:
        flow.station_route_preference = station_route_preference

    route_index = _extract_route_choice_index(lowered)
    if route_index is not None:
        flow.selected_route_index = route_index

    route_via = _extract_route_via(lowered)
    if route_via is not None:
        flow.final_route_via = route_via

    if mentions_current_navigation:
        flow.needs_current_navigation = True

    if _is_route_energy_navigation_setup_request(lowered):
        flow.wants_navigation_setup = True
        flow.search_mode = None
        if "charging station" in lowered or "charging stop" in lowered or "charger" in lowered:
            flow.wants_charger_search = True
            flow.search_mode = "along_route"

    companion_category = _companion_poi_category_for_charging_stop(lowered)
    if companion_category is not None:
        flow.companion_poi_category = companion_category
        flow.wants_charger_search = True
        flow.search_mode = "along_route"
        time_window = _extract_time_window_minutes(lowered)
        if time_window is None and "dinner" in lowered:
            time_window = (19 * 60, 21 * 60)
        elif time_window is None and "lunch" in lowered:
            time_window = (12 * 60, 14 * 60)
        if time_window is not None:
            (
                flow.arrival_window_start_minutes,
                flow.arrival_window_end_minutes,
            ) = time_window
        if any(
            marker in lowered
            for marker in ("open", "dinner", "lunch", "between", "7 pm", "19:")
        ):
            flow.companion_poi_filters = ["any::currently_open"]

    if "final destination" in lowered or "longer journey" in lowered:
        flow.route_selection = "last_segment"

    if any(term in lowered for term in ("calculate the range", "calculate range")):
        flow.wants_range = True
    elif "driving distance" in lowered:
        flow.wants_range = True
    elif "how far" in lowered and any(
        marker in lowered
        for marker in ("%", "percent", "down to", "until empty", "state of charge")
    ):
        flow.wants_range = True

    if _is_charge_stop_count_request(lowered):
        flow.wants_stop_count = True
        flow.wants_range = True
        if flow.route_preference is None and "fastest" in lowered:
            flow.route_preference = "fastest"

    distance_range = _extract_soc_distance_range(lowered)
    if distance_range is not None:
        flow.initial_soc, flow.final_soc = distance_range
        if flow.wants_stop_count or flow.wants_range:
            flow.wants_range = True

    safety_buffer = _extract_safety_buffer_soc(lowered)
    if safety_buffer is not None:
        flow.final_soc = safety_buffer
        if "range" in lowered or "battery" in lowered:
            flow.wants_range = True

    if "until empty" in lowered and flow.initial_soc is not None:
        flow.final_soc = 0
        flow.wants_range = True

    target_soc = _extract_charging_target_soc(lowered)
    if target_soc is not None:
        flow.target_soc = target_soc
        if not flow.wants_stop_count:
            flow.wants_charging_time = "charge" in lowered or "charging" in lowered

    start_soc = _extract_charging_start_soc(lowered)
    if start_soc is not None:
        flow.start_soc_for_charging = start_soc

    at_kilometer = _extract_route_kilometer(lowered)
    if at_kilometer is not None:
        flow.at_kilometer = at_kilometer

    filters = _extract_charging_filters(lowered)
    for filter_name in filters:
        if filter_name not in flow.filters:
            flow.filters.append(filter_name)

    if (
        not flow.wants_navigation_setup
        and not _is_named_charging_station_followup(lowered)
        and ("charging station" in lowered or "charger" in lowered)
    ):
        flow.wants_charger_search = True
        flow.search_mode = _route_energy_search_mode(lowered, flow)
    elif flow.wants_stop_count and flow.destination_name is not None:
        flow.search_mode = "along_route"

    if "how long" in lowered and ("charge" in lowered or "charging" in lowered):
        flow.wants_charging_time = True

    if any(
        term in lowered
        for term in (
            "current range",
            "current charge",
            "current battery",
            "battery status",
            "battery range",
            "battery information",
            "charging information",
            "charging capabilities",
            "remaining range",
            "how much can it hold",
            "how fast can it charge",
            "how far can i go",
            "battery level",
            "enough battery",
            "charging station needed",
            "low battery",
        )
    ):
        if any(
            range_term in lowered
            for range_term in (
                "battery range",
                "current range",
                "remaining range",
                "how far can i go",
            )
        ):
            flow.wants_current_range = True
        flow.needs_charging_status = True
    if "enough battery" in lowered or re.search(
        r"\b(?:battery|range)\b[^.?!]{0,40}\b(?:enough|sufficient)\b", lowered
    ):
        flow.wants_battery_sufficiency = True
        flow.needs_charging_status = True

    if flow.wants_charging_time and flow.start_soc_for_charging is None:
        flow.needs_charging_status = True

    if "phone number" in lowered or re.search(r"\bcall\b", lowered):
        flow.wants_call = True


def _extract_route_energy_destination(text: str) -> str | None:
    destination = _extract_navigation_destination(text)
    then_destination = _extract_then_destination(text)
    if then_destination is not None:
        return then_destination
    if destination is not None:
        return destination

    location = _location_name_pattern()
    patterns = (
        rf"\b(?i:set up|start|create|plan)\b[^.?!]{{0,60}}\b(?i:navigation|route)\b[^.?!]{{0,40}}\b(?i:to)\s+{location}",
        rf"\b(?i:trip|journey|route|drive|driving|travel|travelling|traveling|heading|head)\b"
        rf"[^.?!]{{0,60}}\b(?i:to|toward|towards)\s+{location}",
        rf"\b(?i:get|go|going|reach)\s+(?i:to)\s+{location}",
        rf"\b(?i:on my way|on the way)\s+(?i:to)\s+{location}",
        rf"\b(?i:for)\s+{location}\s*(?:[.?!]|$)",
    )
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if not matches:
            continue
        value = matches[-1]
        if isinstance(value, tuple):
            value = value[-1]
        destination = _clean_location_query(value)
        if destination:
            return destination
    return None


def _extract_then_destination(text: str) -> str | None:
    location = _location_name_pattern()
    match = re.search(rf"\b(?i:then)\s+(?i:to)\s+{location}", text)
    if not match:
        return None
    value = match.group(1) if match.groups() else match.group(0)
    return _clean_location_query(value)


def _route_energy_mentions_current_navigation(text: str) -> bool:
    return any(
        phrase in text
        for phrase in (
            "current route",
            "active navigation",
            "my route",
            "actual route",
            "along the route",
            "along my route",
            "along the way",
            "first destination",
            "multi-stop",
            "multistop",
        )
    )


def _is_route_energy_navigation_setup_request(text: str) -> bool:
    if re.search(
        r"\b(?:do not|don't|dont|not)\b[^.?!]{0,40}\b(?:add|set up|start|create|plan)\b"
        r"[^.?!]{0,80}\bnavigation\b",
        text,
    ):
        return False
    return bool(
        re.search(r"\b(set up|start|create|add|plan)\b[^.?!]{0,80}\bnavigation\b", text)
        and any(
            phrase in text
            for phrase in ("charging stop", "charging station", "charger")
        )
    )


def _is_named_charging_station_followup(text: str) -> bool:
    if any(
        phrase in text
        for phrase in (
            "search for",
            "find",
            "look for",
            "nearby",
            "along the route",
            "along my route",
            "around",
            "available right now",
            "currently available",
        )
    ):
        return False
    return bool(
        re.search(
            r"\b(?:at|for)\s+(?:the\s+)?[a-z0-9+&][a-z0-9+&\s'-]{1,60}"
            r"\s+charging station\b",
            text,
        )
    )


def _extract_station_route_preference(
    text: str,
) -> Literal["fastest", "shortest"] | None:
    if re.search(r"\bto\b[^.?!]{0,80}\bfastest route\b[^.?!]{0,80}\bthen to\b", text):
        return "fastest"
    if re.search(r"\bto\b[^.?!]{0,80}\bshortest route\b[^.?!]{0,80}\bthen to\b", text):
        return "shortest"
    if re.search(
        r"\bfastest\b[^.?!]{0,80}\b(?:charging station|charging stop|charger)\b",
        text,
    ):
        return "fastest"
    if re.search(
        r"\bshortest\b[^.?!]{0,80}\b(?:charging station|charging stop|charger)\b",
        text,
    ):
        return "shortest"
    return None


def _extract_route_via(text: str) -> str | None:
    match = re.search(r"\bvia\s+([a-z0-9,\s-]{2,60})(?:[.?!]|$)", text)
    if not match:
        return None
    value = " ".join(match.group(1).strip(" ,.;").split())
    return value or None


def _route_energy_search_mode(
    text: str, flow: RouteEnergyFlow
) -> Literal["nearby", "along_route"]:
    if flow.at_kilometer is not None:
        if any(
            phrase in text
            for phrase in (
                "from now",
                "from my current location",
                "from current location",
                "into my journey",
                "into the journey",
                "along",
                "first destination",
                "my route",
                "the route",
            )
        ):
            return "along_route"
        if flow.needs_current_navigation and flow.at_kilometer >= 20:
            return "along_route"
    if any(
        phrase in text
        for phrase in (
            "nearby",
            "near me",
            "around here",
            "current location",
            "before i start",
            "before starting",
            "before the trip",
        )
    ):
        return "nearby"
    if any(phrase in text for phrase in ("along", "route", "on the way", "way to")):
        return "along_route"
    if flow.destination_name is not None:
        return "along_route"
    return "nearby"


def _is_charge_stop_count_request(text: str) -> bool:
    if not re.search(r"\b(?:charging|charge)\s+stops?\b", text):
        return False
    return any(phrase in text for phrase in ("how many", "need", "would i need"))


def _extract_soc_distance_range(text: str) -> tuple[float, float] | None:
    charge_cycle_patterns = (
        r"\b(?:start(?:ing)?\s+to\s+charge|start(?:ing)?\s+charging|start(?:ing)?\s+charge)"
        r"\s+(?:at|from)?\s*(\d{1,3})\s*(?:%|percent)"
        r"[^.?!]{0,120}\b(?:stop(?:ping)?|stopping\s+charging|until|to)"
        r"\s+(?:at\s+)?(\d{1,3})\s*(?:%|percent)",
        r"\b(?:charge|charging)\s+from\s*(\d{1,3})\s*(?:%|percent)"
        r"[^.?!]{0,80}\b(?:to|until)\s*(\d{1,3})\s*(?:%|percent)",
        r"\b(?:charge|charging)\s+from\s*(\d{1,3})"
        r"[^.?!]{0,80}\b(?:to|until)\s*(\d{1,3})\s*(?:%|percent)",
    )
    for pattern in charge_cycle_patterns:
        match = re.search(pattern, text)
        if match:
            low = float(match.group(1))
            high = float(match.group(2))
            if high > low:
                return high, low
            return low, high

    patterns = (
        r"\b(?:from|at|current(?:ly)? at|currently have)\s*(\d{1,3})\s*(?:%|percent)"
        r"[^.?!]{0,80}\b(?:down to|until|to|hit|reach)\s*(\d{1,3})\s*(?:%|percent)",
        r"\b(\d{1,3})\s*(?:%|percent)[^.?!]{0,80}\b(?:down to|until|to|hit|reach)"
        r"\s*(\d{1,3})\s*(?:%|percent)",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return float(match.group(1)), float(match.group(2))

    charge_cycle = re.search(
        r"\b(?:get to|reach|at)\s*(\d{1,3})\s*(?:%|percent)"
        r"[^.?!]{0,120}\b(?:until|to)\s*(\d{1,3})\s*(?:%|percent)",
        text,
    )
    if charge_cycle:
        low = float(charge_cycle.group(1))
        high = float(charge_cycle.group(2))
        return high, low

    until_empty = re.search(
        r"\b(\d{1,3})\s*(?:%|percent)[^.?!]{0,80}\b(?:until|to)\s+empty",
        text,
    )
    if until_empty:
        return float(until_empty.group(1)), 0.0

    return None


def _extract_safety_buffer_soc(text: str) -> float | None:
    if match := re.search(
        r"\bdown to\s*(\d{1,3})\s*(?:%|percent)[^.?!]{0,50}"
        r"\b(?:safety buffer|buffer|minimum)\b",
        text,
    ):
        return float(match.group(1))
    if match := re.search(
        r"\b(?:safety buffer|buffer|minimum)\b[^.?!]{0,50}"
        r"\b(?:at least|of|at|to)?\s*(\d{1,3})\s*(?:%|percent)",
        text,
    ):
        return float(match.group(1))
    if match := re.search(
        r"((?:\d{1,3}\s*(?:%|percent)[^.?!]{0,30}){1,3})"
        r"\b(?:safety buffer|buffer|minimum)\b",
        text,
    ):
        values = re.findall(r"\d{1,3}", match.group(1))
        if values:
            return float(values[-1])
    patterns = (
        r"\b(?:keep|keeping|hit|reach|minimum|buffer)[^.?!]{0,40}"
        r"(\d{1,3})\s*(?:%|percent)",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return float(match.group(1))
    return None


def _extract_charging_target_soc(text: str) -> float | None:
    patterns = (
        r"\b(?:charge|charging)[^.?!]{0,80}\b(?:to|until|target)\s*(\d{1,3})\s*(?:%|percent)",
        r"\b(?:to|until|target|standard)\s*(\d{1,3})\s*(?:%|percent)[^.?!]{0,80}\b(?:state of charge|soc|charge)",
        r"\bstandard\s*(\d{1,3})\s*(?:%|percent)",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return float(match.group(1))
    return None


def _extract_charging_start_soc(text: str) -> float | None:
    patterns = (
        r"\b(?:from|starting at|start(?:ing)? from)\s*(\d{1,3})\s*(?:%|percent)"
        r"[^.?!]{0,80}\b(?:to|until)\s*(\d{1,3})\s*(?:%|percent)",
        r"\b(?:once you get to|when .*?get to|at)\s*(\d{1,3})\s*(?:%|percent)"
        r"[^.?!]{0,120}\b(?:charge|charging)",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return float(match.group(1))
    return None


def _extract_route_kilometer(text: str) -> float | None:
    patterns = (
        r"\b(?:around|about|near|at|approximately)\s*(\d+(?:\.\d+)?)\s*(?:km|kilometers?)\b",
        r"\b(\d+(?:\.\d+)?)\s*(?:km|kilometers?)\s+(?:from now|into|along)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return float(match.group(1))
    return None


def _extract_charging_filters(text: str) -> list[str]:
    filters: list[str] = []
    if any(term in text for term in ("available plug", "currently available", "available")):
        filters.append("charging_stations::has_available_plug")
    if any(term in text for term in ("dc", "fast charger", "fast charging")):
        filters.append("charging_stations::has_dc_plug")
    return filters


def _resolve_route_energy_route_id(flow: RouteEnergyFlow) -> None:
    if flow.route_id is not None:
        return
    if flow.routes_to_final_destination_id:
        index = -1 if flow.route_selection == "last_segment" else 0
        flow.route_id = flow.routes_to_final_destination_id[index]
        return
    selected_route = _select_requested_route(
        flow.routes, flow.route_preference, flow.selected_route_index
    )
    if selected_route is None and len(flow.routes) == 1:
        selected_route = flow.routes[0]
    if selected_route is None:
        return
    route_id = selected_route.get("route_id")
    if isinstance(route_id, str) and route_id:
        flow.route_id = route_id


def _route_energy_start_id(
    state: ControllerState, flow: RouteEnergyFlow
) -> str | None:
    if flow.route_start_id:
        return flow.route_start_id
    if flow.route_selection == "last_segment" and len(flow.waypoints_id) >= 2:
        return flow.waypoints_id[-2]
    return state.runtime.location_id


def _record_route_energy_selected_poi(
    flow: RouteEnergyFlow, poi: dict[str, Any]
) -> None:
    poi_id = poi.get("id")
    if isinstance(poi_id, str):
        flow.selected_poi_id = poi_id
    name = poi.get("name")
    if isinstance(name, str):
        flow.selected_poi_name = name
    phone_number = poi.get("phone_number")
    if isinstance(phone_number, str):
        flow.selected_phone_number = phone_number
    plug_id = _select_charging_plug_id(poi, flow.filters)
    if plug_id is not None:
        flow.selected_plug_id = plug_id


def _select_route_energy_charging_option(
    flow: RouteEnergyFlow,
) -> dict[str, Any] | None:
    if not flow.pois:
        return None
    if flow.selected_poi_id is not None:
        for poi in flow.pois:
            if poi.get("id") == flow.selected_poi_id:
                return poi
    candidates = [
        poi
        for poi in flow.pois
        if _select_charging_plug_id(poi, flow.filters) is not None
    ]
    if not candidates:
        candidates = flow.pois
    return max(candidates, key=lambda poi: _charging_poi_score(poi, flow.filters))


def _select_charging_plug_id(
    poi: dict[str, Any], filters: list[str]
) -> str | None:
    plugs = poi.get("charging_plugs")
    if not isinstance(plugs, list):
        return None
    scored: list[tuple[tuple[int, int, float], str]] = []
    require_available = "charging_stations::has_available_plug" in filters
    require_dc = "charging_stations::has_dc_plug" in filters
    for plug in plugs:
        if not isinstance(plug, dict):
            continue
        plug_id = plug.get("plug_id")
        if not isinstance(plug_id, str):
            continue
        available = str(plug.get("availability", "")).lower() == "available"
        dc = str(plug.get("power_type", "")).upper() == "DC"
        if require_available and not available:
            continue
        if require_dc and not dc:
            continue
        power = _safe_float(plug.get("power_kw"), 0.0) or 0.0
        scored.append(((1 if available else 0, 1 if dc else 0, power), plug_id))
    if not scored:
        return None
    return max(scored)[1]


def _charging_poi_score(poi: dict[str, Any], filters: list[str]) -> tuple[int, int, float]:
    plugs = poi.get("charging_plugs")
    if not isinstance(plugs, list):
        return (0, 0, 0.0)
    best = (0, 0, 0.0)
    for plug in plugs:
        if not isinstance(plug, dict):
            continue
        available = str(plug.get("availability", "")).lower() == "available"
        dc = str(plug.get("power_type", "")).upper() == "DC"
        power = _safe_float(plug.get("power_kw"), 0.0) or 0.0
        best = max(best, (1 if available else 0, 1 if dc else 0, power))
    return best


def _route_energy_start_soc(flow: RouteEnergyFlow) -> float | None:
    if flow.start_soc_for_charging is not None:
        return flow.start_soc_for_charging
    if flow.search_mode == "nearby" and flow.state_of_charge is not None:
        return flow.state_of_charge
    if (
        flow.state_of_charge is not None
        and flow.remaining_range_km
        and flow.at_kilometer is not None
    ):
        soc_per_km = flow.state_of_charge / flow.remaining_range_km
        estimated = max(0.0, flow.state_of_charge - flow.at_kilometer * soc_per_km)
        return max(0.0, min(100.0, round(estimated / 5.0) * 5.0))
    if flow.initial_soc is not None:
        return flow.initial_soc
    return None


def _route_energy_selected_route_distance(flow: RouteEnergyFlow) -> float | None:
    selected_route = _route_energy_selected_route(flow)
    if selected_route is not None:
        distance = _safe_float(selected_route.get("distance_km"))
        if distance is not None:
            return distance

    if flow.route_id is not None:
        for route in flow.route_details:
            if route.get("route_id") != flow.route_id:
                continue
            distance = _safe_float(route.get("distance_km"))
            if distance is not None:
                return distance
    return None


def _route_energy_default_search_kilometer(flow: RouteEnergyFlow) -> float | None:
    distance = _route_energy_selected_route_distance(flow)
    if distance is None or distance <= 0:
        return None
    return int(round(distance / 2.0))


def _route_energy_selected_route_duration_minutes(flow: RouteEnergyFlow) -> int | None:
    route = _route_energy_selected_route(flow)
    if route is None:
        return None
    hours = _safe_int(route.get("duration_hours"), 0) or 0
    minutes = _safe_int(route.get("duration_minutes"), 0) or 0
    total = hours * 60 + minutes
    return total if total > 0 else None


def _ensure_route_energy_search_kilometers(
    flow: RouteEnergyFlow, state: ControllerState
) -> None:
    if flow.search_kilometers:
        return
    if flow.at_kilometer is not None:
        flow.search_kilometers = [int(round(flow.at_kilometer))]
    else:
        flow.search_kilometers = _route_energy_time_window_search_kilometers(
            flow, state
        )
    if not flow.search_kilometers:
        default_km = _route_energy_default_search_kilometer(flow)
        if default_km is not None:
            flow.search_kilometers = [int(default_km)]
    if flow.search_kilometers:
        flow.search_kilometer_index = min(
            max(flow.search_kilometer_index, 0), len(flow.search_kilometers) - 1
        )
        flow.at_kilometer = flow.search_kilometers[flow.search_kilometer_index]


def _route_energy_time_window_search_kilometers(
    flow: RouteEnergyFlow, state: ControllerState
) -> list[int]:
    if (
        flow.arrival_window_start_minutes is None
        or flow.arrival_window_end_minutes is None
        or state.runtime.hour is None
    ):
        return []
    distance = _route_energy_selected_route_distance(flow)
    duration_minutes = _route_energy_selected_route_duration_minutes(flow)
    if distance is None or distance <= 0 or not duration_minutes:
        return []

    current_minutes = state.runtime.hour * 60 + (state.runtime.minute or 0)
    start_delta = _minutes_until(current_minutes, flow.arrival_window_start_minutes)
    end_delta = _minutes_until(current_minutes, flow.arrival_window_end_minutes)
    if end_delta < start_delta:
        end_delta += 24 * 60
    if start_delta > duration_minutes:
        return []

    start_ratio = max(0.0, min(1.0, start_delta / duration_minutes))
    end_ratio = max(start_ratio, min(1.0, end_delta / duration_minutes))
    start_km = distance * start_ratio
    end_km = distance * end_ratio
    return _route_energy_candidate_kilometers(start_km, end_km, distance)


def _minutes_until(current_minutes: int, target_minutes: int) -> int:
    delta = target_minutes - current_minutes
    if delta < 0:
        delta += 24 * 60
    return delta


def _route_energy_candidate_kilometers(
    start_km: float, end_km: float, route_distance_km: float
) -> list[int]:
    step = 50
    start = int(round(start_km / step) * step)
    end = int(round(end_km / step) * step)
    lower = max(1, min(start, end))
    upper = min(int(route_distance_km), max(start, end))
    if upper < lower:
        return [max(1, min(int(route_distance_km), lower))]
    candidates = list(range(lower, upper + 1, step))
    for point in (start, end):
        bounded = max(1, min(int(route_distance_km), point))
        if bounded not in candidates:
            candidates.append(bounded)
    return sorted(candidates)


def _advance_route_energy_search_kilometer(flow: RouteEnergyFlow) -> bool:
    if not flow.search_kilometers:
        return False
    next_index = flow.search_kilometer_index + 1
    if next_index >= len(flow.search_kilometers):
        return False
    flow.search_kilometer_index = next_index
    flow.at_kilometer = flow.search_kilometers[next_index]
    flow.pois_checked = False
    flow.pois = []
    flow.companion_pois_checked = False
    flow.companion_pois = []
    flow.selected_poi_id = None
    flow.selected_poi_name = None
    flow.selected_plug_id = None
    flow.selected_phone_number = None
    return True


def _filter_companion_pois_for_route_constraints(
    flow: RouteEnergyFlow, state: ControllerState, pois: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if not pois:
        return []
    if (
        flow.arrival_window_start_minutes is None
        or state.runtime.hour is None
        or flow.at_kilometer is None
    ):
        return pois
    arrival_minutes = _route_energy_arrival_minutes_at_kilometer(flow, state)
    if arrival_minutes is None:
        return pois
    return [poi for poi in pois if _poi_open_at_minutes(poi, arrival_minutes)]


def _route_energy_arrival_minutes_at_kilometer(
    flow: RouteEnergyFlow, state: ControllerState
) -> int | None:
    distance = _route_energy_selected_route_distance(flow)
    duration_minutes = _route_energy_selected_route_duration_minutes(flow)
    if (
        distance is None
        or distance <= 0
        or duration_minutes is None
        or flow.at_kilometer is None
        or state.runtime.hour is None
    ):
        return None
    current_minutes = state.runtime.hour * 60 + (state.runtime.minute or 0)
    elapsed = int(round(duration_minutes * (flow.at_kilometer / distance)))
    return (current_minutes + elapsed) % (24 * 60)


def _route_energy_selected_route(flow: RouteEnergyFlow) -> dict[str, Any] | None:
    if not flow.routes:
        return None
    selected_route = _select_requested_route(
        flow.routes, flow.route_preference, flow.selected_route_index
    )
    if selected_route is not None:
        return selected_route
    if flow.route_id is not None:
        for route in flow.routes:
            if route.get("route_id") == flow.route_id:
                return route
    return None


def _format_route_energy_completion(flow: RouteEnergyFlow) -> str:
    if flow.completion_message:
        return flow.completion_message
    if flow.call_completed:
        return "Done, I called the charging station."
    if flow.charging_time_checked:
        target = _format_percentage(flow.target_soc)
        if flow.charging_minutes is not None:
            minutes = int(round(flow.charging_minutes))
            station = flow.selected_poi_name or "the charging station"
            return f"Charging at {station} to {target} will take about {minutes} minutes."
        return f"I calculated the charging time to {target}."
    if flow.pois_checked and flow.pois:
        station = flow.selected_poi_name or str(flow.pois[0].get("name", "a station"))
        if flow.search_mode == "along_route":
            return f"I found {station} along the route."
        return f"I found {station} nearby."
    if flow.wants_battery_sufficiency and flow.remaining_range_km is not None:
        route_distance = _route_energy_selected_route_distance(flow)
        if route_distance is not None:
            if flow.remaining_range_km >= route_distance:
                return (
                    f"Yes, your remaining range is about {flow.remaining_range_km:g} km "
                    f"and the route is about {route_distance:g} km."
                )
            return (
                f"No, your remaining range is about {flow.remaining_range_km:g} km "
                f"and the route is about {route_distance:g} km."
            )
        return f"Your remaining range is about {flow.remaining_range_km:g} km."
    if (
        flow.wants_current_range
        and flow.remaining_range_km is not None
        and not flow.distance_checked
    ):
        return f"Your remaining range is about {flow.remaining_range_km:g} km."
    if flow.distance_checked:
        if flow.wants_stop_count and flow.distance_km and flow.distance_km > 0:
            route_distance = _route_energy_selected_route_distance(flow)
            if route_distance is not None:
                stops = max(0, math.ceil(route_distance / flow.distance_km) - 1)
                plural = "stop" if stops == 1 else "stops"
                selected_route = _route_energy_selected_route(flow)
                route_label = (
                    _route_choice_label(selected_route, flow.route_preference)
                    if selected_route is not None
                    else None
                )
                route_phrase = (
                    f"the {route_label} route" if route_label else "that route"
                )
                message = (
                    f"You would need about {stops} charging {plural} "
                    f"for {route_phrase}."
                )
                if route_label and len(flow.routes) > 1:
                    message += " Would you like more information on alternative routes?"
                return message
        if flow.distance_km is not None:
            route_summary = _format_current_route_energy_summary(flow)
            if route_summary is not None:
                return (
                    f"{route_summary} You can drive about {flow.distance_km:g} km "
                    "for that battery range."
                )
            return f"You can drive about {flow.distance_km:g} km for that battery range."
        return "I checked the driving distance for that battery range."
    return "Done, I checked the route and charging information."


def _format_route_energy_navigation_completion(
    flow: RouteEnergyFlow,
    route_to_poi: dict[str, Any],
    route_from_poi: dict[str, Any],
) -> str:
    selected_fastest = _route_has_alias(route_to_poi, "fastest") and _route_has_alias(
        route_from_poi, "fastest"
    )
    if selected_fastest:
        message = (
            "Done, I started navigation using the fastest available route with the "
            "charging stop."
        )
    else:
        message = "Done, navigation is started with the charging stop."

    has_alternatives = (
        len(flow.routes_to_poi) > 1
        or len(flow.routes_from_poi) > 1
        or len(flow.routes) > 1
    )
    if has_alternatives:
        message += " Would you like more information on the alternative routes?"
    return message


def _format_current_route_energy_summary(flow: RouteEnergyFlow) -> str | None:
    if not flow.route_details:
        return None
    distances = [
        distance
        for route in flow.route_details
        if (distance := _safe_float(route.get("distance_km"))) is not None
    ]
    if not distances:
        return None
    total_distance = sum(distances)
    fastest = all(_route_has_alias(route, "fastest") for route in flow.route_details)
    route_text = f"The total route distance is about {total_distance:g} km."
    if fastest:
        route_text += " The current route uses the fastest route segments."
    return route_text


def _extract_km_value(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:km|kilometers?)", value)
    if match:
        return float(match.group(1))
    return None


def _extract_first_km_value(value: Any) -> float | None:
    if isinstance(value, dict):
        for nested in value.values():
            extracted = _extract_first_km_value(nested)
            if extracted is not None:
                return extracted
    if isinstance(value, list):
        for nested in value:
            extracted = _extract_first_km_value(nested)
            if extracted is not None:
                return extracted
    return _extract_km_value(value)


def _extract_minutes_value(value: Any) -> float | None:
    if isinstance(value, dict):
        if "minutes" in value and isinstance(value.get("minutes"), (int, float)):
            hours = _safe_float(value.get("hour"), 0.0) or 0.0
            return hours * 60 + float(value["minutes"])
        for nested in value.values():
            extracted = _extract_minutes_value(nested)
            if extracted is not None:
                return extracted
    if isinstance(value, list):
        for nested in value:
            extracted = _extract_minutes_value(nested)
            if extracted is not None:
                return extracted
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"(\d+(?:\.\d+)?)\s*(?:minutes?|mins?|min)\b", value)
        if match:
            return float(match.group(1))
    return None


def _is_poi_request(text: str) -> bool:
    if _poi_category_from_text(text) is None:
        return False
    if (
        "charging station" in text
        and ("along the route" in text or "stop" in text)
        and re.search(r"\b(route|navigation|navigate)\b", text)
    ):
        return False
    return bool(
        any(
            marker in text
            for marker in (
                "find",
                "search",
                "look for",
                "nearby",
                "near me",
                "around here",
                "options",
                "available",
                "good",
                "where",
                "directions",
                "destination",
                "go to",
                "grab",
                "stop",
                "still open",
            )
        )
    )


def _is_along_route_poi_request(text: str) -> bool:
    return bool(
        re.search(r"\b(along|on)\b[^.?!]{0,40}\b(route|way|trip)\b", text)
        or "along the whole route" in text
        or "along this route" in text
        or "on the way" in text
    )


def _route_id_for_poi_followup(state: ControllerState) -> str | None:
    if state.planned_navigation_route_ids:
        return state.planned_navigation_route_ids[-1]
    if state.recent_navigation_route_ids:
        return state.recent_navigation_route_ids[-1]
    if state.navigation.routes_to_final_destination_id:
        return state.navigation.routes_to_final_destination_id[-1]
    return None


def _extract_poi_location(text: str) -> str | None:
    name = (
        r"[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ'’.-]*"
        r"(?:\s+(?:[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ'’.-]*|"
        r"la|de|del|di|du|da|das|dos|and|of)){0,5}"
    )
    patterns = (
        rf"\b(?i:restaurants?|fast food|bakery|supermarket|parking|toilets?)"
        rf"\b[^.?!]{{0,80}}\b(?i:in|near|around|at)\s+({name})",
        rf"\b(?i:find|search|look for|show|get)\b[^.?!]{{0,100}}"
        rf"\b(?i:in|near|around|at)\s+({name})",
        rf"\b(?i:go|going|drive|head|navigate)\b[^.?!]{{0,100}}"
        rf"\b(?i:to)\s+({name})",
        rf"\b(?i:destination)\b[^.?!]{{0,80}}\b(?i:to)\s+({name})",
    )
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if not matches:
            continue
        location = _clean_poi_location(matches[-1])
        if location:
            return location
    return None


def _clean_poi_location(value: str) -> str | None:
    location = _clean_location_query(value)
    if location is None:
        return None
    lowered = location.lower()
    blocked = {
        "restaurant",
        "restaurants",
        "parking",
        "toilet",
        "toilets",
        "supermarket",
        "bakery",
        "airport",
    }
    if lowered in blocked:
        return None
    return location


def _is_nearby_poi_request(text: str) -> bool:
    return any(
        phrase in text
        for phrase in (
            "nearby",
            "near me",
            "around here",
            "close by",
            "current location",
            "where i am",
            "here",
        )
    )


def _do_not_set_navigation(text: str) -> bool:
    if (
        _is_replace_destination_request(text)
        or "changed my mind" in text
        or "destination" in text
    ):
        return False
    if any(
        phrase in text
        for phrase in (
            "don't set",
            "do not set",
            "without setting",
            "don't navigate",
            "do not navigate",
        )
    ):
        return True
    if any(
        phrase in text
        for phrase in (
            "options",
            "available",
            "curious",
            "potentially",
            "maybe",
            "how long",
            "nearby",
        )
    ):
        return True
    return not bool(
        re.search(r"\b(navigate|directions|route|drive|go|take me|set)\b", text)
    )


def _is_navigation_planning_only_request(text: str) -> bool:
    if any(
        phrase in text
        for phrase in (
            "don't want to start navigation",
            "do not want to start navigation",
            "don't start navigation",
            "do not start navigation",
            "not start navigation yet",
            "don't want to set up navigation",
            "do not want to set up navigation",
        )
    ):
        return True
    return bool(
        ("plan a trip" in text or "plan the trip" in text or "plan my trip" in text)
        and not re.search(r"\b(set up|start)\s+navigation\b", text)
    )


def _is_trip_planning_navigation_request(text: str) -> bool:
    if not any(
        phrase in text
        for phrase in (
            "plan a trip",
            "plan the trip",
            "plan my trip",
            "multi-stop trip",
            "multistop trip",
            "whole trip",
        )
    ):
        return False
    return bool(re.search(r"\b(navigate|navigation|route|go|drive|travel)\b", text))


def _extract_two_leg_navigation_destinations(text: str) -> tuple[str, str] | None:
    location = _location_name_pattern()
    patterns = (
        rf"\b(?i:navigate|navigation|directions?|drive|go|travel|route)\b"
        rf"[^.?!]{{0,120}}?\bto\s+{location}\s*,?\s*(?:and\s+)?then\s+to\s+{location}",
        rf"\b(?i:from)\s+{location}\s+\b(?i:to)\s+{location}"
        rf"\s*,?\s*(?:and\s+)?then\s+to\s+{location}",
        rf"\b(?i:to)\s+{location}\s*,?\s*(?:and\s+)?then\s+to\s+{location}",
    )
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if not matches:
            continue
        match = matches[-1]
        values = match if isinstance(match, tuple) else (match,)
        if len(values) < 2:
            continue
        first_raw, final_raw = values[-2], values[-1]
        first_destination = _clean_location_query(first_raw)
        final_destination = _clean_location_query(final_raw)
        if (
            first_destination
            and final_destination
            and not _is_generic_navigation_destination(first_destination)
            and not _is_generic_navigation_destination(final_destination)
        ):
            return first_destination, final_destination
    return None


def _is_generic_navigation_destination(name: str) -> bool:
    normalized = _normalized_text(name)
    generic_terms = {
        "airport",
        "bakery",
        "charging station",
        "charger",
        "fast food",
        "grocery",
        "parking",
        "public toilet",
        "restaurant",
        "restroom",
        "supermarket",
    }
    return normalized in generic_terms


def _is_planned_navigation_setup_request(text: str) -> bool:
    if any(
        phrase in text
        for phrase in (
            "don't",
            "do not",
            "not yet",
            "without starting",
            "without setting",
        )
    ):
        return False
    if re.search(
        r"\b(after|find|search|look for|open|restaurant|supermarket|charging station|"
        r"what(?:'s| is)|which|where)\b",
        text,
    ):
        return False
    return bool(
        re.search(
            r"\b(set up|start|begin|launch)\b[^.?!]{0,80}\b(navigation|trip|route)\b",
            text,
        )
        or re.search(
            r"\b(yes|confirm|confirmed)\b[^.?!]{0,80}"
            r"\b(set up|start|begin|launch)\b",
            text,
        )
    )


def _is_add_waypoint_poi_request(text: str) -> bool:
    return bool(
        re.search(r"\b(add|insert|include|put)\b[^.?!]{0,80}\b(stop|waypoint)\b", text)
        or re.search(r"\b(stop at|add a stop|add it as a stop)\b", text)
    )


def _extract_toll_avoidance_tolerance_minutes(text: str) -> int | None:
    if "toll" not in text:
        return None
    if not any(phrase in text for phrase in ("without toll", "avoid toll", "no toll")):
        return None
    match = re.search(
        r"(?:not more than|within|no more than)\s+(\d{1,3})\s*minutes?",
        text,
    )
    if match:
        return int(match.group(1))
    return 0


def _calendar_meeting_navigation_location(
    text: str, state: ControllerState
) -> str | None:
    if not re.search(
        r"\b(direction|directions|navigate|navigation|route|drive|get to|go to)\b",
        text,
    ):
        return None
    for meeting in state.recent_calendar_meetings:
        topic = meeting.get("topic")
        location = meeting.get("location")
        if not isinstance(location, str) or not location.strip():
            continue
        topic_matches = isinstance(topic, str) and _normalized_text(
            topic
        ) in _normalized_text(text)
        location_matches = _normalized_text(location) in _normalized_text(text)
        if topic_matches or location_matches:
            return location
    return None


def _is_replace_destination_request(text: str) -> bool:
    return bool(
        (
            re.search(r"\b(change|replace|switch|update)\b", text)
            and re.search(r"\b(destination|navigation|route)\b", text)
        )
        or "rather go to" in text
        or "instead of" in text
    )


def _extract_navigation_destination(text: str) -> str | None:
    patterns = (
        r"\b(?i:change|replace|switch|update)\b[^.?!]{0,100}?\b(?i:destination)\b[^.?!]{0,80}?\bto\s+([A-Z][A-Za-z]*(?:\s+(?:[A-Z][A-Za-z]*|la|de|del|di|and)){0,4})",
        r"\b(?i:change|replace|switch|update)\b[^.?!]{0,100}?\b(?i:navigation|route)\b[^.?!]{0,80}?\bto\s+([A-Z][A-Za-z]*(?:\s+(?:[A-Z][A-Za-z]*|la|de|del|di|and)){0,4})",
        r"\b(?i:navigate|drive|go|travel|route)\b[^.?!]{0,80}?\bto\s+([A-Z][A-Za-z]*(?:\s+(?:[A-Z][A-Za-z]*|la|de|del|di|and)){0,4})",
        r"\b(?i:directions?)\b[^.?!]{0,40}?\bto\s+([A-Z][A-Za-z]*(?:\s+(?:[A-Z][A-Za-z]*|la|de|del|di|and)){0,4})",
        r"\b(?i:rather go to)\s+([A-Z][A-Za-z]*(?:\s+(?:[A-Z][A-Za-z]*|la|de|del|di|and)){0,4})",
    )
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            destination = _clean_location_query(matches[-1])
            if destination:
                return destination

    return None


def _clean_location_query(value: str) -> str | None:
    value = value.split(",", 1)[0]
    value = re.split(r"[.?!]\s+", value, maxsplit=1)[0]
    value = re.split(
        r"\b(?:instead|because|with|without|if|when|that|which|where|for|from|via|then|first|next|now|rather|and|how|about)\b",
        value,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    value = re.sub(
        r"\b(?:city center|city centre|city|centre)\b",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = " ".join(value.strip(" .?!;:-").split())
    value = re.sub(r"\s+\b(?:and|or)\b$", "", value, flags=re.IGNORECASE)
    if not value:
        return None
    if value.lower() in {"the", "there", "home", "work"}:
        return None
    return value


def _matching_recent_location_id(
    requested_name: str | None,
    candidate_name: str | None,
    candidate_id: str | None,
) -> str | None:
    if not candidate_id:
        return None
    if _locations_match(requested_name, candidate_name):
        return candidate_id
    return None


def _locations_match(left: str | None, right: str | None) -> bool:
    if not left or not right:
        return False
    return _normalized_text(left) == _normalized_text(right)


def _extract_route_preference(
    text: str,
) -> Literal["fastest", "shortest"] | None:
    if "fastest" in text or "quickest" in text:
        return "fastest"
    if "shortest" in text:
        return "shortest"
    return None


def _extract_route_choice_index(text: str) -> int | None:
    ordinal_to_index = {
        "first": 0,
        "1st": 0,
        "one": 0,
        "second": 1,
        "2nd": 1,
        "two": 1,
        "third": 2,
        "3rd": 2,
        "three": 2,
    }
    if match := re.search(r"\b(?:route|option)\s+([123])\b", text):
        return int(match.group(1)) - 1
    if match := re.search(r"\b([123])(?:st|nd|rd)?\s+(?:route|option)\b", text):
        return int(match.group(1)) - 1
    for word, index in ordinal_to_index.items():
        if re.search(rf"\b{word}\s+(?:route|option)\b", text) or re.search(
            rf"\b(?:route|option)\s+{word}\b", text
        ):
            return index
    if re.search(r"\b(another|different|alternative|other)\s+(?:route|option)\b", text):
        return 1
    return None


def _select_route_index_by_via(
    routes: list[dict[str, Any]], text: str
) -> int | None:
    text_norm = _normalized_text(text)
    requested_roads = _extract_road_tokens(text)
    best_index: int | None = None
    best_overlap = 0

    for index, route in enumerate(routes):
        via = route.get("name_via")
        if not isinstance(via, str) or not via.strip():
            continue

        via_norm = _normalized_text(via)
        if via_norm and via_norm in text_norm:
            return index

        route_roads = _extract_road_tokens(via)
        if not requested_roads or not route_roads:
            continue
        overlap = len(requested_roads & route_roads)
        if overlap == len(requested_roads):
            return index
        if overlap > best_overlap:
            best_overlap = overlap
            best_index = index

    return best_index if best_overlap else None


def _extract_road_tokens(text: str) -> set[str]:
    return {
        re.sub(r"[^A-Z0-9]", "", token)
        for token in re.findall(r"\b[A-Z]{1,4}\s*[- ]?\s*\d+[A-Z]?\b", text.upper())
    }


def _select_poi_from_text(
    pois: list[dict[str, Any]], text: str
) -> dict[str, Any] | None:
    option_index = _extract_poi_option_index(text)
    if option_index is not None:
        if 0 <= option_index < len(pois):
            return pois[option_index]

    normalized_text = _normalized_text(text)
    for poi in pois:
        name = poi.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        normalized_name = _normalized_text(name)
        if normalized_name and normalized_name in normalized_text:
            return poi
        name_tokens = {
            token
            for token in normalized_name.split()
            if len(token) > 2 and token not in {"the", "and", "restaurant"}
        }
        if name_tokens and name_tokens.issubset(set(normalized_text.split())):
            return poi
    return None


def _extract_poi_option_index(text: str) -> int | None:
    ordinal_to_index = {
        "first": 0,
        "1st": 0,
        "one": 0,
        "second": 1,
        "2nd": 1,
        "two": 1,
        "third": 2,
        "3rd": 2,
        "three": 2,
        "fourth": 3,
        "4th": 3,
        "four": 3,
    }
    if match := re.search(r"\b(?:option|place|restaurant)\s+([1-4])\b", text):
        return int(match.group(1)) - 1
    if match := re.search(r"\b([1-4])(?:st|nd|rd|th)?\s+(?:option|place|restaurant)\b", text):
        return int(match.group(1)) - 1
    for word, index in ordinal_to_index.items():
        if re.search(rf"\b{word}\s+(?:option|place|restaurant|one)\b", text):
            return index
        if re.search(rf"\b(?:option|place|restaurant|one)\s+{word}\b", text):
            return index
    return None


def _navigation_route_start_id(
    state: ControllerState,
    navigation: NavigationFlow,
) -> str | None:
    if navigation.route_start_id:
        return navigation.route_start_id
    if (
        navigation.mode == "replace_final_destination"
        and len(navigation.waypoints_id) >= 2
    ):
        return navigation.waypoints_id[-2]
    return state.runtime.location_id


def _record_navigation_routes(
    navigation: NavigationFlow, routes: list[dict[str, Any]]
) -> None:
    if navigation.route_lookup == "delete_without_waypoint":
        navigation.route_without_waypoint_checked = True
        navigation.routes_without_waypoint = routes
    elif navigation.route_lookup == "to_new_waypoint":
        navigation.route_to_new_waypoint_checked = True
        navigation.routes_to_new_waypoint = routes
    elif navigation.route_lookup == "from_new_waypoint":
        navigation.route_from_new_waypoint_checked = True
        navigation.routes_from_new_waypoint = routes
    else:
        navigation.routes_checked = True
        navigation.routes = routes


def _navigation_choice_routes(navigation: NavigationFlow) -> list[dict[str, Any]]:
    if navigation.route_lookup == "delete_without_waypoint":
        return navigation.routes_without_waypoint
    if navigation.route_lookup == "to_new_waypoint":
        return navigation.routes_to_new_waypoint
    if navigation.route_lookup == "from_new_waypoint":
        return navigation.routes_from_new_waypoint
    return navigation.routes


def _resolve_navigation_waypoint_from_state(navigation: NavigationFlow) -> None:
    if navigation.waypoint_id:
        return

    waypoint_name = navigation.waypoint_name
    if waypoint_name:
        normalized = waypoint_name.casefold()
        for waypoint in navigation.waypoint_details:
            name = waypoint.get("name")
            waypoint_id = waypoint.get("id")
            if (
                isinstance(name, str)
                and isinstance(waypoint_id, str)
                and name.casefold() == normalized
            ):
                navigation.waypoint_id = waypoint_id
                return
        return

    intermediate_waypoints = navigation.waypoints_id[1:-1]
    if len(intermediate_waypoints) == 1:
        navigation.waypoint_id = intermediate_waypoints[0]


def _resolve_navigation_final_destination_from_state(
    navigation: NavigationFlow,
) -> None:
    if navigation.destination_id:
        return
    if not navigation.waypoints_id:
        return

    final_destination_id = navigation.waypoints_id[-1]
    if not navigation.destination_name:
        navigation.destination_id = final_destination_id
        return

    normalized = navigation.destination_name.casefold()
    for waypoint in navigation.waypoint_details:
        name = waypoint.get("name")
        waypoint_id = waypoint.get("id")
        if (
            isinstance(name, str)
            and isinstance(waypoint_id, str)
            and name.casefold() == normalized
            and waypoint_id == final_destination_id
        ):
            navigation.destination_id = waypoint_id
            return


def _navigation_adjacent_waypoints(
    navigation: NavigationFlow, waypoint_id: str
) -> tuple[str, str] | None:
    try:
        waypoint_index = navigation.waypoints_id.index(waypoint_id)
    except ValueError:
        return None

    if waypoint_index <= 0 or waypoint_index >= len(navigation.waypoints_id) - 1:
        return None

    return (
        navigation.waypoints_id[waypoint_index - 1],
        navigation.waypoints_id[waypoint_index + 1],
    )


def _select_navigation_requested_route(
    navigation: NavigationFlow,
) -> dict[str, Any] | None:
    if navigation.selected_route_index is not None:
        return _select_requested_route(
            navigation.routes,
            navigation.route_preference,
            navigation.selected_route_index,
        )
    if navigation.toll_avoidance_tolerance_minutes is None:
        return _select_requested_route(
            navigation.routes,
            navigation.route_preference,
            None,
        )
    return _select_toll_aware_route(
        navigation.routes,
        navigation.route_preference or "fastest",
        navigation.toll_avoidance_tolerance_minutes,
    )


def _is_route_alternative_detail_request(text: str) -> bool:
    patterns = (
        r"\b(?:what|which)\b[^.?!]{0,40}\b(?:other|another|alternative|second|third)\b"
        r"[^.?!]{0,40}\b(?:route|alternative|option|one)\b",
        r"\b(?:tell|describe)\b[^.?!]{0,40}\b(?:other|another|alternative|second|third)\b"
        r"[^.?!]{0,40}\b(?:route|alternative|option|one)\b",
        r"\b(?:other|another|alternative|second|third)\b[^.?!]{0,40}"
        r"\b(?:route|alternative|option|one)\b[^.?!]{0,40}\b(?:details?|info|information)\b",
    )
    return any(re.search(pattern, text) for pattern in patterns)


def _is_route_choice_preview_request(text: str) -> bool:
    if _is_route_alternative_detail_request(text):
        return True
    if not _do_not_set_navigation(text):
        return False
    return bool(
        re.search(r"\b(show|see|view|check|look)\b", text)
        and re.search(r"\b(route|option|alternative|via)\b", text)
    )


def _is_close_request(text: str) -> bool:
    return any(word in text for word in ("close", "closed", "shut", "off"))


def _extract_window_target(text: str) -> str:
    if "driver rear" in text or "rear driver" in text or "left rear" in text:
        return "DRIVER_REAR"
    if "passenger rear" in text or "rear passenger" in text or "right rear" in text:
        return "PASSENGER_REAR"
    if "driver" in text and "rear" not in text:
        return "DRIVER"
    if "passenger" in text and "rear" not in text:
        return "PASSENGER"
    return "ALL"


def _extract_window_percentage(text: str) -> int | None:
    return _extract_percentage(text)


def _extract_sunroof_percentage(
    text: str, *, allow_standalone: bool = False
) -> int | None:
    if any(word in text for word in ("half", "halfway")):
        return 50

    if re.search(r"\b50\s*%?", text):
        return 50

    if _sunroof_full_requested(text):
        return 100

    if allow_standalone:
        if re.search(r"\b100\s*%?", text) or "fully" in text or "all the way" in text:
            return 100
        match = re.search(r"\b([1-9][0-9]?|100)\s*%?\b", text)
        if match:
            return int(match.group(1))

    sunroof_clause = re.search(r"sunroof[^.?!,;]*?([1-9][0-9]?|100)\s*%?", text)
    if sunroof_clause:
        return int(sunroof_clause.group(1))

    return None


def _extract_sunshade_percentage(text: str) -> int | None:
    return _extract_percentage(text)


def _extract_percentage(text: str, *, allow_standalone: bool = False) -> int | None:
    if any(word in text for word in ("half", "halfway")):
        return 50
    if re.search(r"\b50\s*%?", text):
        return 50
    if any(phrase in text for phrase in ("fully", "full", "all the way")):
        return 100

    percent_match = re.search(r"\b([1-9][0-9]?|100)\s*%", text)
    if percent_match:
        return int(percent_match.group(1))

    if allow_standalone:
        number_match = re.search(r"\b([1-9][0-9]?|100)\b", text)
        if number_match:
            return int(number_match.group(1))

    return None


def _sunroof_full_requested(text: str) -> bool:
    # Avoid treating "open the sunshade all the way" as a sunroof target.
    if "sunshade" in text and re.search(
        r"sunshade[^.?!,;]*(fully|all the way|100)", text
    ):
        return False
    return bool(
        re.search(r"sunroof[^.?!,;]*(fully|all the way|100\s*%)", text)
        or re.search(r"(fully|all the way)[^.?!,;]*sunroof", text)
    )


AMBIENT_COLORS = {
    "RED",
    "GREEN",
    "BLUE",
    "YELLOW",
    "WHITE",
    "PINK",
    "ORANGE",
    "PURPLE",
    "CYAN",
    "NONE",
}


def _extract_ambient_color(text: str) -> str | None:
    upper = text.upper()
    for color in sorted(AMBIENT_COLORS - {"NONE"}, key=len, reverse=True):
        if re.search(rf"\b{re.escape(color)}\b", upper):
            return color
    if "off" in text or "none" in text:
        return "NONE"
    return None


def _extract_level(text: str) -> int | None:
    match = re.search(r"\blevel\s*([0-5])\b", text)
    if match:
        return int(match.group(1))

    match = re.search(r"\bfan[^.?!,;]*?\b([0-5])\b", text)
    if match:
        return int(match.group(1))

    return None


def _extract_heating_level(text: str) -> int | None:
    level = _extract_level(text)
    if level is not None:
        return level
    if "medium" in text:
        return 2
    if "low" in text:
        return 1
    if "high" in text:
        return 3
    return None


def _extract_steering_wheel_heating_level(text: str) -> int | None:
    if "steering wheel" not in text:
        return None
    match = re.search(r"steering wheel[^.?!,;]*?\blevel\s*([0-3])\b", text)
    if match:
        return int(match.group(1))
    match = re.search(r"\blevel\s*([0-3])\b[^.?!,;]*?steering wheel", text)
    if match:
        return int(match.group(1))
    return None


def _extract_seat_heating_target_level(text: str) -> int | None:
    if not _mentions_seat_heating(text):
        return None
    if _extract_heating_delta(text) is not None:
        return None
    match = re.search(r"seat heating[^.?!,;]*?\blevel\s*([0-3])\b", text)
    if match:
        return int(match.group(1))
    match = re.search(r"\blevel\s*([0-3])\b[^.?!,;]*?seat heating", text)
    if match:
        return int(match.group(1))
    if "steering wheel" in text:
        return None
    return _extract_heating_level(text)


def _extract_heating_delta(text: str) -> int | None:
    if not _mentions_seat_heating(text) and "seat" not in text:
        return None
    if not re.search(r"\b(increase|raise|bump|boost|up)\b", text):
        return None

    match = re.search(r"\b(?:by|up)\s+([1-3])\s+levels?\b", text)
    if match:
        return int(match.group(1))

    word_numbers = {"one": 1, "two": 2, "three": 3}
    match = re.search(r"\b(?:by|up)\s+(one|two|three)\s+levels?\b", text)
    if match:
        return word_numbers[match.group(1)]

    return None


def _extract_reading_light_position(text: str) -> str | None:
    if "driver rear" in text or "rear driver" in text or "left rear" in text:
        return "DRIVER_REAR"
    if "passenger rear" in text or "rear passenger" in text or "right rear" in text:
        return "PASSENGER_REAR"
    if re.search(r"\bdriver\b", text) and "rear" not in text:
        return "DRIVER"
    if re.search(r"\bpassenger\b", text) and "rear" not in text:
        return "PASSENGER"
    if re.search(r"\b(all|both)\b", text):
        return "ALL"
    return None


def _extract_air_circulation_mode(
    text: str,
) -> Literal["FRESH_AIR", "RECIRCULATION", "AUTO"] | None:
    if "fresh air" in text or "outside air" in text:
        return "FRESH_AIR"
    if "recirculation" in text or "recirculate" in text:
        if any(
            phrase in text
            for phrase in ("don't like", "do not like", "not recirculation")
        ):
            return None
        return "RECIRCULATION"
    if re.search(r"\bauto(?:matic)?\b", text):
        return "AUTO"
    return None


def _extract_defrost_window(
    text: str,
) -> Literal["ALL", "FRONT", "REAR"] | None:
    if "front" in text or "windshield" in text or "windscreen" in text:
        return "FRONT"
    if re.search(r"\brear(?: window)? defrost\b", text) or re.search(
        r"\bdefrost(?: the)? rear\b", text
    ):
        return "REAR"
    if re.search(r"\b(all|both) (?:window )?defrost\b", text) or re.search(
        r"\bdefrost (?:all|both)\b", text
    ):
        return "ALL"
    return None


def _is_affirmative(text: str) -> bool:
    text = text.strip().lower()
    return bool(
        re.search(r"\b(yes|yeah|yep|sure|ok|okay|confirm|confirmed)\b", text)
        or "go ahead" in text
        or "still want" in text
        or "proceed" in text
        or "do it" in text
    )


def _is_negative(text: str) -> bool:
    text = text.strip().lower()
    return bool(
        re.search(r"\b(no|nope|nah|cancel|stop|decline|declined)\b", text)
        or "don't" in text
        or "do not" in text
    )


def _extract_preferred_sunroof_percentage(result: dict[str, Any]) -> int | None:
    text = json.dumps(result, ensure_ascii=False).lower()
    match = re.search(r"default value to open the sunroof is\s*(\d{1,3})\s*%", text)
    if match:
        return _bounded_percentage(match.group(1))

    match = re.search(r"sunroof[^.]*?(\d{1,3})\s*%", text)
    if match:
        value = _bounded_percentage(match.group(1))
        if value is not None and not ("never" in text and value == 100):
            return value

    return None


def _extract_preferred_ambient_color(result: dict[str, Any]) -> str | None:
    text = json.dumps(result, ensure_ascii=False).upper()
    explicit = re.search(
        r"(?:LIGHTCOLOR|AMBIENT[^\"']*LIGHT|AMBIENT[^\"']*COLOR)[^A-Z0-9]+"
        r"(RED|GREEN|BLUE|YELLOW|WHITE|PINK|ORANGE|PURPLE|CYAN|NONE)\b",
        text,
    )
    if explicit:
        return explicit.group(1)

    for color in sorted(AMBIENT_COLORS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(color)}\b", text):
            return color
    return None


def _extract_preferred_fan_speed_level(result: dict[str, Any]) -> int | None:
    text = json.dumps(result, ensure_ascii=False).lower()
    if "fan" not in text:
        return None

    patterns = (
        r"fan[^.?!;]*?level\s*([0-5])",
        r"fan[^.?!;]*?speed[^.?!;]*?([0-5])",
        r"level\s*([0-5])[^.?!;]*?fan",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return _safe_int(match.group(1))
    return None


def _extract_preferred_steering_heating_level(result: dict[str, Any]) -> int | None:
    text = json.dumps(result, ensure_ascii=False).lower()
    if "steering wheel" not in text:
        return None

    match = re.search(r"steering wheel[^.?!;]*?level\s*([0-5])", text)
    if match:
        return _safe_int(match.group(1))

    match = re.search(r"level\s*([0-5])[^.?!;]*?steering wheel", text)
    if match:
        return _safe_int(match.group(1))

    if "medium" in text:
        return 2
    if "low" in text:
        return 1
    if "high" in text:
        return 3
    return None


def _extract_preferred_air_circulation_mode(
    result: dict[str, Any],
) -> Literal["FRESH_AIR", "RECIRCULATION", "AUTO"] | None:
    text = json.dumps(result, ensure_ascii=False).lower()
    if "fresh air" in text or "outside air" in text:
        return "FRESH_AIR"
    if "recirculation" in text or "recirculate" in text:
        return "RECIRCULATION"
    if re.search(r"\bauto(?:matic)?\b", text):
        return "AUTO"
    return None


def _extract_preferred_temperature(result: dict[str, Any]) -> float | None:
    text = json.dumps(result, ensure_ascii=False).lower()
    if "temperature" not in text and "temp" not in text:
        return None
    return _extract_temperature_setting(text)


def _record_defrost_window_position(
    defrost: DefrostFlow, window: str, percentage: int
) -> None:
    if window == "ALL":
        defrost.window_driver_position = percentage
        defrost.window_passenger_position = percentage
        defrost.window_driver_rear_position = percentage
        defrost.window_passenger_rear_position = percentage
    elif window == "DRIVER":
        defrost.window_driver_position = percentage
    elif window == "PASSENGER":
        defrost.window_passenger_position = percentage
    elif window in {"DRIVER_REAR", "RIGHT_REAR"}:
        defrost.window_driver_rear_position = percentage
    elif window in {"PASSENGER_REAR", "LEFT_REAR"}:
        defrost.window_passenger_rear_position = percentage


def _record_window_match_position(
    match: WindowMatchFlow, window: str, percentage: int
) -> None:
    if window == "ALL":
        match.window_driver_position = percentage
        match.window_passenger_position = percentage
        match.window_driver_rear_position = percentage
        match.window_passenger_rear_position = percentage
    elif window == "DRIVER":
        match.window_driver_position = percentage
    elif window == "PASSENGER":
        match.window_passenger_position = percentage
    elif window in {"DRIVER_REAR", "RIGHT_REAR"}:
        match.window_driver_rear_position = percentage
    elif window in {"PASSENGER_REAR", "LEFT_REAR"}:
        match.window_passenger_rear_position = percentage


def _defrost_window_positions(defrost: DefrostFlow) -> tuple[int | None, ...]:
    return (
        defrost.window_driver_position,
        defrost.window_passenger_position,
        defrost.window_driver_rear_position,
        defrost.window_passenger_rear_position,
    )


def _defrost_window_status_unknown(defrost: DefrostFlow) -> bool:
    return any(value is None for value in _defrost_window_positions(defrost))


def _record_air_conditioning_window_position(
    ac: AirConditioningFlow, window: str, percentage: int
) -> None:
    if window == "ALL":
        ac.window_driver_position = percentage
        ac.window_passenger_position = percentage
        ac.window_driver_rear_position = percentage
        ac.window_passenger_rear_position = percentage
    elif window == "DRIVER":
        ac.window_driver_position = percentage
    elif window == "PASSENGER":
        ac.window_passenger_position = percentage
    elif window in {"DRIVER_REAR", "RIGHT_REAR"}:
        ac.window_driver_rear_position = percentage
    elif window in {"PASSENGER_REAR", "LEFT_REAR"}:
        ac.window_passenger_rear_position = percentage


def _air_conditioning_window_positions(
    ac: AirConditioningFlow,
) -> tuple[tuple[str, int | None], ...]:
    return (
        ("DRIVER", ac.window_driver_position),
        ("PASSENGER", ac.window_passenger_position),
        ("DRIVER_REAR", ac.window_driver_rear_position),
        ("PASSENGER_REAR", ac.window_passenger_rear_position),
    )


def _air_conditioning_window_status_unknown(ac: AirConditioningFlow) -> bool:
    return any(
        position is None for _, position in _air_conditioning_window_positions(ac)
    )


def _air_conditioning_open_windows(ac: AirConditioningFlow) -> list[tuple[str, int]]:
    return [
        (window, position)
        for window, position in _air_conditioning_window_positions(ac)
        if position is not None and position > 20
    ]


def _record_recent_window_position(
    state: ControllerState,
    window: str,
    percentage: int,
) -> None:
    if window == "ALL":
        for key in ("DRIVER", "PASSENGER", "DRIVER_REAR", "PASSENGER_REAR"):
            state.recent_window_positions[key] = percentage
        return
    state.recent_window_positions[window] = percentage


def _is_window_ac_status_query(text: str) -> bool:
    return bool(
        "window" in text
        and ("ac" in text or "air conditioning" in text)
        and any(
            phrase in text
            for phrase in (
                "still open",
                "with the ac",
                "with ac",
                "even with",
                "are still",
            )
        )
    )


def _reading_light_actions_for_occupancy(
    seats_occupied: dict[str, bool],
) -> list[tuple[str, bool]]:
    seat_to_light = {
        "driver": "DRIVER",
        "passenger": "PASSENGER",
        "driver_rear": "DRIVER_REAR",
        "passenger_rear": "PASSENGER_REAR",
    }
    occupied_order = ("driver", "passenger", "driver_rear", "passenger_rear")
    unoccupied_order = ("driver_rear", "passenger_rear", "passenger", "driver")

    actions: list[tuple[str, bool]] = []
    for seat in occupied_order:
        if seats_occupied.get(seat) is True:
            actions.append((seat_to_light[seat], True))
    for seat in unoccupied_order:
        if seats_occupied.get(seat) is False:
            actions.append((seat_to_light[seat], False))
    return actions


def _any_defrost_window_open_over(defrost: DefrostFlow, threshold: int) -> bool:
    return any(
        value is not None and value > threshold
        for value in _defrost_window_positions(defrost)
    )


def _airflow_includes_windshield(direction: str | None) -> bool:
    return bool(direction and "WINDSHIELD" in direction)


def _seat_heating_zone_for_occupied_front_seats(
    seats_occupied: dict[str, bool],
) -> str | None:
    driver = seats_occupied.get("driver") is True
    passenger = seats_occupied.get("passenger") is True
    if driver and passenger:
        return "ALL_ZONES"
    if driver:
        return "DRIVER"
    if passenger:
        return "PASSENGER"
    return None


def _unoccupied_front_seat_heating_zone(
    comfort: OccupancyComfortFlow,
) -> str | None:
    driver_unoccupied = comfort.seats_occupied.get("driver") is False
    passenger_unoccupied = comfort.seats_occupied.get("passenger") is False

    driver_on = comfort.seat_heating_driver is None or comfort.seat_heating_driver > 0
    passenger_on = (
        comfort.seat_heating_passenger is None or comfort.seat_heating_passenger > 0
    )

    if driver_unoccupied and passenger_unoccupied and driver_on and passenger_on:
        return "ALL_ZONES"
    if driver_unoccupied and driver_on:
        return "DRIVER"
    if passenger_unoccupied and passenger_on:
        return "PASSENGER"
    return None


def _seat_zone_is_unoccupied_front(
    comfort: OccupancyComfortFlow, seat_zone: Any
) -> bool:
    if seat_zone == "ALL_ZONES":
        return (
            comfort.seats_occupied.get("driver") is False
            and comfort.seats_occupied.get("passenger") is False
        )
    if seat_zone == "DRIVER":
        return comfort.seats_occupied.get("driver") is False
    if seat_zone == "PASSENGER":
        return comfort.seats_occupied.get("passenger") is False
    return False


def _occupied_front_seat_heating_target(comfort: OccupancyComfortFlow) -> int | None:
    if comfort.heating_delta is None:
        return comfort.target_heating_level

    occupied_levels: list[int] = []
    if comfort.seats_occupied.get("driver") is True:
        if comfort.seat_heating_driver is None:
            return None
        occupied_levels.append(comfort.seat_heating_driver)
    if comfort.seats_occupied.get("passenger") is True:
        if comfort.seat_heating_passenger is None:
            return None
        occupied_levels.append(comfort.seat_heating_passenger)

    if not occupied_levels:
        return None

    return max(0, min(3, max(occupied_levels) + comfort.heating_delta))


def _weather_arguments(runtime: RuntimeContext) -> dict[str, Any] | None:
    if (
        runtime.location_id is None
        or runtime.month is None
        or runtime.day is None
        or runtime.hour is None
    ):
        return None
    args: dict[str, Any] = {
        "location_or_poi_id": runtime.location_id,
        "month": runtime.month,
        "day": runtime.day,
        "time_hour_24hformat": runtime.hour,
    }
    if runtime.minute is not None:
        args["time_minutes"] = runtime.minute
    return args


def _email_weather_needs_location_lookup(
    state: ControllerState, email: EmailFlow, tool_index: ToolIndex
) -> bool:
    if email.weather_location_id is not None:
        return False
    if email.weather_location_lookup_attempted:
        return False
    if email.weather_location_name is None:
        return False
    if not tool_index.has("get_location_id_by_location_name"):
        return False

    recent_id = _matching_recent_location_id(
        email.weather_location_name,
        state.recent_location_lookup_name,
        state.recent_location_lookup_id,
    )
    if recent_id is not None:
        email.weather_location_id = recent_id
        return False

    weather_args = tool_index.arg_names("get_weather")
    return not weather_args or "location_or_poi_id" in weather_args


def _email_weather_arguments(
    state: ControllerState, email: EmailFlow, tool_index: ToolIndex
) -> dict[str, Any] | None:
    arg_names = tool_index.arg_names("get_weather")
    include_all = not arg_names
    location_value = (
        email.weather_location_id
        or _matching_recent_location_id(
            email.weather_location_name,
            state.recent_location_lookup_name,
            state.recent_location_lookup_id,
        )
        or email.weather_location_name
        or state.runtime.location_id
    )
    month = state.runtime.month
    day = state.runtime.day
    hour = email.weather_hour if email.weather_hour is not None else state.runtime.hour
    minute = (
        email.weather_minute
        if email.weather_minute is not None
        else state.runtime.minute
    )
    if location_value is None or month is None or day is None or hour is None:
        return None

    args: dict[str, Any] = {}
    if include_all or "location_or_poi_id" in arg_names:
        args["location_or_poi_id"] = location_value
    elif "location" in arg_names:
        args["location"] = email.weather_location_name or location_value
    else:
        return None

    if include_all or "month" in arg_names:
        args["month"] = month
    if include_all or "day" in arg_names:
        args["day"] = day
    if include_all or "time_hour_24hformat" in arg_names:
        args["time_hour_24hformat"] = hour
    elif "hour" in arg_names:
        args["hour"] = hour
    if minute is not None and (include_all or "time_minutes" in arg_names):
        args["time_minutes"] = minute
    return args


def _extract_weather_summary(result: dict[str, Any]) -> tuple[str | None, float | None]:
    slot = _select_weather_slot(result)
    condition = _weather_field(slot, result, ("condition", "weather_condition"))
    temperature = _safe_float(
        _weather_field(
            slot,
            result,
            (
                "temperature",
                "temperature_celsius",
                "temperature_c",
                "temp",
                "temp_c",
            ),
        )
    )
    return condition, temperature


def _select_weather_slot(result: dict[str, Any]) -> dict[str, Any]:
    for key in (
        "current_slot",
        "requested_slot",
        "forecast_slot",
        "weather",
        "forecast",
    ):
        value = result.get(key)
        if isinstance(value, dict):
            return value
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    return item
    return {}


def _weather_field(
    slot: dict[str, Any], result: dict[str, Any], names: tuple[str, ...]
) -> Any:
    for name in names:
        value = slot.get(name)
        if value is not None:
            return value
    for name in names:
        value = result.get(name)
        if value is not None:
            return value
    return None


def _requires_weather_confirmation(condition: str | None) -> bool:
    if not condition:
        return True
    return condition not in SAFE_SUNROOF_WEATHER


def _requires_fog_light_weather_confirmation(condition: str | None) -> bool:
    if not condition:
        return True
    return condition not in SAFE_FOG_LIGHT_WEATHER


def _friendly_weather(condition: str | None) -> str:
    if not condition:
        return "not in a confirmed safe weather condition"
    return condition.replace("_", " ")
