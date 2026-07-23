"""Conversation state flows for the policy-aware controller."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class RuntimeContext:
    location_id: str | None = None
    month: int | None = None
    day: int | None = None
    hour: int | None = None
    minute: int | None = None


@dataclass
class SunroofFlow:
    active: bool = False
    target_percentage: int | None = None
    requested_close: bool = False
    position_checked: bool = False
    weather_checked: bool = False
    preferences_checked: bool = False
    weather_confirmation_requested: bool = False
    weather_confirmed: bool = False
    sunroof_position: float | None = None
    sunshade_position: float | None = None
    weather_condition: str | None = None
    completed: bool = False


@dataclass
class SunshadeFlow:
    active: bool = False
    target_percentage: int | None = None
    completed: bool = False


@dataclass
class WindowFlow:
    active: bool = False
    window: str = "ALL"
    target_percentage: int | None = None
    climate_checked: bool = False
    ac_on: bool | None = None
    ac_confirmation_requested: bool = False
    ac_confirmed: bool = False
    completed: bool = False


@dataclass
class WindowMatchFlow:
    active: bool = False
    reference_window: Literal["FRONT", "PASSENGER_REAR"] = "FRONT"
    all_windows: bool = False
    followup_defrost_window: Literal["ALL", "FRONT", "REAR"] | None = None
    windows_checked: bool = False
    window_driver_position: int | None = None
    window_passenger_position: int | None = None
    window_driver_rear_position: int | None = None
    window_passenger_rear_position: int | None = None
    completed: bool = False


@dataclass
class AmbientLightFlow:
    active: bool = False
    target_color: str | None = None
    on: bool = True
    match_car_color: bool = False
    car_color_checked: bool = False
    preferences_checked: bool = False
    completed: bool = False


@dataclass
class TrunkFlow:
    active: bool = False
    action: Literal["OPEN", "CLOSE"] = "OPEN"
    confirmation_requested: bool = False
    confirmed: bool = False
    completed: bool = False


@dataclass
class AirCirculationFlow:
    active: bool = False
    mode: Literal["FRESH_AIR", "RECIRCULATION", "AUTO"] | None = None
    preferences_checked: bool = False
    combined_with_air_conditioning: bool = False
    combined_fan_level: int | None = None
    combined_window: str | None = None
    combined_window_percentage: int | None = None
    completed: bool = False


@dataclass
class AirConditioningFlow:
    active: bool = False
    on: bool = True
    fan_target_level: int | None = None
    climate_checked: bool = False
    windows_checked: bool = False
    preserve_open_windows: bool = False
    fan_speed: int | None = None
    air_conditioning: bool | None = None
    closed_windows_for_ac: bool = False
    window_driver_position: int | None = None
    window_passenger_position: int | None = None
    window_driver_rear_position: int | None = None
    window_passenger_rear_position: int | None = None
    completed: bool = False


@dataclass
class AirQualityFlow:
    active: bool = False
    climate_checked: bool = False
    fan_speed: int | None = None


@dataclass
class FanSpeedFlow:
    active: bool = False
    level: int | None = None
    query_current: bool = False
    current_checked: bool = False
    current_level: int | None = None
    preferences_checked: bool = False
    completed: bool = False


@dataclass
class CabinTemperatureFlow:
    active: bool = False
    target_temperature: float | None = None
    seat_zone: Literal["ALL_ZONES", "DRIVER", "PASSENGER"] = "ALL_ZONES"
    completed: bool = False


@dataclass
class SteeringWheelHeatingFlow:
    active: bool = False
    level: int | None = None
    preferences_checked: bool = False
    completed: bool = False


@dataclass
class ClimateInspectionFlow:
    active: bool = False
    needs_temperature: bool = True
    needs_seat_heating: bool = False
    temperature_checked: bool = False
    seat_heating_checked: bool = False
    driver_temperature: float | None = None
    passenger_temperature: float | None = None
    temperature_unit: str = "Celsius"
    seat_heating_driver: int | None = None
    seat_heating_passenger: int | None = None
    completed: bool = False


@dataclass
class ClimateEnergyInspectionFlow:
    active: bool = False
    climate_checked: bool = False
    seats_checked: bool = False
    fan_speed: int | None = None
    fan_airflow_direction: str | None = None
    air_conditioning: bool | None = None
    air_circulation: str | None = None
    window_front_defrost: bool | None = None
    window_rear_defrost: bool | None = None
    seats_occupied: dict[str, bool] = field(default_factory=dict)
    completed: bool = False


@dataclass
class OccupancyComfortFlow:
    active: bool = False
    seats_checked: bool = False
    seats_occupied: dict[str, bool] = field(default_factory=dict)
    needs_temperature: bool = False
    target_temperature: float | None = None
    target_temperature_zone: Literal["ALL_ZONES", "DRIVER", "PASSENGER"] = "ALL_ZONES"
    match_driver_to_passenger_temperature: bool = False
    temperature_checked: bool = False
    driver_temperature: float | None = None
    passenger_temperature: float | None = None
    target_heating_level: int | None = None
    heating_delta: int | None = None
    seat_heating_checked: bool = False
    seat_heating_driver: int | None = None
    seat_heating_passenger: int | None = None
    turn_off_unoccupied: bool = False
    query_unoccupied_heating_status: bool = False
    unoccupied_heating_set: bool = False
    preferences_checked: bool = False
    steering_wheel_requested: bool = False
    steering_wheel_level: int | None = None
    temperature_set: bool = False
    seat_heating_set: bool = False
    steering_wheel_set: bool = False
    completed: bool = False


@dataclass
class DriverComfortTemperatureFlow:
    active: bool = False
    passenger_heating_required: bool = False
    passenger_heating_off: bool = False
    target_temperature: float | None = None
    preferences_checked: bool = False
    temperature_set: bool = False
    completed: bool = False


@dataclass
class PassengerComfortMatchFlow:
    active: bool = False
    target_temperature: float | None = None
    temperature_set: bool = False
    seat_heating_checked: bool = False
    driver_heating_level: int | None = None
    passenger_heating_level: int | None = None
    passenger_heating_set: bool = False
    steering_wheel_set: bool = False
    completed: bool = False


@dataclass
class ReadingLightFlow:
    active: bool = False
    position: str | None = None
    on: bool = True
    completed: bool = False


@dataclass
class ReadingLightOccupancyFlow:
    active: bool = False
    seats_checked: bool = False
    seats_occupied: dict[str, bool] = field(default_factory=dict)
    pending_actions: list[tuple[str, bool]] = field(default_factory=list)
    completed: bool = False


@dataclass
class NavigationFlow:
    active: bool = False
    mode: (
        Literal[
            "set_new",
            "replace_final_destination",
            "delete_final_destination",
            "delete_waypoint",
            "replace_one_waypoint",
        ]
        | None
    ) = None
    destination_name: str | None = None
    destination_id: str | None = None
    route_start_id: str | None = None
    route_preference: Literal["fastest", "shortest"] | None = None
    toll_avoidance_tolerance_minutes: int | None = None
    planning_only: bool = False
    routes_checked: bool = False
    routes: list[dict[str, Any]] = field(default_factory=list)
    route_choice_requested: bool = False
    route_detail_requested: bool = False
    selected_route_index: int | None = None
    current_navigation_checked: bool = False
    needs_current_navigation: bool = False
    navigation_active: bool | None = None
    waypoints_id: list[str] = field(default_factory=list)
    routes_to_final_destination_id: list[str] = field(default_factory=list)
    waypoint_details: list[dict[str, Any]] = field(default_factory=list)
    waypoint_name: str | None = None
    waypoint_id: str | None = None
    new_waypoint_name: str | None = None
    new_waypoint_id: str | None = None
    route_lookup: (
        Literal[
            "destination",
            "delete_without_waypoint",
            "to_new_waypoint",
            "from_new_waypoint",
        ]
        | None
    ) = None
    route_without_waypoint_checked: bool = False
    routes_without_waypoint: list[dict[str, Any]] = field(default_factory=list)
    route_to_new_waypoint_checked: bool = False
    routes_to_new_waypoint: list[dict[str, Any]] = field(default_factory=list)
    route_from_new_waypoint_checked: bool = False
    routes_from_new_waypoint: list[dict[str, Any]] = field(default_factory=list)
    completion_message: str | None = None
    completed: bool = False
    failure_message: str | None = None


@dataclass
class MultiStopNavigationFlow:
    active: bool = False
    first_destination_name: str | None = None
    first_destination_id: str | None = None
    final_destination_name: str | None = None
    final_destination_id: str | None = None
    route_preference: Literal["fastest", "shortest"] | None = None
    first_routes_checked: bool = False
    first_routes: list[dict[str, Any]] = field(default_factory=list)
    final_routes_checked: bool = False
    final_routes: list[dict[str, Any]] = field(default_factory=list)
    completion_message: str | None = None
    completed: bool = False
    failure_message: str | None = None


@dataclass
class POIFlow:
    active: bool = False
    category: str = "restaurants"
    location_name: str | None = None
    location_id: str | None = None
    search_along_route: bool = False
    route_id: str | None = None
    route_start_id: str | None = None
    route_end_id: str | None = None
    route_prefix_ids: list[str] = field(default_factory=list)
    route_prefix_selected_routes: list[dict[str, Any]] = field(default_factory=list)
    route_prefix_considered_routes: list[dict[str, Any]] = field(default_factory=list)
    route_lookup: Literal["to_poi", "from_poi"] | None = None
    current_navigation_checked: bool = False
    required_open_at_minutes: int | None = None
    pois_checked: bool = False
    pois: list[dict[str, Any]] = field(default_factory=list)
    selected_poi_id: str | None = None
    selected_poi_name: str | None = None
    routes_checked: bool = False
    routes: list[dict[str, Any]] = field(default_factory=list)
    routes_from_poi_checked: bool = False
    routes_from_poi: list[dict[str, Any]] = field(default_factory=list)
    route_choice_requested: bool = False
    route_preference: Literal["fastest", "shortest"] | None = None
    selected_route_index: int | None = None
    replace_final_destination: bool = False
    do_not_set_navigation: bool = False
    defer_navigation_setup: bool = False
    completion_message: str | None = None
    completed: bool = False
    failure_message: str | None = None


@dataclass
class RouteEnergyFlow:
    active: bool = False
    destination_name: str | None = None
    destination_id: str | None = None
    route_preference: Literal["fastest", "shortest"] | None = None
    station_route_preference: Literal["fastest", "shortest"] | None = None
    route_start_id: str | None = None
    routes_checked: bool = False
    routes: list[dict[str, Any]] = field(default_factory=list)
    route_choice_requested: bool = False
    route_detail_requested: bool = False
    selected_route_index: int | None = None
    final_route_via: str | None = None
    route_lookup: Literal["destination", "setup_to_poi", "setup_from_poi"] | None = None
    setup_to_poi_checked: bool = False
    routes_to_poi: list[dict[str, Any]] = field(default_factory=list)
    setup_from_poi_checked: bool = False
    routes_from_poi: list[dict[str, Any]] = field(default_factory=list)
    current_navigation_checked: bool = False
    needs_current_navigation: bool = False
    navigation_active: bool | None = None
    waypoints_id: list[str] = field(default_factory=list)
    routes_to_final_destination_id: list[str] = field(default_factory=list)
    route_details: list[dict[str, Any]] = field(default_factory=list)
    route_id: str | None = None
    route_selection: Literal["first_segment", "last_segment", "planned_route"] = (
        "first_segment"
    )
    needs_charging_status: bool = False
    charging_status_checked: bool = False
    state_of_charge: float | None = None
    remaining_range_km: float | None = None
    wants_range: bool = False
    wants_current_range: bool = False
    wants_battery_sufficiency: bool = False
    wants_stop_count: bool = False
    wants_navigation_setup: bool = False
    initial_soc: float | None = None
    final_soc: float | None = None
    distance_checked: bool = False
    distance_km: float | None = None
    wants_charger_search: bool = False
    search_mode: Literal["nearby", "along_route"] | None = None
    at_kilometer: float | None = None
    search_kilometers: list[int] = field(default_factory=list)
    search_kilometer_index: int = 0
    filters: list[str] = field(default_factory=list)
    poi_search_category: str | None = None
    pois_checked: bool = False
    pois: list[dict[str, Any]] = field(default_factory=list)
    selected_poi_id: str | None = None
    selected_poi_name: str | None = None
    selected_plug_id: str | None = None
    selected_phone_number: str | None = None
    companion_poi_category: str | None = None
    companion_poi_filters: list[str] = field(default_factory=list)
    companion_pois_checked: bool = False
    companion_pois: list[dict[str, Any]] = field(default_factory=list)
    arrival_window_start_minutes: int | None = None
    arrival_window_end_minutes: int | None = None
    wants_charging_time: bool = False
    target_soc: float | None = None
    start_soc_for_charging: float | None = None
    charging_time_checked: bool = False
    charging_minutes: float | None = None
    wants_call: bool = False
    call_completed: bool = False
    completion_message: str | None = None
    completed: bool = False
    failure_message: str | None = None


@dataclass
class EmailFlow:
    active: bool = False
    mode: Literal["meeting_delay", "meeting_attendees", "share_contact"] | None = None
    recipient_name: str | None = None
    recipient_first_name: str | None = None
    recipient_last_name: str | None = None
    recipient_contact_id: str | None = None
    recipient_email: str | None = None
    subject_name: str | None = None
    subject_first_name: str | None = None
    subject_last_name: str | None = None
    subject_contact_id: str | None = None
    subject_email: str | None = None
    subject_phone: str | None = None
    attendee_contact_ids: list[str] = field(default_factory=list)
    attendee_emails_by_id: dict[str, str] = field(default_factory=dict)
    calendar_checked: bool = False
    meeting_topic: str | None = None
    meeting_started: bool | None = None
    meeting_start_hour: int | None = None
    meeting_start_minute: int | None = None
    meeting_location_name: str | None = None
    user_claimed_late: bool = False
    weather_requested: bool = False
    weather_checked: bool = False
    weather_location_name: str | None = None
    weather_location_id: str | None = None
    weather_location_lookup_attempted: bool = False
    weather_hour: int | None = None
    weather_minute: int | None = None
    weather_condition: str | None = None
    weather_temperature: float | None = None
    pending_lookup_role: Literal["recipient", "subject"] | None = None
    pending_contact_matches: dict[str, str] = field(default_factory=dict)
    preferences_checked: bool = False
    content_message: str | None = None
    confirmation_requested: bool = False
    confirmed: bool = False
    completed: bool = False
    failure_message: str | None = None


@dataclass
class HighBeamFlow:
    active: bool = False
    on: bool = True
    exterior_lights_checked: bool = False
    fog_lights_on: bool | None = None
    confirmation_requested: bool = False
    confirmed: bool = False
    declined: bool = False
    completed: bool = False


@dataclass
class FogLightFlow:
    active: bool = False
    on: bool = True
    weather_checked: bool = False
    weather_condition: str | None = None
    weather_confirmation_requested: bool = False
    weather_confirmed: bool = False
    exterior_lights_checked: bool = False
    fog_lights_on: bool | None = None
    low_beams_on: bool | None = None
    high_beams_on: bool | None = None
    high_beam_confirmation_requested: bool = False
    high_beam_confirmed: bool = False
    declined: bool = False
    completed: bool = False


@dataclass
class DefrostFlow:
    active: bool = False
    on: bool = True
    defrost_window: Literal["ALL", "FRONT", "REAR"] | None = None
    climate_checked: bool = False
    windows_checked: bool = False
    fan_speed: int | None = None
    fan_airflow_direction: str | None = None
    air_conditioning: bool | None = None
    window_driver_position: int | None = None
    window_passenger_position: int | None = None
    window_driver_rear_position: int | None = None
    window_passenger_rear_position: int | None = None
    completed: bool = False


@dataclass
class ControllerState:
    runtime: RuntimeContext = field(default_factory=RuntimeContext)
    sunroof: SunroofFlow = field(default_factory=SunroofFlow)
    sunshade: SunshadeFlow = field(default_factory=SunshadeFlow)
    window: WindowFlow = field(default_factory=WindowFlow)
    window_match: WindowMatchFlow = field(default_factory=WindowMatchFlow)
    ambient_light: AmbientLightFlow = field(default_factory=AmbientLightFlow)
    trunk: TrunkFlow = field(default_factory=TrunkFlow)
    air_circulation: AirCirculationFlow = field(default_factory=AirCirculationFlow)
    air_conditioning: AirConditioningFlow = field(default_factory=AirConditioningFlow)
    air_quality: AirQualityFlow = field(default_factory=AirQualityFlow)
    fan_speed: FanSpeedFlow = field(default_factory=FanSpeedFlow)
    cabin_temperature: CabinTemperatureFlow = field(
        default_factory=CabinTemperatureFlow
    )
    steering_wheel_heating: SteeringWheelHeatingFlow = field(
        default_factory=SteeringWheelHeatingFlow
    )
    climate_inspection: ClimateInspectionFlow = field(
        default_factory=ClimateInspectionFlow
    )
    climate_energy_inspection: ClimateEnergyInspectionFlow = field(
        default_factory=ClimateEnergyInspectionFlow
    )
    occupancy_comfort: OccupancyComfortFlow = field(
        default_factory=OccupancyComfortFlow
    )
    driver_comfort_temperature: DriverComfortTemperatureFlow = field(
        default_factory=DriverComfortTemperatureFlow
    )
    passenger_comfort_match: PassengerComfortMatchFlow = field(
        default_factory=PassengerComfortMatchFlow
    )
    reading_light: ReadingLightFlow = field(default_factory=ReadingLightFlow)
    reading_light_occupancy: ReadingLightOccupancyFlow = field(
        default_factory=ReadingLightOccupancyFlow
    )
    navigation: NavigationFlow = field(default_factory=NavigationFlow)
    poi: POIFlow = field(default_factory=POIFlow)
    route_energy: RouteEnergyFlow = field(default_factory=RouteEnergyFlow)
    email: EmailFlow = field(default_factory=EmailFlow)
    high_beam: HighBeamFlow = field(default_factory=HighBeamFlow)
    fog_lights: FogLightFlow = field(default_factory=FogLightFlow)
    defrost: DefrostFlow = field(default_factory=DefrostFlow)
    last_user_text: str = ""
    recent_meeting_topic: str | None = None
    recent_calendar_meetings: list[dict[str, Any]] = field(default_factory=list)
    business_email_extra_recipients: list[str] = field(default_factory=list)
    pending_location_lookup_name: str | None = None
    recent_location_lookup_name: str | None = None
    recent_location_lookup_id: str | None = None
    recent_navigation_route_ids: list[str] = field(default_factory=list)
    recent_navigation_waypoints: list[str] = field(default_factory=list)
    planned_navigation_route_ids: list[str] = field(default_factory=list)
    planned_navigation_waypoints: list[str] = field(default_factory=list)
    planned_navigation_selected_routes: list[dict[str, Any]] = field(
        default_factory=list
    )
    planned_navigation_considered_routes: list[dict[str, Any]] = field(
        default_factory=list
    )
    planned_navigation_arrival_minutes: int | None = None
    pending_poi_after_navigation: POIFlow | None = None
    pending_set_navigation_route_ids: list[str] = field(default_factory=list)
    multi_stop_navigation: MultiStopNavigationFlow = field(
        default_factory=MultiStopNavigationFlow
    )
    recent_driver_temperature: float | None = None
    recent_passenger_temperature: float | None = None
    recent_charging_pois: list[dict[str, Any]] = field(default_factory=list)
    recent_charging_route_id: str | None = None
    recent_charging_at_kilometer: float | None = None
    recent_selected_charging_poi_id: str | None = None
    recent_selected_charging_poi_name: str | None = None
    recent_selected_charging_plug_id: str | None = None
    recent_selected_charging_phone_number: str | None = None
    recent_window_positions: dict[str, int] = field(default_factory=dict)
    recent_air_conditioning: bool | None = None
    recent_air_circulation: str | None = None
    pending_direct_response: str | None = None
