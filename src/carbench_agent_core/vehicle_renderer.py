"""Vehicle-domain user-visible text helpers."""

from __future__ import annotations

from typing import Literal

from .flows import (
    AirConditioningFlow,
    ClimateEnergyInspectionFlow,
    ClimateInspectionFlow,
    ControllerState,
    OccupancyComfortFlow,
)


def _format_air_conditioning_completion(ac: AirConditioningFlow) -> str:
    completed = []
    if ac.closed_windows_for_ac:
        completed.append("closed the open windows")
    if ac.fan_target_level is not None and ac.fan_speed == ac.fan_target_level:
        completed.append(f"set the fan speed to level {ac.fan_target_level}")
    completed.append("turned on the air conditioning")

    if len(completed) == 1:
        summary = completed[0]
    elif len(completed) == 2:
        summary = f"{completed[0]} and {completed[1]}"
    else:
        summary = f"{', '.join(completed[:-1])}, and {completed[-1]}"
    return f"Done, I {summary}."


def _format_combined_window_completion(
    window: str | None,
    percentage: int | None,
) -> str | None:
    if window is None or percentage is None:
        return None
    return f"opened {friendly_window_name(window)} to {_format_percentage(percentage)}"


def _format_recent_window_ac_status(state: ControllerState) -> str | None:
    positions = state.recent_window_positions
    if not positions:
        return None
    known_positions = [
        positions.get(key)
        for key in ("DRIVER", "PASSENGER", "DRIVER_REAR", "PASSENGER_REAR")
    ]
    if any(position is None for position in known_positions):
        return None
    unique_positions = set(known_positions)
    if len(unique_positions) != 1:
        return None
    window_position = unique_positions.pop()
    ac = state.recent_air_conditioning
    if ac is None:
        return None
    ac_text = "on" if ac else "off"
    air_mode = (
        f" and air circulation is set to {_friendly_air_mode(state.recent_air_circulation)}"
        if state.recent_air_circulation in {"FRESH_AIR", "RECIRCULATION", "AUTO"}
        else ""
    )
    return (
        f"Yes, all windows are open to {_format_percentage(window_position)} "
        f"and the air conditioning is {ac_text}{air_mode}."
    )


def _join_completed_actions(actions: list[str]) -> str:
    if not actions:
        return "completed that"
    if len(actions) == 1:
        return actions[0]
    if len(actions) == 2:
        return f"{actions[0]} and {actions[1]}"
    return f"{', '.join(actions[:-1])}, and {actions[-1]}"


def _friendly_defrost_window(
    defrost_window: Literal["ALL", "FRONT", "REAR"] | None,
) -> str:
    return {
        "ALL": "all-window",
        "FRONT": "front-window",
        "REAR": "rear-window",
        None: "window",
    }[defrost_window]


def friendly_window_name(window: str) -> str:
    return {
        "ALL": "all windows",
        "DRIVER": "the driver window",
        "PASSENGER": "the passenger window",
        "DRIVER_REAR": "the driver rear window",
        "PASSENGER_REAR": "the passenger rear window",
        "RIGHT_REAR": "the right rear window",
        "LEFT_REAR": "the left rear window",
    }.get(window, "the window")


def _friendly_air_mode(
    mode: Literal["FRESH_AIR", "RECIRCULATION", "AUTO"] | None,
) -> str:
    return {
        "FRESH_AIR": "fresh air mode",
        "RECIRCULATION": "recirculation mode",
        "AUTO": "auto mode",
        None: "the requested mode",
    }[mode]


def _friendly_reading_light_position(position: str | None) -> str:
    return {
        "ALL": "all",
        "DRIVER": "the driver",
        "PASSENGER": "the passenger",
        "DRIVER_REAR": "the driver rear",
        "PASSENGER_REAR": "the passenger rear",
        None: "the requested",
    }.get(position, "the requested")


def _format_climate_energy_inspection_summary(
    inspection: ClimateEnergyInspectionFlow,
) -> str:
    climate_parts = []
    if inspection.fan_speed is not None:
        climate_parts.append(f"fan speed {inspection.fan_speed}")
    if inspection.fan_airflow_direction:
        airflow = inspection.fan_airflow_direction.lower().replace("_", " ")
        climate_parts.append(f"airflow {airflow}")
    if inspection.air_conditioning is not None:
        climate_parts.append(
            "air conditioning on"
            if inspection.air_conditioning
            else "air conditioning off"
        )
    if inspection.air_circulation:
        circulation = inspection.air_circulation.lower().replace("_", " ")
        climate_parts.append(f"air circulation {circulation}")
    defrost_parts = []
    if inspection.window_front_defrost is not None:
        defrost_parts.append(
            "front defrost on"
            if inspection.window_front_defrost
            else "front defrost off"
        )
    if inspection.window_rear_defrost is not None:
        defrost_parts.append(
            "rear defrost on"
            if inspection.window_rear_defrost
            else "rear defrost off"
        )
    climate_summary = ", ".join([*climate_parts, *defrost_parts])
    if not climate_summary:
        climate_summary = "current climate settings are unavailable"

    occupancy_summary = _format_seat_occupancy_summary(inspection.seats_occupied)
    empty_front = _format_empty_front_seats(inspection.seats_occupied)
    next_step = (
        f"The next energy-efficiency check is whether seat heating is on for "
        f"the empty {empty_front}."
        if empty_front is not None
        else "There are no empty controllable front seats to check for wasted seat heating."
    )
    return f"Current climate: {climate_summary}. Occupancy: {occupancy_summary}. {next_step}"


def _format_seat_occupancy_summary(seats_occupied: dict[str, bool]) -> str:
    if not seats_occupied:
        return "unknown"
    labels = {
        "driver": "driver",
        "passenger": "passenger",
        "driver_rear": "driver rear",
        "passenger_rear": "passenger rear",
    }
    occupied = [
        label
        for seat, label in labels.items()
        if seats_occupied.get(seat) is True
    ]
    empty = [
        label
        for seat, label in labels.items()
        if seats_occupied.get(seat) is False
    ]
    if occupied and empty:
        return f"{', '.join(occupied)} occupied; {', '.join(empty)} empty"
    if occupied:
        return f"{', '.join(occupied)} occupied"
    if empty:
        return f"{', '.join(empty)} empty"
    return "unknown"


def _format_empty_front_seats(seats_occupied: dict[str, bool]) -> str | None:
    empty = []
    if seats_occupied.get("driver") is False:
        empty.append("driver seat")
    if seats_occupied.get("passenger") is False:
        empty.append("passenger seat")
    if not empty:
        return None
    return " and ".join(empty)


def _format_climate_inspection_summary(climate: ClimateInspectionFlow) -> str:
    parts = []
    if climate.needs_temperature:
        parts.append(
            "Current temperatures are "
            f"driver {_format_temperature(climate.driver_temperature, climate.temperature_unit)} "
            f"and passenger {_format_temperature(climate.passenger_temperature, climate.temperature_unit)}"
        )
    if climate.needs_seat_heating:
        parts.append(
            "seat heating is "
            f"driver level {climate.seat_heating_driver} "
            f"and passenger level {climate.seat_heating_passenger}"
        )
    return f"{'; '.join(parts)}."


def _format_temperature(value: float | None, unit: str) -> str:
    if value is None:
        return "unknown"
    unit_text = "C" if unit.lower().startswith("celsius") else unit
    if float(value).is_integer():
        number = str(int(value))
    else:
        number = f"{value:.1f}"
    return f"{number} {unit_text}"


def _format_temperature_target(value: float | None) -> str:
    if value is None:
        return "the requested temperature"
    if float(value).is_integer():
        return f"{int(value)} C"
    return f"{value:.1f} C"


def _format_child_seat_heating_status(
    comfort: OccupancyComfortFlow,
) -> str | None:
    if not comfort.seats_occupied:
        return None
    if (
        comfort.seats_occupied.get("driver_rear") is True
        or comfort.seats_occupied.get("passenger_rear") is True
    ):
        return (
            "No, the occupied rear seat does not have an available seat-heating "
            "control, so seat heating is not activated there."
        )
    if comfort.seats_occupied.get("passenger") is True:
        if comfort.target_heating_level is None:
            return None
        return (
            "Yes, passenger seat heating is set to "
            f"level {comfort.target_heating_level}."
        )
    return None


def _format_unoccupied_seat_heating_status(comfort: OccupancyComfortFlow) -> str:
    heated = []
    if comfort.seats_occupied.get("driver") is False and (
        comfort.seat_heating_driver or 0
    ) > 0:
        heated.append(f"driver seat at level {comfort.seat_heating_driver}")
    if comfort.seats_occupied.get("passenger") is False and (
        comfort.seat_heating_passenger or 0
    ) > 0:
        heated.append(f"passenger seat at level {comfort.seat_heating_passenger}")

    if heated:
        return f"Yes, seat heating is on for the empty {' and '.join(heated)}."
    return "No, seat heating is not on for any empty front seat I can control."


def _format_percentage(value: float | int | None) -> str:
    if value is None:
        return "the requested position"
    if float(value).is_integer():
        return f"{int(value)}%"
    return f"{value}%"
