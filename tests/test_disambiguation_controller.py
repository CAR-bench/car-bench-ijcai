import json
from typing import Any

from carbench_agent_core import PolicyAwareController


def fake_tool(
    name: str,
    properties: dict[str, Any] | None = None,
    required: list[str] | None = None,
) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "",
            "parameters": {
                "type": "object",
                "properties": properties or {},
                "required": required or [],
            },
        },
    }


def tool_result(tool_name: str, result: dict) -> dict:
    return {
        "tool_name": tool_name,
        "content": json.dumps({"status": "SUCCESS", "result": result}),
    }


def messages() -> list[dict]:
    return [{"role": "system", "content": ""}]


def climate_tools(*, include_fan_level: bool = True) -> list[dict]:
    fan_properties = {"level": {"type": "integer"}} if include_fan_level else {}
    fan_required = ["level"] if include_fan_level else []
    return [
        fake_tool("get_climate_settings"),
        fake_tool("get_vehicle_window_positions"),
        fake_tool(
            "open_close_window",
            {
                "window": {"type": "string"},
                "percentage": {"type": "number"},
            },
            ["window", "percentage"],
        ),
        fake_tool("set_fan_speed", fan_properties, fan_required),
        fake_tool("set_air_conditioning", {"on": {"type": "boolean"}}, ["on"]),
    ]


def climate_inspection_tools() -> list[dict]:
    return [
        fake_tool("get_temperature_inside_car"),
        fake_tool("get_seat_heating_level"),
    ]


def occupancy_comfort_tools() -> list[dict]:
    return [
        fake_tool("get_seats_occupancy"),
        fake_tool(
            "set_climate_temperature",
            {
                "seat_zone": {"type": "string"},
                "temperature": {"type": "number"},
            },
            ["seat_zone", "temperature"],
        ),
        fake_tool(
            "set_seat_heating",
            {
                "seat_zone": {"type": "string"},
                "level": {"type": "integer"},
            },
            ["seat_zone", "level"],
        ),
    ]


def comprehensive_comfort_tools() -> list[dict]:
    return [
        fake_tool("get_user_preferences"),
        fake_tool("get_seat_heating_level"),
        *occupancy_comfort_tools(),
        fake_tool(
            "set_steering_wheel_heating",
            {"level": {"type": "integer"}},
            ["level"],
        ),
    ]


def driver_comfort_tools() -> list[dict]:
    return [
        fake_tool("get_user_preferences"),
        fake_tool(
            "set_climate_temperature",
            {
                "seat_zone": {"type": "string"},
                "temperature": {"type": "number"},
            },
            ["seat_zone", "temperature"],
        ),
        fake_tool(
            "set_seat_heating",
            {
                "seat_zone": {"type": "string"},
                "level": {"type": "integer"},
            },
            ["seat_zone", "level"],
        ),
    ]


def passenger_comfort_match_tools(*, include_seat_zone: bool = True) -> list[dict]:
    seat_properties = {"level": {"type": "integer"}}
    seat_required = ["level"]
    if include_seat_zone:
        seat_properties["seat_zone"] = {"type": "string"}
        seat_required.append("seat_zone")

    return [
        fake_tool(
            "set_climate_temperature",
            {
                "seat_zone": {"type": "string"},
                "temperature": {"type": "number"},
            },
            ["seat_zone", "temperature"],
        ),
        fake_tool("get_seat_heating_level"),
        fake_tool("set_seat_heating", seat_properties, seat_required),
        fake_tool(
            "set_steering_wheel_heating",
            {"level": {"type": "integer"}},
            ["level"],
        ),
    ]


def climate_result(*, fan_speed: int = 0, air_conditioning: bool = False) -> dict:
    return {
        "fan_speed": fan_speed,
        "fan_airflow_direction": "HEAD",
        "air_conditioning": air_conditioning,
        "air_circulation": "FRESH_AIR",
    }


def window_positions(
    *,
    driver: int = 0,
    passenger: int = 0,
    driver_rear: int = 0,
    passenger_rear: int = 0,
) -> dict:
    return {
        "window_driver_position": driver,
        "window_passenger_position": passenger,
        "window_driver_rear_position": driver_rear,
        "window_passenger_rear_position": passenger_rear,
    }


def test_climate_inspection_checks_temperature_and_seat_heating() -> None:
    controller = PolicyAwareController()

    action = controller.decide(
        context_id="ctx-climate-inspection",
        messages=messages(),
        tools=climate_inspection_tools(),
        latest_user_text=(
            "Can you tell me what the current climate settings are, like the "
            "temperatures and seat heating levels for both sides?"
        ),
    )
    assert action is not None
    assert action.tool_calls == [
        {"tool_name": "get_temperature_inside_car", "arguments": {}}
    ]

    action = controller.decide(
        context_id="ctx-climate-inspection",
        messages=messages(),
        tools=climate_inspection_tools(),
        latest_tool_results=[
            tool_result(
                "get_temperature_inside_car",
                {
                    "climate_temperature_driver": 18,
                    "climate_temperature_passenger": 23,
                    "temperature_unit": "Celsius",
                },
            )
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {"tool_name": "get_seat_heating_level", "arguments": {}}
    ]

    action = controller.decide(
        context_id="ctx-climate-inspection",
        messages=messages(),
        tools=climate_inspection_tools(),
        latest_tool_results=[
            tool_result(
                "get_seat_heating_level",
                {"seat_heating_driver": 3, "seat_heating_passenger": 3},
            )
        ],
    )
    assert action is not None
    assert action.action == "respond"
    assert "driver 18 C" in action.content
    assert "passenger 23 C" in action.content
    assert "driver level 3" in action.content
    assert "passenger level 3" in action.content


def test_temperature_change_request_is_not_climate_inspection() -> None:
    controller = PolicyAwareController()

    action = controller.decide(
        context_id="ctx-climate-change-not-inspection",
        messages=messages(),
        tools=climate_inspection_tools(),
        latest_user_text=(
            "Raise the driver zone temperature by 4 degrees Celsius. "
            "It's currently 18 C."
        ),
    )
    assert action is None


def test_climate_energy_inspection_checks_climate_and_occupancy() -> None:
    controller = PolicyAwareController()
    tools = [fake_tool("get_climate_settings"), fake_tool("get_seats_occupancy")]

    action = controller.decide(
        context_id="ctx-climate-energy-inspection",
        messages=messages(),
        tools=tools,
        latest_user_text=(
            "Can you tell me what the current climate settings are and "
            "who's in the car? I'm trying to save a bit of energy."
        ),
    )
    assert action is not None
    assert action.tool_calls == [{"tool_name": "get_climate_settings", "arguments": {}}]

    action = controller.decide(
        context_id="ctx-climate-energy-inspection",
        messages=messages(),
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_climate_settings",
                {
                    "fan_speed": 0,
                    "fan_airflow_direction": "WINDSHIELD_FEET",
                    "air_conditioning": False,
                    "air_circulation": "AUTO",
                    "window_front_defrost": False,
                    "window_rear_defrost": False,
                },
            )
        ],
    )
    assert action is not None
    assert action.tool_calls == [{"tool_name": "get_seats_occupancy", "arguments": {}}]

    action = controller.decide(
        context_id="ctx-climate-energy-inspection",
        messages=messages(),
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_seats_occupancy",
                {
                    "seats_occupied": {
                        "driver": True,
                        "passenger": False,
                        "driver_rear": False,
                        "passenger_rear": False,
                    }
                },
            )
        ],
    )
    assert action is not None
    assert action.action == "respond"
    assert "passenger seat" in action.content.lower()
    assert "seat heating" in action.content.lower()
    assert "next energy-efficiency check" in action.content.lower()


def test_climate_occupancy_status_query_uses_energy_inspection_flow() -> None:
    controller = PolicyAwareController()
    tools = [fake_tool("get_climate_settings"), fake_tool("get_seats_occupancy")]

    action = controller.decide(
        context_id="ctx-climate-occupancy-status",
        messages=messages(),
        tools=tools,
        latest_user_text=(
            "Can you tell me what the climate settings are right now and "
            "who's in the car?"
        ),
    )
    assert action is not None
    assert action.tool_calls == [{"tool_name": "get_climate_settings", "arguments": {}}]

    action = controller.decide(
        context_id="ctx-climate-occupancy-status",
        messages=messages(),
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_climate_settings",
                {
                    "fan_speed": 0,
                    "fan_airflow_direction": "WINDSHIELD_FEET",
                    "air_conditioning": False,
                    "air_circulation": "AUTO",
                    "window_front_defrost": False,
                    "window_rear_defrost": False,
                },
            )
        ],
    )
    assert action is not None
    assert action.tool_calls == [{"tool_name": "get_seats_occupancy", "arguments": {}}]

    action = controller.decide(
        context_id="ctx-climate-occupancy-status",
        messages=messages(),
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_seats_occupancy",
                {
                    "seats_occupied": {
                        "driver": True,
                        "passenger": False,
                        "driver_rear": False,
                        "passenger_rear": False,
                    }
                },
            )
        ],
    )
    assert action is not None
    assert action.action == "respond"
    assert "seat heating" in action.content.lower()


def test_occupancy_comfort_asks_for_settings_then_sets_occupied_seats() -> None:
    controller = PolicyAwareController()
    tools = occupancy_comfort_tools()

    action = controller.decide(
        context_id="ctx-occupancy-comfort",
        messages=messages(),
        tools=tools,
        latest_user_text=(
            "It's cold in here. Warm up the car efficiently, only for the "
            "seats that are occupied."
        ),
    )
    assert action is not None
    assert action.tool_calls == [{"tool_name": "get_seats_occupancy", "arguments": {}}]

    action = controller.decide(
        context_id="ctx-occupancy-comfort",
        messages=messages(),
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_seats_occupancy",
                {"seats_occupied": {"driver": True, "passenger": True}},
            )
        ],
    )
    assert action is not None
    assert action.action == "respond"
    assert "temperature" in action.content.lower()

    action = controller.decide(
        context_id="ctx-occupancy-comfort",
        messages=messages(),
        tools=tools,
        latest_user_text="Use 22 degrees for temperature and level 3 for seat heating.",
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "set_climate_temperature",
            "arguments": {"seat_zone": "ALL_ZONES", "temperature": 22.0},
        }
    ]

    action = controller.decide(
        context_id="ctx-occupancy-comfort",
        messages=messages(),
        tools=tools,
        latest_tool_results=[
            tool_result(
                "set_climate_temperature",
                {"seat_zone": "ALL_ZONES", "temperature": 22},
            )
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "set_seat_heating",
            "arguments": {"seat_zone": "ALL_ZONES", "level": 3},
        }
    ]


def test_comprehensive_occupancy_warming_uses_preferences_and_current_heating() -> None:
    controller = PolicyAwareController()
    tools = comprehensive_comfort_tools()

    action = controller.decide(
        context_id="ctx-comprehensive-warming",
        messages=messages(),
        tools=tools,
        latest_user_text=(
            "It's very cold. Warm up the car comprehensively: set the cabin "
            "to my usual comfortable temperature, increase the seat heating "
            "by two levels where people are sitting, and turn on steering "
            "wheel heating."
        ),
    )
    assert action is not None
    assert action.tool_calls == [
        {"tool_name": "get_seat_heating_level", "arguments": {}}
    ]

    action = controller.decide(
        context_id="ctx-comprehensive-warming",
        messages=messages(),
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_seat_heating_level",
                {"seat_heating_driver": 0, "seat_heating_passenger": 0},
            )
        ],
    )
    assert action is not None
    assert action.tool_calls == [{"tool_name": "get_seats_occupancy", "arguments": {}}]

    action = controller.decide(
        context_id="ctx-comprehensive-warming",
        messages=messages(),
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_seats_occupancy",
                {"seats_occupied": {"driver": True, "passenger": True}},
            )
        ],
    )
    assert action is not None
    assert action.tool_calls[0]["tool_name"] == "get_user_preferences"

    action = controller.decide(
        context_id="ctx-comprehensive-warming",
        messages=messages(),
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_user_preferences",
                {
                    "vehicle_settings": {
                        "climate_control": [
                            "default comfortable temperature is 22 degree"
                        ]
                    }
                },
            )
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "set_climate_temperature",
            "arguments": {"seat_zone": "ALL_ZONES", "temperature": 22.0},
        }
    ]

    action = controller.decide(
        context_id="ctx-comprehensive-warming",
        messages=messages(),
        tools=tools,
        latest_tool_results=[
            tool_result(
                "set_climate_temperature",
                {"seat_zone": "ALL_ZONES", "temperature": 22},
            )
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "set_seat_heating",
            "arguments": {"seat_zone": "ALL_ZONES", "level": 2},
        }
    ]

    action = controller.decide(
        context_id="ctx-comprehensive-warming",
        messages=messages(),
        tools=tools,
        latest_tool_results=[
            tool_result("set_seat_heating", {"seat_zone": "ALL_ZONES", "level": 2})
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {"tool_name": "set_steering_wheel_heating", "arguments": {"level": 2}}
    ]


def test_occupancy_seat_heating_only_turns_off_unoccupied_front_seats() -> None:
    controller = PolicyAwareController()
    tools = comprehensive_comfort_tools()

    action = controller.decide(
        context_id="ctx-seat-heating-only-occupancy",
        messages=messages(),
        tools=tools,
        latest_user_text=(
            "Turn off seat heating for unoccupied seats and set it to "
            "level 1 for occupied seats."
        ),
    )
    assert action is not None
    assert action.tool_calls == [{"tool_name": "get_seats_occupancy", "arguments": {}}]

    action = controller.decide(
        context_id="ctx-seat-heating-only-occupancy",
        messages=messages(),
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_seats_occupancy",
                {
                    "seats_occupied": {
                        "driver": True,
                        "passenger": False,
                        "driver_rear": False,
                        "passenger_rear": True,
                    }
                },
            )
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {"tool_name": "get_seat_heating_level", "arguments": {}}
    ]

    action = controller.decide(
        context_id="ctx-seat-heating-only-occupancy",
        messages=messages(),
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_seat_heating_level",
                {"seat_heating_driver": 2, "seat_heating_passenger": 2},
            )
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "set_seat_heating",
            "arguments": {"seat_zone": "PASSENGER", "level": 0},
        }
    ]

    action = controller.decide(
        context_id="ctx-seat-heating-only-occupancy",
        messages=messages(),
        tools=tools,
        latest_tool_results=[
            tool_result("set_seat_heating", {"seat_zone": "PASSENGER", "level": 0})
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "set_seat_heating",
            "arguments": {"seat_zone": "DRIVER", "level": 1},
        }
    ]

    action = controller.decide(
        context_id="ctx-seat-heating-only-occupancy",
        messages=messages(),
        tools=tools,
        latest_user_text="Is seat heating activated for my child?",
    )
    assert action is not None
    assert action.action == "respond"
    assert "rear seat" in action.content.lower()


def test_occupancy_comfort_turns_off_empty_seats_and_matches_driver_temperature() -> None:
    controller = PolicyAwareController()
    tools = [*climate_inspection_tools(), *occupancy_comfort_tools()]

    action = controller.decide(
        context_id="ctx-occupancy-match-driver-temp",
        messages=messages(),
        tools=tools,
        latest_user_text=(
            "Only I'm in the car. Turn off seat heating for empty seats and "
            "make the driver's temperature the same as the passenger's temperature."
        ),
    )
    assert action is not None
    assert action.tool_calls == [{"tool_name": "get_seats_occupancy", "arguments": {}}]

    action = controller.decide(
        context_id="ctx-occupancy-match-driver-temp",
        messages=messages(),
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_seats_occupancy",
                {
                    "seats_occupied": {
                        "driver": True,
                        "passenger": False,
                        "driver_rear": False,
                        "passenger_rear": False,
                    }
                },
            )
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {"tool_name": "get_temperature_inside_car", "arguments": {}}
    ]

    action = controller.decide(
        context_id="ctx-occupancy-match-driver-temp",
        messages=messages(),
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_temperature_inside_car",
                {
                    "climate_temperature_driver": 18.0,
                    "climate_temperature_passenger": 23.0,
                },
            )
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {"tool_name": "get_seat_heating_level", "arguments": {}}
    ]

    action = controller.decide(
        context_id="ctx-occupancy-match-driver-temp",
        messages=messages(),
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_seat_heating_level",
                {"seat_heating_driver": 2, "seat_heating_passenger": 3},
            )
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "set_seat_heating",
            "arguments": {"seat_zone": "PASSENGER", "level": 0},
        }
    ]

    action = controller.decide(
        context_id="ctx-occupancy-match-driver-temp",
        messages=messages(),
        tools=tools,
        latest_tool_results=[
            tool_result("set_seat_heating", {"seat_zone": "PASSENGER", "level": 0})
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "set_climate_temperature",
            "arguments": {"seat_zone": "DRIVER", "temperature": 23.0},
        }
    ]


def test_occupancy_comfort_answers_empty_seat_heating_status() -> None:
    controller = PolicyAwareController()
    tools = [*climate_inspection_tools(), *occupancy_comfort_tools()]

    action = controller.decide(
        context_id="ctx-empty-seat-heating-status",
        messages=messages(),
        tools=tools,
        latest_user_text="Is seat heating on for any of the empty seats?",
    )
    assert action is not None
    assert action.tool_calls == [{"tool_name": "get_seats_occupancy", "arguments": {}}]

    action = controller.decide(
        context_id="ctx-empty-seat-heating-status",
        messages=messages(),
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_seats_occupancy",
                {
                    "seats_occupied": {
                        "driver": True,
                        "passenger": False,
                        "driver_rear": False,
                        "passenger_rear": False,
                    }
                },
            )
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {"tool_name": "get_seat_heating_level", "arguments": {}}
    ]

    action = controller.decide(
        context_id="ctx-empty-seat-heating-status",
        messages=messages(),
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_seat_heating_level",
                {"seat_heating_driver": 2, "seat_heating_passenger": 3},
            )
        ],
    )
    assert action is not None
    assert action.action == "respond"
    assert "passenger seat at level 3" in action.content


def test_cabin_temperature_controller_sets_all_zones() -> None:
    controller = PolicyAwareController()
    tools = occupancy_comfort_tools()

    action = controller.decide(
        context_id="ctx-cabin-temperature",
        messages=messages(),
        tools=tools,
        latest_user_text="Set the climate temperature to 22 degrees Celsius for all zones.",
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "set_climate_temperature",
            "arguments": {"seat_zone": "ALL_ZONES", "temperature": 22.0},
        }
    ]


def test_occupancy_comfort_refuses_missing_occupancy_result() -> None:
    controller = PolicyAwareController()
    tools = occupancy_comfort_tools()

    controller.decide(
        context_id="ctx-occupancy-comfort-missing-result",
        messages=messages(),
        tools=tools,
        latest_user_text=(
            "Warm up the car efficiently, only for the occupied seats."
        ),
    )
    action = controller.decide(
        context_id="ctx-occupancy-comfort-missing-result",
        messages=messages(),
        tools=tools,
        latest_tool_results=[tool_result("get_seats_occupancy", {})],
    )
    assert action is not None
    assert action.action == "respond"
    assert "can't tell which seats are occupied" in action.content.lower()


def test_passenger_comfort_match_sets_temperature_and_matches_driver_heating() -> None:
    controller = PolicyAwareController()
    tools = passenger_comfort_match_tools()

    action = controller.decide(
        context_id="ctx-passenger-match",
        messages=messages(),
        tools=tools,
        latest_user_text=(
            "Set the passenger side temperature to 24 degrees Celsius, "
            "make the passenger seat heating match the driver's seat heating, "
            "and set the steering wheel heating to the same level."
        ),
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "set_climate_temperature",
            "arguments": {"seat_zone": "PASSENGER", "temperature": 24.0},
        }
    ]

    action = controller.decide(
        context_id="ctx-passenger-match",
        messages=messages(),
        tools=tools,
        latest_tool_results=[
            tool_result(
                "set_climate_temperature",
                {"seat_zone": "PASSENGER", "temperature": 24},
            )
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {"tool_name": "get_seat_heating_level", "arguments": {}}
    ]

    action = controller.decide(
        context_id="ctx-passenger-match",
        messages=messages(),
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_seat_heating_level",
                {"seat_heating_driver": 3, "seat_heating_passenger": 0},
            )
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "set_seat_heating",
            "arguments": {"seat_zone": "PASSENGER", "level": 3},
        }
    ]

    action = controller.decide(
        context_id="ctx-passenger-match",
        messages=messages(),
        tools=tools,
        latest_tool_results=[
            tool_result("set_seat_heating", {"seat_zone": "PASSENGER", "level": 3})
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {"tool_name": "set_steering_wheel_heating", "arguments": {"level": 3}}
    ]


def test_passenger_comfort_match_refuses_missing_seat_zone_parameter() -> None:
    controller = PolicyAwareController()
    tools = passenger_comfort_match_tools(include_seat_zone=False)

    controller.decide(
        context_id="ctx-passenger-match-missing-seat-zone",
        messages=messages(),
        tools=tools,
        latest_user_text=(
            "Set the passenger side temperature to 24 degrees Celsius, "
            "make the passenger seat heating match the driver's seat heating, "
            "and set the steering wheel heating to the same level."
        ),
    )
    controller.decide(
        context_id="ctx-passenger-match-missing-seat-zone",
        messages=messages(),
        tools=tools,
        latest_tool_results=[
            tool_result(
                "set_climate_temperature",
                {"seat_zone": "PASSENGER", "temperature": 24},
            )
        ],
    )
    action = controller.decide(
        context_id="ctx-passenger-match-missing-seat-zone",
        messages=messages(),
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_seat_heating_level",
                {"seat_heating_driver": 3, "seat_heating_passenger": 0},
            )
        ],
    )
    assert action is not None
    assert action.action == "respond"
    assert "seat selector is unavailable" in action.content.lower()


def test_driver_comfort_turns_off_passenger_heat_and_uses_preference() -> None:
    controller = PolicyAwareController()
    tools = driver_comfort_tools()

    action = controller.decide(
        context_id="ctx-driver-comfort",
        messages=messages(),
        tools=tools,
        latest_user_text=(
            "Since it's just me, turn off the passenger seat heating and "
            "raise the driver zone temperature to my comfort level."
        ),
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "set_seat_heating",
            "arguments": {"seat_zone": "PASSENGER", "level": 0},
        }
    ]

    action = controller.decide(
        context_id="ctx-driver-comfort",
        messages=messages(),
        tools=tools,
        latest_tool_results=[
            tool_result(
                "set_seat_heating",
                {"seat_zone": "PASSENGER", "level": 0},
            )
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_user_preferences",
            "arguments": {
                "preference_categories": {
                    "vehicle_settings": {"climate_control": True}
                }
            },
        }
    ]

    action = controller.decide(
        context_id="ctx-driver-comfort",
        messages=messages(),
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_user_preferences",
                {
                    "vehicle_settings": {
                        "climate_control": [
                            "user driver comfort temperature level is 22 degrees"
                        ]
                    }
                },
            )
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "set_climate_temperature",
            "arguments": {"seat_zone": "DRIVER", "temperature": 22.0},
        }
    ]


def test_driver_comfort_understands_my_temperature_after_passenger_heat_off() -> None:
    controller = PolicyAwareController()
    tools = driver_comfort_tools()

    action = controller.decide(
        context_id="ctx-driver-comfort-my-temp",
        messages=messages(),
        tools=tools,
        latest_user_text=(
            "Since it's just me, turn off the passenger seat heating and "
            "then raise my temperature to my comfort level."
        ),
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "set_seat_heating",
            "arguments": {"seat_zone": "PASSENGER", "level": 0},
        }
    ]


def test_driver_comfort_uses_final_temperature_after_relative_phrase() -> None:
    controller = PolicyAwareController()
    tools = driver_comfort_tools()

    controller.decide(
        context_id="ctx-driver-comfort-relative-temp",
        messages=messages(),
        tools=tools,
        latest_user_text=(
            "Since it's just me, turn off the passenger seat heating and "
            "then raise my temperature to my comfort level."
        ),
    )
    controller.decide(
        context_id="ctx-driver-comfort-relative-temp",
        messages=messages(),
        tools=tools,
        latest_tool_results=[
            tool_result("set_seat_heating", {"seat_zone": "PASSENGER", "level": 0})
        ],
    )
    controller.decide(
        context_id="ctx-driver-comfort-relative-temp",
        messages=messages(),
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_user_preferences",
                {"vehicle_settings": {"climate_control": []}},
            )
        ],
    )
    action = controller.decide(
        context_id="ctx-driver-comfort-relative-temp",
        messages=messages(),
        tools=tools,
        latest_user_text=(
            "I'd like to raise the driver zone temperature by 4 degrees "
            "Celsius, so that would be 22 degrees."
        ),
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "set_climate_temperature",
            "arguments": {"seat_zone": "DRIVER", "temperature": 22.0},
        }
    ]


def test_driver_comfort_uses_observed_temperature_for_relative_delta() -> None:
    controller = PolicyAwareController()
    tools = [*climate_inspection_tools(), *driver_comfort_tools()]

    controller.decide(
        context_id="ctx-driver-comfort-observed-delta",
        messages=messages(),
        tools=tools,
        latest_user_text=(
            "Can you tell me the current climate settings and seat heating levels?"
        ),
    )
    controller.decide(
        context_id="ctx-driver-comfort-observed-delta",
        messages=messages(),
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_temperature_inside_car",
                {
                    "climate_temperature_driver": 18.0,
                    "climate_temperature_passenger": 23.0,
                    "temperature_unit": "Celsius",
                },
            )
        ],
    )
    controller.decide(
        context_id="ctx-driver-comfort-observed-delta",
        messages=messages(),
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_seat_heating_level",
                {"seat_heating_driver": 3, "seat_heating_passenger": 3},
            )
        ],
    )

    action = controller.decide(
        context_id="ctx-driver-comfort-observed-delta",
        messages=messages(),
        tools=tools,
        latest_user_text=(
            "Since it's just me, turn off the passenger seat heating and "
            "raise the driver zone temperature by 4 degrees Celsius."
        ),
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "set_seat_heating",
            "arguments": {"seat_zone": "PASSENGER", "level": 0},
        }
    ]

    action = controller.decide(
        context_id="ctx-driver-comfort-observed-delta",
        messages=messages(),
        tools=tools,
        latest_tool_results=[
            tool_result("set_seat_heating", {"seat_zone": "PASSENGER", "level": 0})
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "set_climate_temperature",
            "arguments": {"seat_zone": "DRIVER", "temperature": 22.0},
        }
    ]


def test_driver_comfort_temperature_only_uses_preference() -> None:
    controller = PolicyAwareController()
    tools = driver_comfort_tools()

    action = controller.decide(
        context_id="ctx-driver-comfort-temperature-only",
        messages=messages(),
        tools=tools,
        latest_user_text="Raise the driver zone temperature to my comfort level.",
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_user_preferences",
            "arguments": {
                "preference_categories": {
                    "vehicle_settings": {"climate_control": True}
                }
            },
        }
    ]

    action = controller.decide(
        context_id="ctx-driver-comfort-temperature-only",
        messages=messages(),
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_user_preferences",
                {
                    "vehicle_settings": {
                        "climate_control": [
                            "user driver comfort temperature level is 22 degrees"
                        ]
                    }
                },
            )
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "set_climate_temperature",
            "arguments": {"seat_zone": "DRIVER", "temperature": 22.0},
        }
    ]


def test_ambient_light_controller_matches_car_color() -> None:
    tools = [
        fake_tool("get_car_color"),
        fake_tool(
            "set_ambient_lights",
            {
                "on": {"type": "boolean"},
                "lightcolor": {"type": "string"},
            },
            ["on", "lightcolor"],
        ),
    ]

    for idx, request in enumerate(
        (
            "Set the ambient lights to match my car's exterior color.",
            "Set the ambient lights to match the color of my car.",
        )
    ):
        controller = PolicyAwareController()
        context_id = f"ctx-ambient-car-color-{idx}"

        action = controller.decide(
            context_id=context_id,
            messages=messages(),
            tools=tools,
            latest_user_text=request,
        )
        assert action is not None
        assert action.tool_calls == [{"tool_name": "get_car_color", "arguments": {}}]

        action = controller.decide(
            context_id=context_id,
            messages=messages(),
            tools=tools,
            latest_tool_results=[tool_result("get_car_color", {"car_color": "PURPLE"})],
        )
        assert action is not None
        assert action.tool_calls == [
            {
                "tool_name": "set_ambient_lights",
                "arguments": {"on": True, "lightcolor": "PURPLE"},
            }
        ]


def test_window_match_controller_sets_rear_windows_to_front_position() -> None:
    controller = PolicyAwareController()
    tools = [
        fake_tool("get_vehicle_window_positions"),
        fake_tool(
            "open_close_window",
            {
                "window": {"type": "string"},
                "percentage": {"type": "number"},
            },
            ["window", "percentage"],
        ),
    ]

    action = controller.decide(
        context_id="ctx-window-match",
        messages=messages(),
        tools=tools,
        latest_user_text="Can you help me adjust the windows? I want the rear windows to match the front ones.",
    )
    assert action is not None
    assert action.tool_calls == [
        {"tool_name": "get_vehicle_window_positions", "arguments": {}}
    ]

    action = controller.decide(
        context_id="ctx-window-match",
        messages=messages(),
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_vehicle_window_positions",
                window_positions(driver=25, passenger=25),
            )
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "open_close_window",
            "arguments": {"window": "DRIVER_REAR", "percentage": 25},
        }
    ]

    action = controller.decide(
        context_id="ctx-window-match",
        messages=messages(),
        tools=tools,
        latest_tool_results=[
            tool_result("open_close_window", {"window": "DRIVER_REAR", "percentage": 25})
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "open_close_window",
            "arguments": {"window": "PASSENGER_REAR", "percentage": 25},
        }
    ]


def test_reading_light_occupancy_ignores_negated_ambient_light() -> None:
    controller = PolicyAwareController()
    tools = [
        fake_tool(
            "set_ambient_lights",
            {
                "on": {"type": "boolean"},
                "lightcolor": {"type": "string"},
            },
            ["on", "lightcolor"],
        ),
        fake_tool("get_seats_occupancy"),
        fake_tool(
            "set_reading_light",
            {
                "position": {"type": "string"},
                "on": {"type": "boolean"},
            },
            ["position", "on"],
        ),
    ]

    action = controller.decide(
        context_id="ctx-reading-not-ambient",
        messages=messages(),
        tools=tools,
        latest_user_text=(
            "Oh, not the ambient lights. I was thinking about the reading lights. "
            "Turn them on for occupied seats and off for empty seats."
        ),
    )
    assert action is not None
    assert action.tool_calls == [{"tool_name": "get_seats_occupancy", "arguments": {}}]


def test_reading_light_occupancy_controller_uses_seat_occupancy() -> None:
    controller = PolicyAwareController()
    tools = [
        fake_tool("get_seats_occupancy"),
        fake_tool(
            "set_reading_light",
            {
                "position": {"type": "string"},
                "on": {"type": "boolean"},
            },
            ["position", "on"],
        ),
    ]

    action = controller.decide(
        context_id="ctx-reading-occupancy",
        messages=messages(),
        tools=tools,
        latest_user_text=(
            "Turn on reading lights for occupied seats and turn off reading lights "
            "for unoccupied seats."
        ),
    )
    assert action is not None
    assert action.tool_calls == [{"tool_name": "get_seats_occupancy", "arguments": {}}]

    expected_calls = [
        {"position": "DRIVER", "on": True},
        {"position": "PASSENGER_REAR", "on": True},
        {"position": "DRIVER_REAR", "on": False},
        {"position": "PASSENGER", "on": False},
    ]
    latest_results = [
        tool_result(
            "get_seats_occupancy",
            {
                "seats_occupied": {
                    "driver": True,
                    "passenger": False,
                    "driver_rear": False,
                    "passenger_rear": True,
                }
            },
        )
    ]
    for expected in expected_calls:
        action = controller.decide(
            context_id="ctx-reading-occupancy",
            messages=messages(),
            tools=tools,
            latest_tool_results=latest_results,
        )
        assert action is not None
        assert action.tool_calls == [
            {"tool_name": "set_reading_light", "arguments": expected}
        ]
        latest_results = [tool_result("set_reading_light", {"position": expected["position"]})]


def test_air_conditioning_controller_satisfies_policy_before_turning_on() -> None:
    controller = PolicyAwareController()
    tools = climate_tools()

    action = controller.decide(
        context_id="ctx-ac-policy",
        messages=messages(),
        tools=tools,
        latest_user_text="Turn on the air conditioning.",
    )
    assert action is not None
    assert action.tool_calls == [{"tool_name": "get_climate_settings", "arguments": {}}]

    action = controller.decide(
        context_id="ctx-ac-policy",
        messages=messages(),
        tools=tools,
        latest_tool_results=[tool_result("get_climate_settings", climate_result())],
    )
    assert action is not None
    assert action.tool_calls == [
        {"tool_name": "get_vehicle_window_positions", "arguments": {}}
    ]

    action = controller.decide(
        context_id="ctx-ac-policy",
        messages=messages(),
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_vehicle_window_positions",
                window_positions(driver=100, passenger=10, passenger_rear=25),
            )
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "open_close_window",
            "arguments": {"window": "DRIVER", "percentage": 0},
        }
    ]

    action = controller.decide(
        context_id="ctx-ac-policy",
        messages=messages(),
        tools=tools,
        latest_tool_results=[
            tool_result("open_close_window", {"window": "DRIVER", "percentage": 0})
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "open_close_window",
            "arguments": {"window": "PASSENGER_REAR", "percentage": 0},
        }
    ]

    action = controller.decide(
        context_id="ctx-ac-policy",
        messages=messages(),
        tools=tools,
        latest_tool_results=[
            tool_result(
                "open_close_window",
                {"window": "PASSENGER_REAR", "percentage": 0},
            )
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {"tool_name": "set_fan_speed", "arguments": {"level": 1}}
    ]

    action = controller.decide(
        context_id="ctx-ac-policy",
        messages=messages(),
        tools=tools,
        latest_tool_results=[tool_result("set_fan_speed", {"level": 1})],
    )
    assert action is not None
    assert action.tool_calls == [
        {"tool_name": "set_air_conditioning", "arguments": {"on": True}}
    ]


def test_cool_down_request_uses_air_conditioning_policy_flow() -> None:
    controller = PolicyAwareController()
    tools = climate_tools()

    action = controller.decide(
        context_id="ctx-cool-down-policy",
        messages=messages(),
        tools=tools,
        latest_user_text="I'm feeling warm inside the car. Can you help me cool down?",
    )
    assert action is not None
    assert action.tool_calls == [{"tool_name": "get_climate_settings", "arguments": {}}]

    action = controller.decide(
        context_id="ctx-cool-down-policy",
        messages=messages(),
        tools=tools,
        latest_tool_results=[
            tool_result("get_climate_settings", climate_result(fan_speed=0))
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {"tool_name": "get_vehicle_window_positions", "arguments": {}}
    ]

    action = controller.decide(
        context_id="ctx-cool-down-policy",
        messages=messages(),
        tools=tools,
        latest_tool_results=[
            tool_result("get_vehicle_window_positions", window_positions(driver=100))
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "open_close_window",
            "arguments": {"window": "DRIVER", "percentage": 0},
        }
    ]

    action = controller.decide(
        context_id="ctx-cool-down-policy",
        messages=messages(),
        tools=tools,
        latest_tool_results=[
            tool_result("open_close_window", {"window": "DRIVER", "percentage": 0})
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {"tool_name": "set_fan_speed", "arguments": {"level": 1}}
    ]


def test_air_conditioning_controller_preserves_explicit_fan_level() -> None:
    controller = PolicyAwareController()
    tools = climate_tools()

    controller.decide(
        context_id="ctx-ac-explicit-fan",
        messages=messages(),
        tools=tools,
        latest_user_text=(
            "Close the window fully open, turn on the air conditioning, "
            "and set the fan speed to level 3."
        ),
    )
    controller.decide(
        context_id="ctx-ac-explicit-fan",
        messages=messages(),
        tools=tools,
        latest_tool_results=[tool_result("get_climate_settings", climate_result())],
    )
    action = controller.decide(
        context_id="ctx-ac-explicit-fan",
        messages=messages(),
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_vehicle_window_positions",
                window_positions(driver=5, driver_rear=100),
            )
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "open_close_window",
            "arguments": {"window": "DRIVER_REAR", "percentage": 0},
        }
    ]

    action = controller.decide(
        context_id="ctx-ac-explicit-fan",
        messages=messages(),
        tools=tools,
        latest_tool_results=[
            tool_result("open_close_window", {"window": "DRIVER_REAR", "percentage": 0})
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {"tool_name": "set_fan_speed", "arguments": {"level": 3}}
    ]

    action = controller.decide(
        context_id="ctx-ac-explicit-fan",
        messages=messages(),
        tools=tools,
        latest_tool_results=[tool_result("set_fan_speed", {"level": 3})],
    )
    assert action is not None
    assert action.tool_calls == [
        {"tool_name": "set_air_conditioning", "arguments": {"on": True}}
    ]

    action = controller.decide(
        context_id="ctx-ac-explicit-fan",
        messages=messages(),
        tools=tools,
        latest_tool_results=[tool_result("set_air_conditioning", {"on": True})],
    )
    assert action is not None
    assert action.action == "respond"
    assert "closed the open windows" in action.content
    assert "fan speed to level 3" in action.content
    assert "turned on the air conditioning" in action.content


def test_air_conditioning_controller_refuses_when_fan_level_parameter_is_missing() -> None:
    controller = PolicyAwareController()
    tools = climate_tools(include_fan_level=False)

    controller.decide(
        context_id="ctx-ac-missing-fan-level",
        messages=messages(),
        tools=tools,
        latest_user_text="Turn on the AC.",
    )
    controller.decide(
        context_id="ctx-ac-missing-fan-level",
        messages=messages(),
        tools=tools,
        latest_tool_results=[tool_result("get_climate_settings", climate_result())],
    )
    action = controller.decide(
        context_id="ctx-ac-missing-fan-level",
        messages=messages(),
        tools=tools,
        latest_tool_results=[
            tool_result("get_vehicle_window_positions", window_positions())
        ],
    )
    assert action is not None
    assert action.action == "respond"
    assert "fan speed level control is unavailable" in action.content.lower()


def test_air_quality_question_clarifies_before_taking_action() -> None:
    controller = PolicyAwareController()
    tools = climate_tools()

    action = controller.decide(
        context_id="ctx-air-quality",
        messages=messages(),
        tools=tools,
        latest_user_text="The air in the car feels a bit stagnant. What can be done about that?",
    )
    assert action is not None
    assert action.tool_calls == [{"tool_name": "get_climate_settings", "arguments": {}}]

    action = controller.decide(
        context_id="ctx-air-quality",
        messages=messages(),
        tools=tools,
        latest_tool_results=[tool_result("get_climate_settings", climate_result())],
    )
    assert action is not None
    assert action.action == "respond"
    assert "fan is currently off" in action.content.lower()
    assert "which would you like" in action.content.lower()

    action = controller.decide(
        context_id="ctx-air-quality",
        messages=messages(),
        tools=tools,
        latest_user_text="Set it to level 2.",
    )
    assert action is not None
    assert action.tool_calls == [
        {"tool_name": "set_fan_speed", "arguments": {"level": 2}}
    ]


def test_air_quality_settings_request_does_not_trigger_air_conditioning_flow() -> None:
    controller = PolicyAwareController()
    tools = climate_tools()

    action = controller.decide(
        context_id="ctx-air-quality-settings",
        messages=messages(),
        tools=tools,
        latest_user_text=(
            "Could you show me the current climate settings? "
            "I feel like the air is a bit stuffy in here."
        ),
    )
    assert action is not None
    assert action.tool_calls == [{"tool_name": "get_climate_settings", "arguments": {}}]

    action = controller.decide(
        context_id="ctx-air-quality-settings",
        messages=messages(),
        tools=tools,
        latest_tool_results=[tool_result("get_climate_settings", climate_result())],
    )
    assert action is not None
    assert action.action == "respond"

    action = controller.decide(
        context_id="ctx-air-quality-settings",
        messages=messages(),
        tools=tools,
        latest_user_text="Could you set the fan to level 2 then?",
    )
    assert action is not None
    assert action.tool_calls == [
        {"tool_name": "set_fan_speed", "arguments": {"level": 2}}
    ]


def test_sunshade_controller_asks_before_setting_ambiguous_percentage() -> None:
    controller = PolicyAwareController()
    tools = [
        fake_tool(
            "open_close_sunshade",
            {"percentage": {"type": "number"}},
            ["percentage"],
        )
    ]

    action = controller.decide(
        context_id="ctx-sunshade",
        messages=messages(),
        tools=tools,
        latest_user_text=(
            "The sun is really bright this morning. Can you help me with the sunshade?"
        ),
    )
    assert action is not None
    assert action.action == "respond"
    assert "how far" in action.content.lower()

    action = controller.decide(
        context_id="ctx-sunshade",
        messages=messages(),
        tools=tools,
        latest_user_text="Let's go with 50%.",
    )
    assert action is not None
    assert action.tool_calls == [
        {"tool_name": "open_close_sunshade", "arguments": {"percentage": 50}}
    ]


def test_fan_speed_controller_uses_preference_before_setting_level() -> None:
    controller = PolicyAwareController()
    tools = [
        fake_tool("get_user_preferences"),
        fake_tool("set_fan_speed", {"level": {"type": "integer"}}, ["level"]),
    ]

    action = controller.decide(
        context_id="ctx-fan-pref",
        messages=messages(),
        tools=tools,
        latest_user_text="Could you turn on the fan?",
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_user_preferences",
            "arguments": {
                "preference_categories": {
                    "vehicle_settings": {"climate_control": True}
                }
            },
        }
    ]

    action = controller.decide(
        context_id="ctx-fan-pref",
        messages=messages(),
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_user_preferences",
                {
                    "vehicle_settings": {
                        "climate_control": [
                            "user prefers fan speed level 3 as default value for moderate airflow"
                        ]
                    }
                },
            )
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {"tool_name": "set_fan_speed", "arguments": {"level": 3}}
    ]


def test_fan_speed_controller_asks_when_preference_is_missing() -> None:
    controller = PolicyAwareController()
    tools = [
        fake_tool("get_user_preferences"),
        fake_tool("set_fan_speed", {"level": {"type": "integer"}}, ["level"]),
    ]

    controller.decide(
        context_id="ctx-fan-ask",
        messages=messages(),
        tools=tools,
        latest_user_text="Turn on the fan.",
    )
    action = controller.decide(
        context_id="ctx-fan-ask",
        messages=messages(),
        tools=tools,
        latest_tool_results=[tool_result("get_user_preferences", {})],
    )
    assert action is not None
    assert action.action == "respond"
    assert "fan speed level" in action.content.lower()

    action = controller.decide(
        context_id="ctx-fan-ask",
        messages=messages(),
        tools=tools,
        latest_user_text="Set it to level 3.",
    )
    assert action is not None
    assert action.tool_calls == [
        {"tool_name": "set_fan_speed", "arguments": {"level": 3}}
    ]


def test_fan_speed_controller_refuses_when_level_parameter_is_missing() -> None:
    controller = PolicyAwareController()
    tools = [fake_tool("set_fan_speed")]

    action = controller.decide(
        context_id="ctx-fan-missing-level",
        messages=messages(),
        tools=tools,
        latest_user_text="Set the fan speed to level 3.",
    )
    assert action is not None
    assert action.action == "respond"
    assert "level control is unavailable" in action.content.lower()


def test_steering_wheel_heating_controller_uses_preference_level() -> None:
    controller = PolicyAwareController()
    tools = [
        fake_tool("get_user_preferences"),
        fake_tool(
            "set_steering_wheel_heating",
            {"level": {"type": "integer"}},
            ["level"],
        ),
    ]

    action = controller.decide(
        context_id="ctx-heating-pref",
        messages=messages(),
        tools=tools,
        latest_user_text="Turn on the steering wheel heating.",
    )
    assert action is not None
    assert action.tool_calls[0]["tool_name"] == "get_user_preferences"

    action = controller.decide(
        context_id="ctx-heating-pref",
        messages=messages(),
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_user_preferences",
                {
                    "vehicle_settings": {
                        "vehicle_settings": [
                            "If steering wheel heating should be turned on, user prefers level 2, because level 1 is not feelable and level 3 is too hot"
                        ]
                    }
                },
            )
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {"tool_name": "set_steering_wheel_heating", "arguments": {"level": 2}}
    ]


def test_reading_light_controller_asks_for_ambiguous_position() -> None:
    controller = PolicyAwareController()
    tools = [
        fake_tool(
            "set_reading_light",
            {
                "position": {"type": "string"},
                "on": {"type": "boolean"},
            },
            ["position", "on"],
        )
    ]

    action = controller.decide(
        context_id="ctx-reading-light",
        messages=messages(),
        tools=tools,
        latest_user_text="Turn on the reading lights.",
    )
    assert action is not None
    assert action.action == "respond"
    assert "which reading light" in action.content.lower()

    action = controller.decide(
        context_id="ctx-reading-light",
        messages=messages(),
        tools=tools,
        latest_user_text="The driver reading light.",
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "set_reading_light",
            "arguments": {"position": "DRIVER", "on": True},
        }
    ]


def test_reading_light_controller_refuses_when_position_parameter_is_missing() -> None:
    controller = PolicyAwareController()
    tools = [fake_tool("set_reading_light", {"on": {"type": "boolean"}}, ["on"])]

    action = controller.decide(
        context_id="ctx-reading-light-missing-position",
        messages=messages(),
        tools=tools,
        latest_user_text="Turn on the driver reading light.",
    )
    assert action is not None
    assert action.action == "respond"
    assert "position control is unavailable" in action.content.lower()


def test_high_beam_generic_beam_request_checks_exterior_status() -> None:
    controller = PolicyAwareController()
    tools = [
        fake_tool("get_exterior_lights_status"),
        fake_tool("set_head_lights_high_beams", {"on": {"type": "boolean"}}, ["on"]),
    ]

    action = controller.decide(
        context_id="ctx-generic-beams",
        messages=messages(),
        tools=tools,
        latest_user_text="Turn on the beams, please.",
    )
    assert action is not None
    assert action.tool_calls == [
        {"tool_name": "get_exterior_lights_status", "arguments": {}}
    ]


def test_high_beam_controller_refuses_when_fog_status_is_missing() -> None:
    controller = PolicyAwareController()
    tools = [
        fake_tool("get_exterior_lights_status"),
        fake_tool("set_head_lights_high_beams", {"on": {"type": "boolean"}}, ["on"]),
    ]

    controller.decide(
        context_id="ctx-high-beams-missing-fog",
        messages=messages(),
        tools=tools,
        latest_user_text="Turn on the high beams.",
    )
    action = controller.decide(
        context_id="ctx-high-beams-missing-fog",
        messages=messages(),
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_exterior_lights_status",
                {
                    "head_lights_low_beams": True,
                    "head_lights_high_beams": False,
                },
            )
        ],
    )
    assert action is not None
    assert action.action == "respond"
    assert "fog light status" in action.content.lower()


def test_completed_window_flow_does_not_block_followup_ac_and_air_circulation() -> None:
    controller = PolicyAwareController()
    tools = climate_tools() + [
        fake_tool("set_air_circulation", {"mode": {"type": "string"}}, ["mode"])
    ]

    action = controller.decide(
        context_id="ctx-window-then-ac-circulation",
        messages=messages(),
        tools=tools,
        latest_user_text="Open all windows to 50%.",
    )
    assert action is not None
    assert action.tool_calls == [{"tool_name": "get_climate_settings", "arguments": {}}]

    action = controller.decide(
        context_id="ctx-window-then-ac-circulation",
        messages=messages(),
        tools=tools,
        latest_tool_results=[tool_result("get_climate_settings", climate_result())],
    )
    assert action is not None
    assert action.tool_calls == [
        {"tool_name": "open_close_window", "arguments": {"window": "ALL", "percentage": 50}}
    ]

    action = controller.decide(
        context_id="ctx-window-then-ac-circulation",
        messages=messages(),
        tools=tools,
        latest_tool_results=[
            tool_result("open_close_window", {"window": "ALL", "percentage": 50})
        ],
    )
    assert action is not None
    assert action.action == "respond"

    action = controller.decide(
        context_id="ctx-window-then-ac-circulation",
        messages=messages(),
        tools=tools,
        latest_user_text=(
            "What about the AC and air circulation? Turn on the AC and set "
            "air circulation to fresh air."
        ),
    )
    assert action is not None
    assert action.tool_calls == [{"tool_name": "get_climate_settings", "arguments": {}}]

    action = controller.decide(
        context_id="ctx-window-then-ac-circulation",
        messages=messages(),
        tools=tools,
        latest_tool_results=[tool_result("get_climate_settings", climate_result())],
    )
    assert action is not None
    assert action.tool_calls == [
        {"tool_name": "get_vehicle_window_positions", "arguments": {}}
    ]

    action = controller.decide(
        context_id="ctx-window-then-ac-circulation",
        messages=messages(),
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_vehicle_window_positions",
                window_positions(
                    driver=50,
                    passenger=50,
                    driver_rear=50,
                    passenger_rear=50,
                ),
            )
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {"tool_name": "open_close_window", "arguments": {"window": "ALL", "percentage": 0}}
    ]

    action = controller.decide(
        context_id="ctx-window-then-ac-circulation",
        messages=messages(),
        tools=tools,
        latest_tool_results=[
            tool_result("open_close_window", {"window": "ALL", "percentage": 0})
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {"tool_name": "set_fan_speed", "arguments": {"level": 1}}
    ]

    action = controller.decide(
        context_id="ctx-window-then-ac-circulation",
        messages=messages(),
        tools=tools,
        latest_tool_results=[tool_result("set_fan_speed", {"level": 1})],
    )
    assert action is not None
    assert action.tool_calls == [
        {"tool_name": "set_air_conditioning", "arguments": {"on": True}}
    ]

    action = controller.decide(
        context_id="ctx-window-then-ac-circulation",
        messages=messages(),
        tools=tools,
        latest_tool_results=[tool_result("set_air_conditioning", {"on": True})],
    )
    assert action is not None
    assert action.tool_calls == [
        {"tool_name": "set_air_circulation", "arguments": {"mode": "FRESH_AIR"}}
    ]

    action = controller.decide(
        context_id="ctx-window-then-ac-circulation",
        messages=messages(),
        tools=tools,
        latest_tool_results=[
            tool_result("set_air_circulation", {"mode": "FRESH_AIR"})
        ],
    )
    assert action is not None
    assert action.action == "respond"
    assert "air conditioning" in action.content.lower()
    assert "fresh air mode" in action.content.lower()


def test_combined_window_ac_request_preserves_requested_open_windows() -> None:
    controller = PolicyAwareController()
    tools = climate_tools() + [
        fake_tool("set_air_circulation", {"mode": {"type": "string"}}, ["mode"])
    ]

    action = controller.decide(
        context_id="ctx-window-ac-same-request",
        messages=messages(),
        tools=tools,
        latest_user_text=(
            "Open all the windows to 50%, turn on the AC, and set air "
            "circulation to fresh air mode."
        ),
    )
    assert action is not None
    assert action.tool_calls == [{"tool_name": "get_climate_settings", "arguments": {}}]

    action = controller.decide(
        context_id="ctx-window-ac-same-request",
        messages=messages(),
        tools=tools,
        latest_tool_results=[tool_result("get_climate_settings", climate_result())],
    )
    assert action is not None
    assert action.tool_calls == [
        {"tool_name": "open_close_window", "arguments": {"window": "ALL", "percentage": 50}}
    ]

    action = controller.decide(
        context_id="ctx-window-ac-same-request",
        messages=messages(),
        tools=tools,
        latest_tool_results=[
            tool_result("open_close_window", {"window": "ALL", "percentage": 50})
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {"tool_name": "set_fan_speed", "arguments": {"level": 1}}
    ]

    action = controller.decide(
        context_id="ctx-window-ac-same-request",
        messages=messages(),
        tools=tools,
        latest_tool_results=[tool_result("set_fan_speed", {"level": 1})],
    )
    assert action is not None
    assert action.tool_calls == [
        {"tool_name": "set_air_conditioning", "arguments": {"on": True}}
    ]

    action = controller.decide(
        context_id="ctx-window-ac-same-request",
        messages=messages(),
        tools=tools,
        latest_tool_results=[tool_result("set_air_conditioning", {"on": True})],
    )
    assert action is not None
    assert action.tool_calls == [
        {"tool_name": "set_air_circulation", "arguments": {"mode": "FRESH_AIR"}}
    ]

    action = controller.decide(
        context_id="ctx-window-ac-same-request",
        messages=messages(),
        tools=tools,
        latest_tool_results=[
            tool_result("set_air_circulation", {"mode": "FRESH_AIR"})
        ],
    )
    assert action is not None
    assert action.action == "respond"
    lowered = action.content.lower()
    assert "all windows" in lowered
    assert "50%" in lowered
    assert "air conditioning" in lowered
    assert "fresh air mode" in lowered


def test_combined_window_ac_request_does_not_invent_unknown_fan_speed() -> None:
    controller = PolicyAwareController()
    tools = climate_tools() + [
        fake_tool("set_air_circulation", {"mode": {"type": "string"}}, ["mode"])
    ]

    action = controller.decide(
        context_id="ctx-window-ac-unknown-fan",
        messages=messages(),
        tools=tools,
        latest_user_text=(
            "Open all the windows to 50%, turn on the AC, and set air "
            "circulation to fresh air mode."
        ),
    )
    assert action is not None
    assert action.tool_calls == [{"tool_name": "get_climate_settings", "arguments": {}}]

    action = controller.decide(
        context_id="ctx-window-ac-unknown-fan",
        messages=messages(),
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_climate_settings",
                {
                    "fan_speed": "unknown",
                    "fan_airflow_direction": "HEAD",
                    "air_conditioning": False,
                    "air_circulation": "RECIRCULATION",
                },
            )
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {"tool_name": "open_close_window", "arguments": {"window": "ALL", "percentage": 50}}
    ]

    action = controller.decide(
        context_id="ctx-window-ac-unknown-fan",
        messages=messages(),
        tools=tools,
        latest_tool_results=[
            tool_result("open_close_window", {"window": "ALL", "percentage": 50})
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {"tool_name": "set_air_conditioning", "arguments": {"on": True}}
    ]

    action = controller.decide(
        context_id="ctx-window-ac-unknown-fan",
        messages=messages(),
        tools=tools,
        latest_tool_results=[tool_result("set_air_conditioning", {"on": True})],
    )
    assert action is not None
    assert action.tool_calls == [
        {"tool_name": "set_air_circulation", "arguments": {"mode": "FRESH_AIR"}}
    ]

    action = controller.decide(
        context_id="ctx-window-ac-unknown-fan",
        messages=messages(),
        tools=tools,
        latest_tool_results=[
            tool_result("set_air_circulation", {"mode": "FRESH_AIR"})
        ],
    )
    assert action is not None
    assert action.action == "respond"
    lowered = action.content.lower()
    assert "all windows" in lowered
    assert "50%" in lowered
    assert "air conditioning" in lowered
    assert "fresh air mode" in lowered
    assert "fan speed" not in lowered

    action = controller.decide(
        context_id="ctx-window-ac-unknown-fan",
        messages=messages(),
        tools=tools,
        latest_user_text="So, the windows are still open at 50% even with the AC on?",
    )
    assert action is not None
    assert action.action == "respond"
    lowered = action.content.lower()
    assert "all windows" in lowered
    assert "50%" in lowered
    assert "air conditioning is on" in lowered
    assert action.tool_calls == []


def test_fan_speed_current_query_reports_unknown_climate_field() -> None:
    controller = PolicyAwareController()
    tools = climate_tools()

    action = controller.decide(
        context_id="ctx-current-fan-speed",
        messages=messages(),
        tools=tools,
        latest_user_text="Can you look up the current fan speed for me?",
    )
    assert action is not None
    assert action.tool_calls == [{"tool_name": "get_climate_settings", "arguments": {}}]

    action = controller.decide(
        context_id="ctx-current-fan-speed",
        messages=messages(),
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_climate_settings",
                {
                    "fan_speed": "unknown",
                    "fan_airflow_direction": "FEET",
                    "air_conditioning": True,
                    "air_circulation": "FRESH_AIR",
                },
            )
        ],
    )
    assert action is not None
    assert action.action == "respond"
    assert "can't determine the current fan speed" in action.content
    assert "what fan speed level" not in action.content.lower()
