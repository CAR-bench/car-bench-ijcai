import json

from carbench_agent_core import PolicyAwareController


def fake_tool(
    name: str,
    properties: dict | None = None,
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


def system_messages(
    location_id: str = "loc_origin", *, hour: int = 10, minute: int = 0
) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                f'CURRENT_LOCATION = {{"id": "{location_id}", "name": "Origin"}}\n'
                'DATETIME = {"year": 2026, "month": 6, "day": 17, '
                f'"hour": {hour}, "minute": {minute}}}'
            ),
        }
    ]


def route_energy_tools(
    *,
    include_along_route: bool = True,
    include_status: bool = True,
    include_route_lookup: bool = False,
) -> list[dict]:
    tools = [
        fake_tool(
            "get_current_navigation_state",
            {"detailed_information": {"type": "boolean"}},
        ),
        fake_tool(
            "get_distance_by_soc",
            {
                "initial_state_of_charge": {"type": "number"},
                "final_state_of_charge": {"type": "number"},
            },
            ["initial_state_of_charge", "final_state_of_charge"],
        ),
        fake_tool(
            "search_poi_at_location",
            {
                "category_poi": {"type": "string"},
                "location_id": {"type": "string"},
                "filters": {"type": "array"},
            },
            ["category_poi", "location_id"],
        ),
        fake_tool(
            "calculate_charging_time_by_soc",
            {
                "charging_station_id": {"type": "string"},
                "charging_station_plug_id": {"type": "string"},
                "start_state_of_charge": {"type": "number"},
                "target_state_of_charge": {"type": "number"},
            },
            [
                "charging_station_id",
                "charging_station_plug_id",
                "start_state_of_charge",
                "target_state_of_charge",
            ],
        ),
        fake_tool(
            "call_phone_by_number",
            {"phone_number": {"type": "string"}},
            ["phone_number"],
        ),
    ]
    if include_status:
        tools.insert(1, fake_tool("get_charging_specs_and_status"))
    if include_route_lookup:
        tools.extend(
            [
                fake_tool(
                    "get_location_id_by_location_name",
                    {"location": {"type": "string"}},
                    ["location"],
                ),
                fake_tool(
                    "get_routes_from_start_to_destination",
                    {
                        "start_id": {"type": "string"},
                        "destination_id": {"type": "string"},
                    },
                    ["start_id", "destination_id"],
                ),
            ]
        )
    if include_along_route:
        tools.insert(
            4,
            fake_tool(
                "search_poi_along_the_route",
                {
                    "category_poi": {"type": "string"},
                    "route_id": {"type": "string"},
                    "at_kilometer": {"type": "number"},
                    "filters": {"type": "array"},
                },
                ["category_poi", "route_id", "at_kilometer"],
            ),
        )
    return tools


def current_navigation_result() -> dict:
    return {
        "navigation_active": True,
        "waypoints_id": ["loc_origin", "loc_mid", "loc_final"],
        "routes_to_final_destination_id": ["route_origin_mid", "route_mid_final"],
        "details": {
            "routes": [
                {"route_id": "route_origin_mid", "distance_km": 410},
                {"route_id": "route_mid_final", "distance_km": 260},
            ]
        },
    }


def charging_station_result() -> dict:
    return {
        "pois_found_along_route": [
            {
                "id": "poi_charge_alpha",
                "name": "VoltHub",
                "phone_number": "+49 111 222333",
                "charging_plugs": [
                    {
                        "plug_id": "plug_ac",
                        "power_type": "AC",
                        "power_kw": 22,
                        "availability": "available",
                    },
                    {
                        "plug_id": "plug_dc_fast",
                        "power_type": "DC",
                        "power_kw": 250,
                        "availability": "available",
                    },
                ],
            }
        ]
    }


def fast_food_result() -> dict:
    return {
        "pois_found_along_route": [
            {
                "id": "poi_fast_alpha",
                "name": "Quick Burger",
                "category": "fast_food",
                "opening_hours": "08:00h - 22:00h",
            }
        ]
    }


def multistop_navigation_result() -> dict:
    return {
        "navigation_active": True,
        "waypoints_id": ["loc_origin", "loc_mid", "loc_cologne"],
        "routes_to_final_destination_id": ["route_origin_mid", "route_mid_cologne"],
        "details": {
            "waypoints": [
                {"id": "loc_origin", "name": "Origin"},
                {"id": "loc_mid", "name": "Midpoint"},
                {"id": "loc_cologne", "name": "Cologne"},
            ],
            "routes": [
                {"route_id": "route_origin_mid", "distance_km": 350},
                {"route_id": "route_mid_cologne", "distance_km": 180},
            ],
        },
    }


def route_lookup_result() -> dict:
    return {
        "routes": [
            {
                "route_id": "route_origin_alpine_fast",
                "distance_km": 900,
                "alias": ["fastest"],
            },
            {
                "route_id": "route_origin_alpine_scenic",
                "distance_km": 1050,
                "alias": ["scenic"],
            },
        ]
    }


def test_route_energy_calculates_distance_from_soc_range() -> None:
    controller = PolicyAwareController()

    action = controller.decide(
        context_id="ctx-range",
        messages=system_messages(),
        tools=route_energy_tools(),
        latest_user_text="How far can I drive from 80% battery down to 20%?",
    )

    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_distance_by_soc",
            "arguments": {
                "initial_state_of_charge": 80.0,
                "final_state_of_charge": 20.0,
            },
        }
    ]


def test_route_energy_estimates_charging_stops_from_charge_cycle() -> None:
    controller = PolicyAwareController()
    tools = route_energy_tools(include_route_lookup=True)

    action = controller.decide(
        context_id="ctx-stops",
        messages=system_messages(),
        tools=tools,
        latest_user_text=(
            "How many charging stops would I need to get to Borealis on the "
            "fastest route if I charge from 10 to 80 percent?"
        ),
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_location_id_by_location_name",
            "arguments": {"location": "Borealis"},
        }
    ]

    action = controller.decide(
        context_id="ctx-stops",
        messages=system_messages(),
        tools=tools,
        latest_tool_results=[
            tool_result("get_location_id_by_location_name", {"id": "loc_alpine"})
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_routes_from_start_to_destination",
            "arguments": {
                "start_id": "loc_origin",
                "destination_id": "loc_alpine",
            },
        }
    ]

    action = controller.decide(
        context_id="ctx-stops",
        messages=system_messages(),
        tools=tools,
        latest_tool_results=[
            tool_result("get_routes_from_start_to_destination", route_lookup_result())
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_distance_by_soc",
            "arguments": {
                "initial_state_of_charge": 80.0,
                "final_state_of_charge": 10.0,
            },
        }
    ]

    action = controller.decide(
        context_id="ctx-stops",
        messages=system_messages(),
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_distance_by_soc",
                {"distance_km_for_80_until_10_percent_soc": "300km"},
            )
        ],
    )
    assert action is not None
    assert action.action == "respond"
    assert "2 charging stops" in action.content
    assert "fastest route" in action.content
    assert "alternative routes" in action.content


def test_route_energy_route_choice_with_battery_range_query_checks_status() -> None:
    controller = PolicyAwareController()
    tools = route_energy_tools(include_route_lookup=True)

    action = controller.decide(
        context_id="ctx-route-choice-range",
        messages=system_messages(),
        tools=tools,
        latest_user_text="Navigate to Harbor City and check my battery range.",
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_location_id_by_location_name",
            "arguments": {"location": "Harbor"},
        }
    ]

    action = controller.decide(
        context_id="ctx-route-choice-range",
        messages=system_messages(),
        tools=tools,
        latest_tool_results=[
            tool_result("get_location_id_by_location_name", {"id": "loc_harbor"})
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_routes_from_start_to_destination",
            "arguments": {
                "start_id": "loc_origin",
                "destination_id": "loc_harbor",
            },
        }
    ]

    action = controller.decide(
        context_id="ctx-route-choice-range",
        messages=system_messages(),
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_routes_from_start_to_destination",
                {
                    "routes": [
                        {"route_id": "route_fast", "distance_km": 100, "alias": ["fastest"]},
                        {
                            "route_id": "route_second",
                            "distance_km": 120,
                            "alias": ["second"],
                            "name_via": "B2",
                        },
                        {"route_id": "route_short", "distance_km": 90, "alias": ["shortest"]},
                    ]
                },
            )
        ],
    )
    assert action is not None
    assert action.action == "respond"

    action = controller.decide(
        context_id="ctx-route-choice-range",
        messages=system_messages(),
        tools=tools,
        latest_user_text=(
            "I want the second route option, via B2. Do not set the navigation "
            "yet. What is my battery range?"
        ),
    )
    assert action is not None
    assert action.tool_calls == [
        {"tool_name": "get_charging_specs_and_status", "arguments": {}}
    ]

    action = controller.decide(
        context_id="ctx-route-choice-range",
        messages=system_messages(),
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_charging_specs_and_status",
                {"state_of_charge": 35.0, "remaining_range": "155km"},
            )
        ],
    )
    assert action is not None
    assert action.action == "respond"
    assert "155 km" in action.content


def test_route_energy_refuses_missing_battery_status_tool() -> None:
    controller = PolicyAwareController()

    action = controller.decide(
        context_id="ctx-missing-status",
        messages=system_messages(),
        tools=route_energy_tools(include_status=False),
        latest_user_text=(
            "What's my current charge, battery capacity, charging power, "
            "and remaining range?"
        ),
    )

    assert action is not None
    assert action.action == "respond"
    assert "can't check the vehicle charging status" in action.content
    assert "percentage range" not in action.content


def test_route_energy_does_not_capture_navigation_delete_request() -> None:
    controller = PolicyAwareController()
    tools = route_energy_tools() + [
        fake_tool(
            "navigation_delete_destination",
            {"destination_id_to_delete": {"type": "string"}},
            ["destination_id_to_delete"],
        )
    ]

    action = controller.decide(
        context_id="ctx-delete-destination",
        messages=system_messages(),
        tools=tools,
        latest_user_text=(
            "Remove Cologne from my route and make the previous stop my final "
            "destination. I don't want to add the charging station right now."
        ),
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_current_navigation_state",
            "arguments": {"detailed_information": True},
        }
    ]

    action = controller.decide(
        context_id="ctx-delete-destination",
        messages=system_messages(),
        tools=tools,
        latest_tool_results=[
            tool_result("get_current_navigation_state", multistop_navigation_result())
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "navigation_delete_destination",
            "arguments": {"destination_id_to_delete": "loc_cologne"},
        }
    ]


def test_route_energy_searches_chargers_along_current_route() -> None:
    controller = PolicyAwareController()
    tools = route_energy_tools()

    action = controller.decide(
        context_id="ctx-along",
        messages=system_messages(),
        tools=tools,
        latest_user_text=(
            "Find currently available DC charging stations along my current route "
            "around 250 km."
        ),
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_current_navigation_state",
            "arguments": {"detailed_information": True},
        }
    ]

    action = controller.decide(
        context_id="ctx-along",
        messages=system_messages(),
        tools=tools,
        latest_tool_results=[
            tool_result("get_current_navigation_state", current_navigation_result())
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "search_poi_along_the_route",
            "arguments": {
                "category_poi": "charging_stations",
                "route_id": "route_origin_mid",
                "at_kilometer": 250.0,
                "filters": [
                    "charging_stations::has_available_plug",
                    "charging_stations::has_dc_plug",
                ],
            },
        }
    ]


def test_route_energy_treats_far_current_location_offset_as_along_route() -> None:
    controller = PolicyAwareController()
    tools = route_energy_tools()

    action = controller.decide(
        context_id="ctx-current-location-offset",
        messages=system_messages(),
        tools=tools,
        latest_user_text=(
            "I have 25% charge and 112 km range. My first destination is far "
            "away, so search for charging stations around 100 km from my current "
            "location."
        ),
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_current_navigation_state",
            "arguments": {"detailed_information": True},
        }
    ]

    action = controller.decide(
        context_id="ctx-current-location-offset",
        messages=system_messages(),
        tools=tools,
        latest_tool_results=[
            tool_result("get_current_navigation_state", current_navigation_result())
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "search_poi_along_the_route",
            "arguments": {
                "category_poi": "charging_stations",
                "route_id": "route_origin_mid",
                "at_kilometer": 100.0,
            },
        }
    ]


def test_route_energy_refuses_missing_along_route_search_tool() -> None:
    controller = PolicyAwareController()
    tools = route_energy_tools(include_along_route=False)

    controller.decide(
        context_id="ctx-missing-along",
        messages=system_messages(),
        tools=tools,
        latest_user_text="Find a charging station along my current route around 90 km.",
    )
    action = controller.decide(
        context_id="ctx-missing-along",
        messages=system_messages(),
        tools=tools,
        latest_tool_results=[
            tool_result("get_current_navigation_state", current_navigation_result())
        ],
    )

    assert action is not None
    assert action.action == "respond"
    assert "can't search" in action.content.lower()


def test_route_energy_uses_recent_station_for_followup_charging_time() -> None:
    controller = PolicyAwareController()
    tools = route_energy_tools()

    controller.decide(
        context_id="ctx-time",
        messages=system_messages(),
        tools=tools,
        latest_user_text=(
            "Find available DC charging stations along my current route around 120 km."
        ),
    )
    controller.decide(
        context_id="ctx-time",
        messages=system_messages(),
        tools=tools,
        latest_tool_results=[
            tool_result("get_current_navigation_state", current_navigation_result())
        ],
    )
    action = controller.decide(
        context_id="ctx-time",
        messages=system_messages(),
        tools=tools,
        latest_tool_results=[
            tool_result("search_poi_along_the_route", charging_station_result())
        ],
    )
    assert action is not None
    assert "VoltHub" in action.content

    action = controller.decide(
        context_id="ctx-time",
        messages=system_messages(),
        tools=tools,
        latest_user_text=(
            "How long would charging to 80% take at the VoltHub charging station?"
        ),
    )
    assert action is not None
    assert action.tool_calls == [
        {"tool_name": "get_charging_specs_and_status", "arguments": {}}
    ]

    action = controller.decide(
        context_id="ctx-time",
        messages=system_messages(),
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_charging_specs_and_status",
                {"state_of_charge": 60.0, "remaining_range": "300km"},
            )
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "calculate_charging_time_by_soc",
            "arguments": {
                "charging_station_id": "poi_charge_alpha",
                "charging_station_plug_id": "plug_dc_fast",
                "start_state_of_charge": 35.0,
                "target_state_of_charge": 80.0,
            },
        }
    ]


def test_route_energy_does_not_reuse_nearby_station_for_new_along_route_search() -> None:
    controller = PolicyAwareController()
    tools = route_energy_tools()

    controller.decide(
        context_id="ctx-nearby-then-along",
        messages=system_messages(),
        tools=tools,
        latest_user_text="Find a charging station nearby before I start.",
    )
    action = controller.decide(
        context_id="ctx-nearby-then-along",
        messages=system_messages(),
        tools=tools,
        latest_tool_results=[
            tool_result(
                "search_poi_at_location",
                {
                    "pois_found": [
                        {
                            "id": "poi_greenway",
                            "name": "GreenWay",
                            "charging_plugs": [
                                {
                                    "plug_id": "plug_greenway",
                                    "power_type": "DC",
                                    "power_kw": 300,
                                    "availability": "available",
                                }
                            ],
                        }
                    ]
                },
            )
        ],
    )
    assert action is not None
    assert "GreenWay" in action.content

    action = controller.decide(
        context_id="ctx-nearby-then-along",
        messages=system_messages(),
        tools=tools,
        latest_user_text=(
            "My first destination is far away, so search for charging stations "
            "around 100 km from my current location. How long would charging to "
            "80% take there?"
        ),
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_current_navigation_state",
            "arguments": {"detailed_information": True},
        }
    ]

    action = controller.decide(
        context_id="ctx-nearby-then-along",
        messages=system_messages(),
        tools=tools,
        latest_tool_results=[
            tool_result("get_current_navigation_state", current_navigation_result())
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {"tool_name": "get_charging_specs_and_status", "arguments": {}}
    ]

    action = controller.decide(
        context_id="ctx-nearby-then-along",
        messages=system_messages(),
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_charging_specs_and_status",
                {"state_of_charge": 25.0, "remaining_range": "112km"},
            )
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "search_poi_along_the_route",
            "arguments": {
                "category_poi": "charging_stations",
                "route_id": "route_origin_mid",
                "at_kilometer": 100.0,
            },
        }
    ]


def test_route_energy_negated_navigation_setup_still_searches_along_route() -> None:
    controller = PolicyAwareController()
    tools = route_energy_tools()

    action = controller.decide(
        context_id="ctx-negated-navigation-setup",
        messages=system_messages(),
        tools=tools,
        latest_user_text=(
            "Find charging stations along my route around 350 kilometers from "
            "here. The first one sounds good, but I don't want to add it to my "
            "navigation."
        ),
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_current_navigation_state",
            "arguments": {"detailed_information": True},
        }
    ]

    action = controller.decide(
        context_id="ctx-negated-navigation-setup",
        messages=system_messages(),
        tools=tools,
        latest_tool_results=[
            tool_result("get_current_navigation_state", current_navigation_result())
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "search_poi_along_the_route",
            "arguments": {
                "category_poi": "charging_stations",
                "route_id": "route_origin_mid",
                "at_kilometer": 350.0,
            },
        }
    ]


def test_route_energy_checks_status_before_charging_time_with_user_soc() -> None:
    controller = PolicyAwareController()
    tools = route_energy_tools()

    action = controller.decide(
        context_id="ctx-user-soc-charging-time",
        messages=system_messages(),
        tools=tools,
        latest_user_text=(
            "My battery is at 25% charge, with 112 km range. I need charging "
            "stations along the way around 100 km from now. Can you find EV+ "
            "charging stations and tell me how long it would take to charge to 80%?"
        ),
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_current_navigation_state",
            "arguments": {"detailed_information": True},
        }
    ]

    action = controller.decide(
        context_id="ctx-user-soc-charging-time",
        messages=system_messages(),
        tools=tools,
        latest_tool_results=[
            tool_result("get_current_navigation_state", current_navigation_result())
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "search_poi_along_the_route",
            "arguments": {
                "category_poi": "charging_stations",
                "route_id": "route_origin_mid",
                "at_kilometer": 100.0,
            },
        }
    ]

    action = controller.decide(
        context_id="ctx-user-soc-charging-time",
        messages=system_messages(),
        tools=tools,
        latest_tool_results=[
            tool_result("search_poi_along_the_route", charging_station_result())
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {"tool_name": "get_charging_specs_and_status", "arguments": {}}
    ]


def test_route_energy_sets_navigation_with_recent_charging_stop() -> None:
    controller = PolicyAwareController()
    tools = route_energy_tools(include_route_lookup=True) + [
        fake_tool(
            "set_new_navigation",
            {"route_ids": {"type": "array"}},
            ["route_ids"],
        )
    ]

    controller.decide(
        context_id="ctx-nav-with-charging-stop",
        messages=system_messages(),
        tools=tools,
        latest_user_text="Find a charging station nearby before I start.",
    )
    action = controller.decide(
        context_id="ctx-nav-with-charging-stop",
        messages=system_messages(),
        tools=tools,
        latest_tool_results=[
            tool_result(
                "search_poi_at_location",
                {
                    "pois_found": charging_station_result()[
                        "pois_found_along_route"
                    ]
                },
            )
        ],
    )
    assert action is not None
    assert action.action == "respond"
    assert "VoltHub" in action.content

    action = controller.decide(
        context_id="ctx-nav-with-charging-stop",
        messages=system_messages(),
        tools=tools,
        latest_user_text=(
            "Set up navigation to Harbor City with the charging stop at "
            "VoltHub. Take the fastest route to the charging station, and for "
            "Harbor City use the second route option via B2."
        ),
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_location_id_by_location_name",
            "arguments": {"location": "Harbor"},
        }
    ]

    action = controller.decide(
        context_id="ctx-nav-with-charging-stop",
        messages=system_messages(),
        tools=tools,
        latest_tool_results=[
            tool_result("get_location_id_by_location_name", {"id": "loc_harbor"})
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_routes_from_start_to_destination",
            "arguments": {
                "start_id": "loc_origin",
                "destination_id": "poi_charge_alpha",
            },
        }
    ]

    action = controller.decide(
        context_id="ctx-nav-with-charging-stop",
        messages=system_messages(),
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_routes_from_start_to_destination",
                {
                    "routes": [
                        {"route_id": "route_origin_charge_fast", "alias": ["fastest"]},
                        {"route_id": "route_origin_charge_short", "alias": ["shortest"]},
                    ]
                },
            )
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_routes_from_start_to_destination",
            "arguments": {
                "start_id": "poi_charge_alpha",
                "destination_id": "loc_harbor",
            },
        }
    ]

    action = controller.decide(
        context_id="ctx-nav-with-charging-stop",
        messages=system_messages(),
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_routes_from_start_to_destination",
                {
                    "routes": [
                        {"route_id": "route_charge_harbor_fast", "alias": ["fastest"]},
                        {
                            "route_id": "route_charge_harbor_second",
                            "alias": ["second"],
                            "name_via": "B2",
                        },
                    ]
                },
            )
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "set_new_navigation",
            "arguments": {
                "route_ids": [
                    "route_origin_charge_fast",
                    "route_charge_harbor_second",
                ]
            },
        }
    ]


def test_route_energy_sets_navigation_with_food_verified_charging_stop() -> None:
    controller = PolicyAwareController()
    tools = route_energy_tools(include_route_lookup=True) + [
        fake_tool(
            "set_new_navigation",
            {"route_ids": {"type": "array"}},
            ["route_ids"],
        )
    ]
    messages = system_messages(location_id="loc_origin", hour=17, minute=0)

    action = controller.decide(
        context_id="ctx-food-charging-stop",
        messages=messages,
        tools=tools,
        latest_user_text=(
            "Set up navigation to Northport. I need a charging station "
            "stop on the way between 6 PM and 6:10 PM where I can eat fast food "
            "that's open."
        ),
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_location_id_by_location_name",
            "arguments": {"location": "Northport"},
        }
    ]

    action = controller.decide(
        context_id="ctx-food-charging-stop",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result("get_location_id_by_location_name", {"id": "loc_northport"})
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_routes_from_start_to_destination",
            "arguments": {
                "start_id": "loc_origin",
                "destination_id": "loc_northport",
            },
        }
    ]

    action = controller.decide(
        context_id="ctx-food-charging-stop",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_routes_from_start_to_destination",
                {
                    "routes": [
                        {
                            "route_id": "route_origin_northport_fast",
                            "distance_km": 300.0,
                            "duration_hours": 3,
                            "duration_minutes": 0,
                            "alias": ["fastest", "first"],
                        },
                        {
                            "route_id": "route_origin_northport_second",
                            "distance_km": 320.0,
                            "alias": ["second"],
                        },
                    ]
                },
            )
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "search_poi_along_the_route",
            "arguments": {
                "category_poi": "fast_food",
                "route_id": "route_origin_northport_fast",
                "at_kilometer": 100,
            },
        }
    ]

    action = controller.decide(
        context_id="ctx-food-charging-stop",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result("search_poi_along_the_route", fast_food_result())
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "search_poi_along_the_route",
            "arguments": {
                "category_poi": "charging_stations",
                "route_id": "route_origin_northport_fast",
                "at_kilometer": 100,
            },
        }
    ]

    action = controller.decide(
        context_id="ctx-food-charging-stop",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result("search_poi_along_the_route", charging_station_result())
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_routes_from_start_to_destination",
            "arguments": {
                "start_id": "loc_origin",
                "destination_id": "poi_charge_alpha",
            },
        }
    ]

    action = controller.decide(
        context_id="ctx-food-charging-stop",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_routes_from_start_to_destination",
                {"routes": [{"route_id": "route_origin_charge", "alias": ["fastest"]}]},
            )
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_routes_from_start_to_destination",
            "arguments": {
                "start_id": "poi_charge_alpha",
                "destination_id": "loc_northport",
            },
        }
    ]

    action = controller.decide(
        context_id="ctx-food-charging-stop",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_routes_from_start_to_destination",
                {"routes": [{"route_id": "route_charge_northport", "alias": ["fastest"]}]},
            )
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "set_new_navigation",
            "arguments": {
                "route_ids": ["route_origin_charge", "route_charge_northport"]
            },
        }
    ]


def test_route_energy_preserves_selected_station_after_charging_time_followup() -> None:
    controller = PolicyAwareController()
    tools = route_energy_tools(include_route_lookup=True) + [
        fake_tool(
            "set_new_navigation",
            {"route_ids": {"type": "array"}},
            ["route_ids"],
        )
    ]
    volthub = charging_station_result()["pois_found_along_route"][0]
    beta_station = {
        "id": "poi_charge_beta",
        "name": "ChargeMax",
        "phone_number": "+49 222 333444",
        "charging_plugs": [
            {
                "plug_id": "plug_beta",
                "power_type": "DC",
                "power_kw": 150,
                "availability": "available",
            }
        ],
    }

    controller.decide(
        context_id="ctx-selected-stop-persisted",
        messages=system_messages(),
        tools=tools,
        latest_user_text="Find a charging station nearby before I start.",
    )
    action = controller.decide(
        context_id="ctx-selected-stop-persisted",
        messages=system_messages(),
        tools=tools,
        latest_tool_results=[
            tool_result(
                "search_poi_at_location",
                {"pois_found": [volthub, beta_station]},
            )
        ],
    )
    assert action is not None
    assert "VoltHub" in action.content

    action = controller.decide(
        context_id="ctx-selected-stop-persisted",
        messages=system_messages(),
        tools=tools,
        latest_user_text=(
            "Yes, choose VoltHub. Calculate how long charging will take if I "
            "charge to 80%."
        ),
    )
    assert action is not None
    assert action.tool_calls == [
        {"tool_name": "get_charging_specs_and_status", "arguments": {}}
    ]

    action = controller.decide(
        context_id="ctx-selected-stop-persisted",
        messages=system_messages(),
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_charging_specs_and_status",
                {"state_of_charge": 35.0, "remaining_range": "155km"},
            )
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "calculate_charging_time_by_soc",
            "arguments": {
                "charging_station_id": "poi_charge_alpha",
                "charging_station_plug_id": "plug_dc_fast",
                "start_state_of_charge": 35.0,
                "target_state_of_charge": 80.0,
            },
        }
    ]

    action = controller.decide(
        context_id="ctx-selected-stop-persisted",
        messages=system_messages(),
        tools=tools,
        latest_tool_results=[
            tool_result(
                "calculate_charging_time_by_soc",
                {"time_from_35_until_80_percent_soc": "20min"},
            )
        ],
    )
    assert action is not None
    assert action.action == "respond"

    action = controller.decide(
        context_id="ctx-selected-stop-persisted",
        messages=system_messages(),
        tools=tools,
        latest_user_text=(
            "Set up navigation to VoltHub with the fastest route, and then to "
            "Harbor City using the second route option, via B2."
        ),
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_location_id_by_location_name",
            "arguments": {"location": "Harbor"},
        }
    ]

    action = controller.decide(
        context_id="ctx-selected-stop-persisted",
        messages=system_messages(),
        tools=tools,
        latest_tool_results=[
            tool_result("get_location_id_by_location_name", {"id": "loc_harbor"})
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_routes_from_start_to_destination",
            "arguments": {
                "start_id": "loc_origin",
                "destination_id": "poi_charge_alpha",
            },
        }
    ]
