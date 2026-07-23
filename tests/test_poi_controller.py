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


def system_messages(location_id: str = "loc_origin") -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                f'CURRENT_LOCATION = {{"id": "{location_id}", "name": "Origin City"}}\n'
                'DATETIME = {"year": 2026, "month": 6, "day": 17, '
                '"hour": 11, "minute": 45}'
            ),
        }
    ]


def poi_tools() -> list[dict]:
    return [
        fake_tool(
            "get_current_navigation_state",
            {"detailed_information": {"type": "boolean"}},
        ),
        fake_tool(
            "get_location_id_by_location_name",
            {"location": {"type": "string"}},
            ["location"],
        ),
        fake_tool(
            "search_poi_at_location",
            {
                "category_poi": {"type": "string"},
                "location_id": {"type": "string"},
            },
            ["category_poi", "location_id"],
        ),
        fake_tool(
            "get_routes_from_start_to_destination",
            {
                "start_id": {"type": "string"},
                "destination_id": {"type": "string"},
            },
            ["start_id", "destination_id"],
        ),
        fake_tool(
            "navigation_replace_final_destination",
            {
                "new_destination_id": {"type": "string"},
                "route_id_leading_to_new_destination": {"type": "string"},
            },
            ["new_destination_id", "route_id_leading_to_new_destination"],
        ),
        fake_tool(
            "navigation_delete_destination",
            {"destination_id_to_delete": {"type": "string"}},
            ["destination_id_to_delete"],
        ),
    ]


def restaurant_results() -> dict:
    return {
        "pois_found": [
            {
                "id": "poi_option_one",
                "name": "Harbor Grill",
                "opening_hours": "10:00 AM - 4:00 PM",
            },
            {
                "id": "poi_option_two",
                "name": "Maple Kitchen",
                "opening_hours": "11:00 AM - 5:00 PM",
            },
        ]
    }


def route_results() -> dict:
    return {
        "routes": [
            {
                "route_id": "route_option_fast",
                "alias": ["fastest"],
                "name_via": "R10, C20",
                "distance_km": 620,
                "duration_hours": 6,
                "duration_minutes": 10,
            },
            {
                "route_id": "route_option_second",
                "alias": ["second"],
                "name_via": "N31, Q42, M7",
                "distance_km": 640,
                "duration_hours": 6,
                "duration_minutes": 45,
            },
        ]
    }


def test_poi_controller_replaces_destination_with_selected_restaurant_route() -> None:
    controller = PolicyAwareController()
    messages = system_messages()
    tools = poi_tools()

    action = controller.decide(
        context_id="ctx-poi-destination",
        messages=messages,
        tools=tools,
        latest_user_text="Change my destination to Harbor Bay. Find a restaurant there.",
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_location_id_by_location_name",
            "arguments": {"location": "Harbor Bay"},
        }
    ]

    action = controller.decide(
        context_id="ctx-poi-destination",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result("get_location_id_by_location_name", {"id": "loc_harbor_bay"})
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "search_poi_at_location",
            "arguments": {
                "category_poi": "restaurants",
                "location_id": "loc_harbor_bay",
            },
        }
    ]

    action = controller.decide(
        context_id="ctx-poi-destination",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result("search_poi_at_location", restaurant_results())
        ],
    )
    assert action is not None
    assert action.action == "respond"
    assert "Maple Kitchen" in action.content
    assert "11:00h - 17:00h" in action.content

    action = controller.decide(
        context_id="ctx-poi-destination",
        messages=messages,
        tools=tools,
        latest_user_text="Maple Kitchen.",
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_routes_from_start_to_destination",
            "arguments": {
                "start_id": "loc_origin",
                "destination_id": "poi_option_two",
            },
        }
    ]

    action = controller.decide(
        context_id="ctx-poi-destination",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result("get_routes_from_start_to_destination", route_results())
        ],
    )
    assert action is not None
    assert action.action == "respond"
    assert "N31, Q42, M7" in action.content

    action = controller.decide(
        context_id="ctx-poi-destination",
        messages=messages,
        tools=tools,
        latest_user_text="Take the route via N31, Q42, M7.",
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "navigation_replace_final_destination",
            "arguments": {
                "new_destination_id": "poi_option_two",
                "route_id_leading_to_new_destination": "route_option_second",
            },
        }
    ]


def test_poi_controller_nearby_options_reports_route_without_setting_navigation() -> None:
    controller = PolicyAwareController()
    messages = system_messages(location_id="loc_current")
    tools = poi_tools()

    action = controller.decide(
        context_id="ctx-poi-nearby",
        messages=messages,
        tools=tools,
        latest_user_text="What restaurant options are available nearby?",
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "search_poi_at_location",
            "arguments": {
                "category_poi": "restaurants",
                "location_id": "loc_current",
            },
        }
    ]

    controller.decide(
        context_id="ctx-poi-nearby",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result("search_poi_at_location", restaurant_results())
        ],
    )
    action = controller.decide(
        context_id="ctx-poi-nearby",
        messages=messages,
        tools=tools,
        latest_user_text="How long would it take to get to the first one?",
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_routes_from_start_to_destination",
            "arguments": {
                "start_id": "loc_current",
                "destination_id": "poi_option_one",
            },
        }
    ]

    action = controller.decide(
        context_id="ctx-poi-nearby",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_routes_from_start_to_destination",
                {"routes": [route_results()["routes"][0]]},
            )
        ],
    )
    assert action is not None
    assert action.action == "respond"
    assert "selected route" in action.content
    assert action.tool_calls == []


def test_poi_flow_yields_to_navigation_delete_destination_followup() -> None:
    controller = PolicyAwareController()
    messages = system_messages(location_id="loc_current")
    tools = poi_tools()

    controller.decide(
        context_id="ctx-poi-then-delete-destination",
        messages=messages,
        tools=tools,
        latest_user_text="Can you show me some restaurant options around here?",
    )
    controller.decide(
        context_id="ctx-poi-then-delete-destination",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result("search_poi_at_location", restaurant_results())
        ],
    )
    controller.decide(
        context_id="ctx-poi-then-delete-destination",
        messages=messages,
        tools=tools,
        latest_user_text="How long would it take to get to the first one?",
    )
    controller.decide(
        context_id="ctx-poi-then-delete-destination",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_routes_from_start_to_destination",
                {"routes": [route_results()["routes"][0], route_results()["routes"][1]]},
            )
        ],
    )

    action = controller.decide(
        context_id="ctx-poi-then-delete-destination",
        messages=messages,
        tools=tools,
        latest_user_text=(
            "I don't want to go there. Cancel Riverton as my final destination "
            "and make Lakeside the end of my trip."
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
        context_id="ctx-poi-then-delete-destination",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_current_navigation_state",
                {
                    "navigation_active": True,
                    "waypoints_id": [
                        "loc_current",
                        "loc_lakeside",
                        "loc_riverton",
                    ],
                    "routes_to_final_destination_id": [
                        "route_current_lakeside",
                        "route_lakeside_riverton",
                    ],
                    "details": {
                        "waypoints": [
                            {"id": "loc_current", "name": "Origin City"},
                            {"id": "loc_lakeside", "name": "Lakeside"},
                            {"id": "loc_riverton", "name": "Riverton"},
                        ]
                    },
                },
            )
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "navigation_delete_destination",
            "arguments": {"destination_id_to_delete": "loc_riverton"},
        }
    ]
