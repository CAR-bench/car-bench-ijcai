import json

from carbench_agent_core import PolicyAwareController
from carbench_agent_core.actions import NextAction
from carbench_agent_core.flows import ControllerState
from carbench_agent_core.tool_index import ToolIndex


def fake_tool(
    name: str,
    *,
    properties: dict | None = None,
    required: list[str] | None = None,
    additional_properties: bool | None = None,
) -> dict:
    tool = {
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
    if additional_properties is not None:
        tool["function"]["parameters"]["additionalProperties"] = additional_properties
    return tool


def tool_result(tool_name: str, result: dict) -> dict:
    return {
        "tool_name": tool_name,
        "content": json.dumps({"status": "SUCCESS", "result": result}),
    }


def system_messages(location_id: str = "loc_bonn_001") -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                f'CURRENT_LOCATION = {{"id": "{location_id}", "name": "Bonn"}}\n'
                'DATETIME = {"year": 2026, "month": 6, "day": 17, '
                '"hour": 9, "minute": 0}'
            ),
        }
    ]


def navigation_tools() -> list[dict]:
    return [
        fake_tool("get_current_navigation_state"),
        fake_tool("get_location_id_by_location_name"),
        fake_tool("get_routes_from_start_to_destination"),
        fake_tool("set_new_navigation"),
        fake_tool("navigation_replace_final_destination"),
        fake_tool("navigation_delete_destination"),
        fake_tool("navigation_delete_waypoint"),
        fake_tool("navigation_replace_one_waypoint"),
    ]


class UnverifiedCompletionController(PolicyAwareController):
    def _next_sunroof_action(
        self, state: ControllerState, tool_index: ToolIndex
    ) -> NextAction:
        return NextAction.respond("Done.", reason="sunroof_done")


def test_controller_tracks_obligations_until_tool_completion() -> None:
    controller = PolicyAwareController()
    tools = [
        fake_tool(
            "open_close_trunk_door",
            properties={"action": {"type": "string"}},
            required=["action"],
        )
    ]

    confirmation = controller.decide(
        context_id="ctx-obligation",
        messages=system_messages(),
        tools=tools,
        latest_user_text="Open the trunk.",
    )
    assert confirmation is not None
    assert confirmation.reason == "trunk_confirmation"
    assert controller.pending_obligations("ctx-obligation") == ("trunk",)

    tool_call = controller.decide(
        context_id="ctx-obligation",
        messages=system_messages(),
        tools=tools,
        latest_user_text="Yes.",
    )
    assert tool_call is not None
    assert tool_call.reason == "trunk_action"
    assert controller.pending_obligations("ctx-obligation") == ("trunk",)

    completion = controller.decide(
        context_id="ctx-obligation",
        messages=system_messages(),
        tools=tools,
        latest_tool_results=[tool_result("open_close_trunk_door", {"action": "OPEN"})],
    )
    assert completion is not None
    assert completion.reason == "trunk_done"
    assert controller.pending_obligations("ctx-obligation") == ()


def test_controller_rejects_unverified_completion_claim() -> None:
    controller = UnverifiedCompletionController()
    state = ControllerState()
    state.sunroof.active = True
    controller._states["ctx-unverified"] = state

    action = controller.decide(
        context_id="ctx-unverified",
        messages=system_messages(),
        tools=[],
    )

    assert action is not None
    assert action.reason == "sunroof_done_evidence_guard"
    assert "couldn't verify" in action.content


def test_navigation_controller_sets_fastest_single_destination_route() -> None:
    controller = PolicyAwareController()
    messages = system_messages()
    tools = navigation_tools()

    action = controller.decide(
        context_id="ctx-nav",
        messages=messages,
        tools=tools,
        latest_user_text="Please navigate to Munich on the fastest route.",
    )
    assert action is not None
    assert action.action == "tool_calls"
    assert action.tool_calls == [
        {
            "tool_name": "get_location_id_by_location_name",
            "arguments": {"location": "Munich"},
        }
    ]

    action = controller.decide(
        context_id="ctx-nav",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result("get_location_id_by_location_name", {"id": "loc_mun_002"})
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_routes_from_start_to_destination",
            "arguments": {
                "start_id": "loc_bonn_001",
                "destination_id": "loc_mun_002",
            },
        }
    ]

    action = controller.decide(
        context_id="ctx-nav",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_routes_from_start_to_destination",
                {
                    "routes": [
                        {
                            "route_id": "rll_bonn_mun_fast",
                            "alias": ["first", "fastest"],
                            "distance_km": 520,
                            "duration_hours": 5,
                            "duration_minutes": 30,
                        },
                        {
                            "route_id": "rll_bonn_mun_short",
                            "alias": ["second", "shortest"],
                            "distance_km": 500,
                            "duration_hours": 6,
                            "duration_minutes": 15,
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
            "arguments": {"route_ids": ["rll_bonn_mun_fast"]},
        }
    ]


def test_navigation_controller_sets_two_leg_navigation_route() -> None:
    controller = PolicyAwareController()
    messages = system_messages(location_id="loc_berlin")
    tools = navigation_tools()

    action = controller.decide(
        context_id="ctx-multi-stop-nav",
        messages=messages,
        tools=tools,
        latest_user_text=(
            "Navigate from Berlin to Bremen, then to Amsterdam. "
            "I need the fastest route."
        ),
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_location_id_by_location_name",
            "arguments": {"location": "Bremen"},
        }
    ]

    action = controller.decide(
        context_id="ctx-multi-stop-nav",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result("get_location_id_by_location_name", {"id": "loc_bremen"})
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_location_id_by_location_name",
            "arguments": {"location": "Amsterdam"},
        }
    ]

    action = controller.decide(
        context_id="ctx-multi-stop-nav",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result("get_location_id_by_location_name", {"id": "loc_amsterdam"})
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_routes_from_start_to_destination",
            "arguments": {
                "start_id": "loc_berlin",
                "destination_id": "loc_bremen",
            },
        }
    ]

    action = controller.decide(
        context_id="ctx-multi-stop-nav",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_routes_from_start_to_destination",
                {
                    "routes": [
                        {"route_id": "route_ber_bre_fast", "alias": ["fastest"]},
                        {"route_id": "route_ber_bre_short", "alias": ["shortest"]},
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
                "start_id": "loc_bremen",
                "destination_id": "loc_amsterdam",
            },
        }
    ]

    action = controller.decide(
        context_id="ctx-multi-stop-nav",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_routes_from_start_to_destination",
                {
                    "routes": [
                        {"route_id": "route_bre_ams_fast", "alias": ["fastest"]},
                        {"route_id": "route_bre_ams_short", "alias": ["shortest"]},
                    ]
                },
            )
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "set_new_navigation",
            "arguments": {"route_ids": ["route_ber_bre_fast", "route_bre_ams_fast"]},
        }
    ]

    action = controller.decide(
        context_id="ctx-multi-stop-nav",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result(
                "set_new_navigation",
                {
                    "navigation_set": True,
                    "waypoints": ["loc_berlin", "loc_bremen", "loc_amsterdam"],
                },
            )
        ],
    )
    assert action is not None
    assert action.action == "respond"
    assert "would you like more information" in action.content.lower()


def test_planned_navigation_followup_poi_search_does_not_start_old_route() -> None:
    controller = PolicyAwareController()
    messages = system_messages(location_id="loc_sofia")
    tools = navigation_tools() + [fake_tool("search_poi_at_location")]
    context_id = "ctx-planned-route-poi-followup"

    action = controller.decide(
        context_id=context_id,
        messages=messages,
        tools=tools,
        latest_user_text=(
            "I want to plan a trip. First, I need to go to Belgrade. "
            "Can you find me fastest route there, but without toll roads if it's "
            "not more than 10 minutes longer than the fastest route with tolls?"
        ),
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_location_id_by_location_name",
            "arguments": {"location": "Belgrade"},
        }
    ]

    action = controller.decide(
        context_id=context_id,
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result("get_location_id_by_location_name", {"id": "loc_belgrade"})
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_routes_from_start_to_destination",
            "arguments": {
                "start_id": "loc_sofia",
                "destination_id": "loc_belgrade",
            },
        }
    ]

    action = controller.decide(
        context_id=context_id,
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_routes_from_start_to_destination",
                {
                    "routes": [
                        {
                            "route_id": "route_sof_bel_fast_toll",
                            "alias": ["fastest"],
                            "duration_hours": 4,
                            "duration_minutes": 0,
                            "includes_toll": True,
                        },
                        {
                            "route_id": "route_sof_bel_no_toll",
                            "alias": ["shortest"],
                            "duration_hours": 4,
                            "duration_minutes": 5,
                            "includes_toll": False,
                        },
                    ]
                },
            )
        ],
    )
    assert action is not None
    assert action.action == "respond"

    action = controller.decide(
        context_id=context_id,
        messages=messages,
        tools=tools,
        latest_user_text=(
            "Okay, after Belgrade, I need to go to a supermarket that's still open. "
            "What's the fastest route for that?"
        ),
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "search_poi_at_location",
            "arguments": {
                "category_poi": "supermarkets",
                "location_id": "loc_belgrade",
            },
        }
    ]

    action = controller.decide(
        context_id=context_id,
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result(
                "search_poi_at_location",
                {
                    "pois_found": [
                        {
                            "id": "poi_market_open",
                            "name": "Billa",
                            "category": "supermarkets",
                            "opening_hours": "06:00h - 21:00h",
                        }
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
                "start_id": "loc_belgrade",
                "destination_id": "poi_market_open",
            },
        }
    ]

    action = controller.decide(
        context_id=context_id,
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_routes_from_start_to_destination",
                {
                    "routes": [
                        {"route_id": "route_bel_market_fast", "alias": ["fastest"]},
                        {"route_id": "route_bel_market_second", "alias": ["second"]},
                    ]
                },
            )
        ],
    )
    assert action is not None
    assert action.action == "respond"
    assert "fastest route segment includes toll roads" in action.content.lower()
    assert "do you want me to start navigation" in action.content.lower()


def test_navigation_controller_asks_when_route_choice_is_ambiguous() -> None:
    controller = PolicyAwareController()
    messages = system_messages()
    tools = navigation_tools()

    controller.decide(
        context_id="ctx-choice",
        messages=messages,
        tools=tools,
        latest_user_text="Please navigate to Munich.",
    )
    controller.decide(
        context_id="ctx-choice",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result("get_location_id_by_location_name", {"id": "loc_mun_002"})
        ],
    )

    action = controller.decide(
        context_id="ctx-choice",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_routes_from_start_to_destination",
                {
                    "routes": [
                        {
                            "route_id": "rll_bonn_mun_fast",
                            "alias": ["fastest"],
                            "distance_km": 520,
                            "duration_hours": 5,
                            "duration_minutes": 30,
                        },
                        {
                            "route_id": "rll_bonn_mun_short",
                            "alias": ["shortest"],
                            "distance_km": 500,
                            "duration_hours": 6,
                            "duration_minutes": 15,
                            "includes_toll": True,
                        },
                    ]
                },
            )
        ],
    )
    assert action is not None
    assert action.action == "respond"
    assert "fastest" in action.content.lower()
    assert "shortest" in action.content.lower()
    assert "toll" in action.content.lower()

    action = controller.decide(
        context_id="ctx-choice",
        messages=messages,
        tools=tools,
        latest_user_text="Take the shortest route.",
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "set_new_navigation",
            "arguments": {"route_ids": ["rll_bonn_mun_short"]},
        }
    ]


def test_navigation_controller_replaces_final_destination_with_route_evidence() -> None:
    controller = PolicyAwareController()
    messages = system_messages(location_id="loc_bochum_001")
    tools = navigation_tools()

    action = controller.decide(
        context_id="ctx-replace",
        messages=messages,
        tools=tools,
        latest_user_text="Change my navigation destination from Milan to Hamburg.",
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_location_id_by_location_name",
            "arguments": {"location": "Hamburg"},
        }
    ]

    action = controller.decide(
        context_id="ctx-replace",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result("get_location_id_by_location_name", {"id": "loc_ham_003"})
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_routes_from_start_to_destination",
            "arguments": {
                "start_id": "loc_bochum_001",
                "destination_id": "loc_ham_003",
            },
        }
    ]

    action = controller.decide(
        context_id="ctx-replace",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_routes_from_start_to_destination",
                {"routes": [{"route_id": "rll_bochum_ham", "alias": ["fastest"]}]},
            )
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "navigation_replace_final_destination",
            "arguments": {
                "new_destination_id": "loc_ham_003",
                "route_id_leading_to_new_destination": "rll_bochum_ham",
            },
        }
    ]


def test_navigation_refuses_when_route_result_is_unavailable() -> None:
    controller = PolicyAwareController()
    messages = system_messages(location_id="loc_bochum_001")
    tools = navigation_tools()

    controller.decide(
        context_id="ctx-route-result-unavailable",
        messages=messages,
        tools=tools,
        latest_user_text="Change my navigation destination from Milan to Hamburg.",
    )
    controller.decide(
        context_id="ctx-route-result-unavailable",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result("get_location_id_by_location_name", {"id": "loc_ham_003"})
        ],
    )
    action = controller.decide(
        context_id="ctx-route-result-unavailable",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result("get_routes_from_start_to_destination", {"routes": "unknown"})
        ],
    )
    assert action is not None
    assert action.action == "respond"
    assert "can't determine the route options" in action.content.lower()


def test_navigation_followup_replaces_destination_after_change_prompt() -> None:
    controller = PolicyAwareController()
    messages = system_messages(location_id="loc_bochum_001")
    tools = navigation_tools()

    action = controller.decide(
        context_id="ctx-replace-followup",
        messages=messages,
        tools=tools,
        latest_user_text="I need to change my navigation destination. Can you help?",
    )
    assert action is None

    action = controller.decide(
        context_id="ctx-replace-followup",
        messages=messages,
        tools=tools,
        latest_user_text="I need to go to Hamburg now.",
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_location_id_by_location_name",
            "arguments": {"location": "Hamburg"},
        }
    ]

    controller.decide(
        context_id="ctx-replace-followup",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result("get_location_id_by_location_name", {"id": "loc_ham_003"})
        ],
    )
    action = controller.decide(
        context_id="ctx-replace-followup",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_routes_from_start_to_destination",
                {"routes": [{"route_id": "rll_bochum_ham", "alias": ["fastest"]}]},
            )
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "navigation_replace_final_destination",
            "arguments": {
                "new_destination_id": "loc_ham_003",
                "route_id_leading_to_new_destination": "rll_bochum_ham",
            },
        }
    ]


def test_navigation_controller_replaces_final_destination_from_route_predecessor() -> (
    None
):
    controller = PolicyAwareController()
    messages = system_messages(location_id="loc_andorra")
    tools = navigation_tools()

    action = controller.decide(
        context_id="ctx-final-predecessor",
        messages=messages,
        tools=tools,
        latest_user_text=(
            "Change my final destination from Rome to Stuttgart and use the "
            "shortest route."
        ),
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_location_id_by_location_name",
            "arguments": {"location": "Stuttgart"},
        }
    ]

    action = controller.decide(
        context_id="ctx-final-predecessor",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result("get_location_id_by_location_name", {"id": "loc_stuttgart"})
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_current_navigation_state",
            "arguments": {"detailed_information": False},
        }
    ]

    action = controller.decide(
        context_id="ctx-final-predecessor",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_current_navigation_state",
                {
                    "navigation_active": True,
                    "waypoints_id": [
                        "loc_andorra",
                        "loc_paris",
                        "loc_milan",
                        "loc_rome",
                    ],
                    "routes_to_final_destination_id": [
                        "rll_and_par",
                        "rll_par_mil",
                        "rll_mil_rom",
                    ],
                },
            )
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_routes_from_start_to_destination",
            "arguments": {
                "start_id": "loc_milan",
                "destination_id": "loc_stuttgart",
            },
        }
    ]

    action = controller.decide(
        context_id="ctx-final-predecessor",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_routes_from_start_to_destination",
                {
                    "routes": [
                        {"route_id": "rll_mil_stu_fast", "alias": ["fastest"]},
                        {"route_id": "rll_mil_stu_short", "alias": ["shortest"]},
                    ]
                },
            )
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "navigation_replace_final_destination",
            "arguments": {
                "new_destination_id": "loc_stuttgart",
                "route_id_leading_to_new_destination": "rll_mil_stu_short",
            },
        }
    ]


def test_navigation_controller_uses_second_route_after_user_choice() -> None:
    controller = PolicyAwareController()
    messages = system_messages(location_id="loc_dortmund")
    tools = navigation_tools()

    controller.decide(
        context_id="ctx-second-route",
        messages=messages,
        tools=tools,
        latest_user_text=(
            "I've changed my mind about going to Düsseldorf. Can you change "
            "my navigation to Dresden instead?"
        ),
    )
    controller.decide(
        context_id="ctx-second-route",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result("get_location_id_by_location_name", {"id": "loc_dresden"})
        ],
    )
    action = controller.decide(
        context_id="ctx-second-route",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_routes_from_start_to_destination",
                {
                    "routes": [
                        {"route_id": "rll_dor_dre_first", "alias": ["first"]},
                        {"route_id": "rll_dor_dre_second", "alias": ["second"]},
                    ]
                },
            )
        ],
    )
    assert action is not None
    assert action.action == "respond"
    assert "Second route" in action.content

    action = controller.decide(
        context_id="ctx-second-route",
        messages=messages,
        tools=tools,
        latest_user_text="Can you show me another route option?",
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "navigation_replace_final_destination",
            "arguments": {
                "new_destination_id": "loc_dresden",
                "route_id_leading_to_new_destination": "rll_dor_dre_second",
            },
        }
    ]

    action = controller.decide(
        context_id="ctx-second-route",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result(
                "navigation_replace_final_destination",
                {
                    "destination_replaced": True,
                    "new_waypoints": ["loc_dortmund", "loc_dresden"],
                    "new_routes": ["rll_dor_dre_second"],
                },
            )
        ],
    )
    assert action is not None
    assert "second route" in action.content.lower()


def test_navigation_controller_deletes_final_destination_from_recent_lookup() -> None:
    controller = PolicyAwareController()
    messages = system_messages(location_id="loc_duisburg") + [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_lookup_barcelona",
                    "type": "function",
                    "function": {
                        "name": "get_location_id_by_location_name",
                        "arguments": '{"location": "Barcelona"}',
                    },
                }
            ],
        }
    ]
    tools = navigation_tools()

    action = controller.decide(
        context_id="ctx-delete-final-known",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result("get_location_id_by_location_name", {"id": "loc_barcelona"})
        ],
    )
    assert action is None

    action = controller.decide(
        context_id="ctx-delete-final-known",
        messages=system_messages(location_id="loc_duisburg"),
        tools=tools,
        latest_user_text=(
            "I don't need to go to Barcelona anymore. Please delete Barcelona "
            "from my current navigation route so Hamburg is the final destination."
        ),
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "navigation_delete_destination",
            "arguments": {"destination_id_to_delete": "loc_barcelona"},
        }
    ]


def test_navigation_controller_deletes_final_destination_from_current_state() -> None:
    controller = PolicyAwareController()
    messages = system_messages(location_id="loc_wiesbaden")
    tools = navigation_tools()

    action = controller.decide(
        context_id="ctx-delete-final-state",
        messages=messages,
        tools=tools,
        latest_user_text="Cancel my final destination, Monaco, and end my trip at Stuttgart.",
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_current_navigation_state",
            "arguments": {"detailed_information": True},
        }
    ]

    action = controller.decide(
        context_id="ctx-delete-final-state",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_current_navigation_state",
                {
                    "navigation_active": True,
                    "waypoints_id": [
                        "loc_wiesbaden",
                        "loc_stuttgart",
                        "loc_monaco",
                    ],
                    "routes_to_final_destination_id": [
                        "rll_wie_stu",
                        "rll_stu_mon",
                    ],
                    "details": {
                        "waypoints": [
                            {"id": "loc_wiesbaden", "name": "Wiesbaden"},
                            {"id": "loc_stuttgart", "name": "Stuttgart"},
                            {"id": "loc_monaco", "name": "Monaco"},
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
            "arguments": {"destination_id_to_delete": "loc_monaco"},
        }
    ]


def test_controller_blocks_tool_call_when_current_schema_rejects_arguments() -> None:
    controller = PolicyAwareController()
    messages = system_messages(location_id="loc_wiesbaden")
    tools = [
        fake_tool("get_current_navigation_state"),
        fake_tool(
            "navigation_delete_destination",
            properties={"different_argument": {"type": "string"}},
            additional_properties=False,
        ),
    ]

    action = controller.decide(
        context_id="ctx-delete-final-guard",
        messages=messages,
        tools=tools,
        latest_user_text="Cancel my final destination, Monaco, and end my trip at Stuttgart.",
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_current_navigation_state",
            "arguments": {"detailed_information": True},
        }
    ]

    action = controller.decide(
        context_id="ctx-delete-final-guard",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_current_navigation_state",
                {
                    "navigation_active": True,
                    "waypoints_id": [
                        "loc_wiesbaden",
                        "loc_stuttgart",
                        "loc_monaco",
                    ],
                    "details": {
                        "waypoints": [
                            {"id": "loc_wiesbaden", "name": "Wiesbaden"},
                            {"id": "loc_stuttgart", "name": "Stuttgart"},
                            {"id": "loc_monaco", "name": "Monaco"},
                        ]
                    },
                },
            )
        ],
    )
    assert action is not None
    assert action.action == "respond"
    assert (
        action.content == "I can't complete this request with the available controls."
    )
    assert "destination_id_to_delete" not in action.content


def test_navigation_followup_reuses_observed_waypoints_after_external_route_edit() -> (
    None
):
    controller = PolicyAwareController()
    messages = system_messages(location_id="loc_andorra")
    tools = navigation_tools()

    controller.decide(
        context_id="ctx-followup-observed-waypoints",
        messages=messages,
        tools=tools,
        latest_user_text="I need to change my final destination.",
    )
    controller.decide(
        context_id="ctx-followup-observed-waypoints",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result(
                "navigation_replace_final_destination",
                {
                    "destination_replaced": True,
                    "new_waypoints": [
                        "loc_andorra",
                        "loc_paris",
                        "loc_milan",
                        "loc_stuttgart",
                    ],
                    "new_routes": [
                        "rll_and_par",
                        "rll_par_mil",
                        "rll_mil_stu",
                    ],
                },
            )
        ],
    )

    action = controller.decide(
        context_id="ctx-followup-observed-waypoints",
        messages=messages,
        tools=tools,
        latest_user_text=(
            "Oh, I meant Munich. Can you find the shortest route to Munich instead?"
        ),
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_location_id_by_location_name",
            "arguments": {"location": "Munich"},
        }
    ]

    action = controller.decide(
        context_id="ctx-followup-observed-waypoints",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result("get_location_id_by_location_name", {"id": "loc_munich"})
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_routes_from_start_to_destination",
            "arguments": {
                "start_id": "loc_milan",
                "destination_id": "loc_munich",
            },
        }
    ]


def test_navigation_followup_checks_current_state_when_waypoints_are_unknown() -> None:
    controller = PolicyAwareController()
    messages = system_messages(location_id="loc_andorra")
    tools = navigation_tools()

    controller.decide(
        context_id="ctx-followup-current-state",
        messages=messages,
        tools=tools,
        latest_user_text="I need to change my final destination.",
    )

    action = controller.decide(
        context_id="ctx-followup-current-state",
        messages=messages,
        tools=tools,
        latest_user_text=(
            "Oh, I meant Munich. Can you find the shortest route to Munich instead?"
        ),
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_location_id_by_location_name",
            "arguments": {"location": "Munich"},
        }
    ]

    action = controller.decide(
        context_id="ctx-followup-current-state",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result("get_location_id_by_location_name", {"id": "loc_munich"})
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_current_navigation_state",
            "arguments": {"detailed_information": False},
        }
    ]

    action = controller.decide(
        context_id="ctx-followup-current-state",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_current_navigation_state",
                {
                    "navigation_active": True,
                    "waypoints_id": [
                        "loc_andorra",
                        "loc_paris",
                        "loc_milan",
                        "loc_rome",
                    ],
                    "routes_to_final_destination_id": [
                        "rll_and_par",
                        "rll_par_mil",
                        "rll_mil_rom",
                    ],
                },
            )
        ],
    )
    assert action is not None
    assert action.tool_calls[0]["tool_name"] == "get_routes_from_start_to_destination"


def test_navigation_controller_clarifies_vague_destination_description() -> None:
    controller = PolicyAwareController()
    messages = system_messages(location_id="loc_andorra")
    tools = navigation_tools()

    action = controller.decide(
        context_id="ctx-vague-destination",
        messages=messages,
        tools=tools,
        latest_user_text=(
            "I need to change my final destination to a German city known for its cars."
        ),
    )
    assert action is not None
    assert action.action == "respond"
    assert "Which destination" in action.content

    action = controller.decide(
        context_id="ctx-vague-destination",
        messages=messages,
        tools=tools,
        latest_user_text="I meant Munich.",
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_location_id_by_location_name",
            "arguments": {"location": "Munich"},
        }
    ]


def test_navigation_controller_deletes_named_intermediate_waypoint() -> None:
    controller = PolicyAwareController()
    messages = system_messages(location_id="loc_mannheim")
    tools = navigation_tools()

    action = controller.decide(
        context_id="ctx-delete-waypoint",
        messages=messages,
        tools=tools,
        latest_user_text=(
            "Remove Stuttgart from my route so I can drive directly to Paris "
            "on the shortest route."
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
        context_id="ctx-delete-waypoint",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_current_navigation_state",
                {
                    "navigation_active": True,
                    "waypoints_id": ["loc_mannheim", "loc_stuttgart", "loc_paris"],
                    "routes_to_final_destination_id": [
                        "rll_man_stu",
                        "rll_stu_par",
                    ],
                    "details": {
                        "waypoints": [
                            {"id": "loc_mannheim", "name": "Mannheim"},
                            {"id": "loc_stuttgart", "name": "Stuttgart"},
                            {"id": "loc_paris", "name": "Paris"},
                        ]
                    },
                },
            )
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_routes_from_start_to_destination",
            "arguments": {
                "start_id": "loc_mannheim",
                "destination_id": "loc_paris",
            },
        }
    ]

    action = controller.decide(
        context_id="ctx-delete-waypoint",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_routes_from_start_to_destination",
                {
                    "routes": [
                        {"route_id": "rll_man_par_fast", "alias": ["fastest"]},
                        {"route_id": "rll_man_par_short", "alias": ["shortest"]},
                    ]
                },
            )
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "navigation_delete_waypoint",
            "arguments": {
                "waypoint_id_to_delete": "loc_stuttgart",
                "route_id_without_waypoint": "rll_man_par_short",
            },
        }
    ]


def test_navigation_controller_does_not_infer_when_named_waypoint_is_absent() -> None:
    controller = PolicyAwareController()
    messages = system_messages(location_id="loc_andorra")
    tools = navigation_tools()

    controller.decide(
        context_id="ctx-delete-absent-waypoint",
        messages=messages,
        tools=tools,
        latest_user_text="Remove Paris from my route and keep the shortest route.",
    )
    action = controller.decide(
        context_id="ctx-delete-absent-waypoint",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_current_navigation_state",
                {
                    "navigation_active": True,
                    "waypoints_id": ["loc_andorra", "loc_milan", "loc_stuttgart"],
                    "routes_to_final_destination_id": [
                        "rll_and_mil",
                        "rll_mil_stu",
                    ],
                    "details": {
                        "waypoints": [
                            {"id": "loc_andorra", "name": "Andorra la Vella"},
                            {"id": "loc_milan", "name": "Milan"},
                            {"id": "loc_stuttgart", "name": "Stuttgart"},
                        ]
                    },
                },
            )
        ],
    )

    assert action is not None
    assert action.action == "respond"
    assert "Paris is not currently an intermediate waypoint" in action.content


def test_navigation_controller_asks_route_choice_when_deleting_waypoint() -> None:
    controller = PolicyAwareController()
    messages = system_messages(location_id="loc_mannheim")
    tools = navigation_tools()

    controller.decide(
        context_id="ctx-delete-fastest-default",
        messages=messages,
        tools=tools,
        latest_user_text="Remove Stuttgart from my route.",
    )
    controller.decide(
        context_id="ctx-delete-fastest-default",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_current_navigation_state",
                {
                    "navigation_active": True,
                    "waypoints_id": ["loc_mannheim", "loc_stuttgart", "loc_paris"],
                    "routes_to_final_destination_id": [
                        "rll_man_stu",
                        "rll_stu_par",
                    ],
                    "details": {
                        "waypoints": [
                            {"id": "loc_mannheim", "name": "Mannheim"},
                            {"id": "loc_stuttgart", "name": "Stuttgart"},
                            {"id": "loc_paris", "name": "Paris"},
                        ]
                    },
                },
            )
        ],
    )

    action = controller.decide(
        context_id="ctx-delete-fastest-default",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_routes_from_start_to_destination",
                {
                    "routes": [
                        {"route_id": "rll_man_par_fast", "alias": ["fastest"]},
                        {"route_id": "rll_man_par_short", "alias": ["shortest"]},
                    ]
                },
            )
        ],
    )
    assert action is not None
    assert action.action == "respond"
    assert "fastest" in action.content.lower()
    assert "shortest" in action.content.lower()

    action = controller.decide(
        context_id="ctx-delete-fastest-default",
        messages=messages,
        tools=tools,
        latest_user_text="Use the shortest route.",
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "navigation_delete_waypoint",
            "arguments": {
                "waypoint_id_to_delete": "loc_stuttgart",
                "route_id_without_waypoint": "rll_man_par_short",
            },
        }
    ]


def test_navigation_controller_replaces_single_intermediate_waypoint() -> None:
    controller = PolicyAwareController()
    messages = system_messages(location_id="loc_belgrade")
    tools = navigation_tools()

    action = controller.decide(
        context_id="ctx-replace-waypoint",
        messages=messages,
        tools=tools,
        latest_user_text=(
            "Replace my intermediate stop with Frankfurt and use the fastest route."
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
        context_id="ctx-replace-waypoint",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_current_navigation_state",
                {
                    "navigation_active": True,
                    "waypoints_id": ["loc_belgrade", "loc_bucharest", "loc_rome"],
                    "routes_to_final_destination_id": [
                        "rll_bel_buc",
                        "rll_buc_rom",
                    ],
                    "details": {
                        "waypoints": [
                            {"id": "loc_belgrade", "name": "Belgrade"},
                            {"id": "loc_bucharest", "name": "Bucharest"},
                            {"id": "loc_rome", "name": "Rome"},
                        ]
                    },
                },
            )
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_location_id_by_location_name",
            "arguments": {"location": "Frankfurt"},
        }
    ]

    action = controller.decide(
        context_id="ctx-replace-waypoint",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result("get_location_id_by_location_name", {"id": "loc_frankfurt"})
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_routes_from_start_to_destination",
            "arguments": {
                "start_id": "loc_belgrade",
                "destination_id": "loc_frankfurt",
            },
        }
    ]

    action = controller.decide(
        context_id="ctx-replace-waypoint",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_routes_from_start_to_destination",
                {"routes": [{"route_id": "rll_bel_fra", "alias": ["fastest"]}]},
            )
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_routes_from_start_to_destination",
            "arguments": {
                "start_id": "loc_frankfurt",
                "destination_id": "loc_rome",
            },
        }
    ]

    action = controller.decide(
        context_id="ctx-replace-waypoint",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_routes_from_start_to_destination",
                {"routes": [{"route_id": "rll_fra_rom", "alias": ["fastest"]}]},
            )
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "navigation_replace_one_waypoint",
            "arguments": {
                "waypoint_id_to_replace": "loc_bucharest",
                "new_waypoint_id": "loc_frankfurt",
                "route_id_leading_to_new_waypoint": "rll_bel_fra",
                "route_id_leading_away_from_new_waypoint": "rll_fra_rom",
            },
        }
    ]


def test_navigation_controller_defaults_to_fastest_when_replacing_waypoint() -> None:
    controller = PolicyAwareController()
    messages = system_messages(location_id="loc_wiesbaden")
    tools = navigation_tools()

    controller.decide(
        context_id="ctx-replace-waypoint-fastest-default",
        messages=messages,
        tools=tools,
        latest_user_text="Replace Frankfurt with Milan in my current navigation.",
    )
    controller.decide(
        context_id="ctx-replace-waypoint-fastest-default",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_current_navigation_state",
                {
                    "navigation_active": True,
                    "waypoints_id": [
                        "loc_wiesbaden",
                        "loc_frankfurt",
                        "loc_munich",
                    ],
                    "routes_to_final_destination_id": [
                        "rll_wie_fra",
                        "rll_fra_mun",
                    ],
                    "details": {
                        "waypoints": [
                            {"id": "loc_wiesbaden", "name": "Wiesbaden"},
                            {"id": "loc_frankfurt", "name": "Frankfurt"},
                            {"id": "loc_munich", "name": "Munich"},
                        ]
                    },
                },
            )
        ],
    )
    controller.decide(
        context_id="ctx-replace-waypoint-fastest-default",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result("get_location_id_by_location_name", {"id": "loc_milan"})
        ],
    )
    controller.decide(
        context_id="ctx-replace-waypoint-fastest-default",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_routes_from_start_to_destination",
                {
                    "routes": [
                        {"route_id": "rll_wie_mil_fast", "alias": ["fastest"]},
                        {"route_id": "rll_wie_mil_short", "alias": ["shortest"]},
                    ]
                },
            )
        ],
    )
    action = controller.decide(
        context_id="ctx-replace-waypoint-fastest-default",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_routes_from_start_to_destination",
                {
                    "routes": [
                        {"route_id": "rll_mil_mun_fast", "alias": ["fastest"]},
                        {
                            "route_id": "rll_mil_mun_short",
                            "alias": ["shortest"],
                            "includes_toll": True,
                        },
                    ]
                },
            )
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "navigation_replace_one_waypoint",
            "arguments": {
                "waypoint_id_to_replace": "loc_frankfurt",
                "new_waypoint_id": "loc_milan",
                "route_id_leading_to_new_waypoint": "rll_wie_mil_fast",
                "route_id_leading_away_from_new_waypoint": "rll_mil_mun_fast",
            },
        }
    ]

    action = controller.decide(
        context_id="ctx-replace-waypoint-fastest-default",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result(
                "navigation_replace_one_waypoint",
                {
                    "waypoint_replaced": True,
                    "new_waypoints": [
                        "loc_wiesbaden",
                        "loc_milan",
                        "loc_munich",
                    ],
                    "new_routes": ["rll_wie_mil_fast", "rll_mil_mun_fast"],
                },
            )
        ],
    )
    assert action is not None
    assert action.action == "respond"
    assert "alternative route segment includes toll roads" in action.content
    assert "would you like more information" in action.content.lower()


def test_navigation_controller_does_not_take_over_complex_poi_routes() -> None:
    controller = PolicyAwareController()

    action = controller.decide(
        context_id="ctx-complex",
        messages=system_messages(),
        tools=navigation_tools(),
        latest_user_text=(
            "Set up navigation to Andorra la Vella with a charging station stop."
        ),
    )

    assert action is None
