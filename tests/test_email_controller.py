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


def system_messages(*, include_secretary: bool = False) -> list[dict]:
    preference = ""
    if include_secretary:
        preference = (
            '\nUSER_PREFERENCES = {"productivity_and_communication": '
            '{"email": ["If the email is business related, user always '
            "wants to send the mail in addition to original recipient to "
            "secretary: 'emma.walker1132@outlook.com'\"]}}"
        )
    return [
        {
            "role": "system",
            "content": (
                'CURRENT_LOCATION = {"id": "loc_ams_749623", "name": "Amsterdam"}\n'
                'DATETIME = {"year": 2025, "month": 1, "day": 20, '
                '"hour": 14, "minute": 15}'
                f"{preference}"
            ),
        }
    ]


def email_tools(
    *, include_preferences: bool = False, include_contact_lookup: bool = True
) -> list[dict]:
    tools = [
        fake_tool("get_entries_from_calendar"),
        fake_tool(
            "get_contact_information",
            {"contact_ids": {"type": "array", "items": {"type": "string"}}},
            ["contact_ids"],
        ),
        fake_tool(
            "send_email",
            {
                "email_addresses": {"type": "array", "items": {"type": "string"}},
                "content_message": {"type": "string"},
            },
            ["email_addresses", "content_message"],
        ),
    ]
    if include_contact_lookup:
        tools.insert(
            0,
            fake_tool(
                "get_contact_id_by_contact_name",
                {
                    "contact_first_name": {"type": "string"},
                    "contact_last_name": {"type": "string"},
                },
            ),
        )
    if include_preferences:
        tools.insert(3, fake_tool("get_user_preferences"))
    return tools


def partnership_calendar() -> dict:
    return {
        "date": {"year": 2025, "month": 1, "day": 20},
        "meetings": [
            {
                "start": {"hour": "14", "minute": "00"},
                "duration": "60min",
                "location": "Minsk",
                "attendees": ["con_5327", "con_8783"],
                "topic": "Partnership Discussion",
            }
        ],
    }


def attendee_calendar() -> dict:
    return {
        "date": {"year": 2025, "month": 1, "day": 20},
        "meetings": [
            {
                "start": {"hour": "15", "minute": "30"},
                "duration": "30min",
                "location": "Harbor Bay",
                "attendees": ["con_alpha", "con_beta"],
                "topic": "Launch Review",
            },
            {
                "start": {"hour": "18", "minute": "00"},
                "duration": "30min",
                "location": "Lakeside",
                "attendees": ["con_gamma"],
                "topic": "Budget Sync",
            },
        ],
    }


def attendee_contact_info() -> dict:
    return {
        "con_alpha": {
            "id": "con_alpha",
            "name": {"first_name": "Avery", "last_name": "Stone"},
            "phone_number": "+49 100 000001",
            "email": "avery.stone@example.com",
        },
        "con_beta": {
            "id": "con_beta",
            "name": {"first_name": "Blair", "last_name": "Reed"},
            "phone_number": "+49 100 000002",
            "email": "blair.reed@example.com",
        },
        "con_gamma": {
            "id": "con_gamma",
            "name": {"first_name": "Casey", "last_name": "Vale"},
            "phone_number": "+49 100 000003",
            "email": "casey.vale@example.com",
        },
    }


def active_route_result() -> dict:
    return {
        "navigation_active": True,
        "waypoints_id": ["loc_origin", "loc_destination"],
        "routes_to_final_destination_id": ["route_origin_destination"],
        "details": {
            "routes": [
                {
                    "route_id": "route_origin_destination",
                    "distance_km": 460,
                    "alias": ["fastest"],
                }
            ]
        },
    }


def charging_stop_result() -> dict:
    return {
        "pois_found_along_route": [
            {
                "id": "poi_charge_alpha",
                "name": "VoltHub",
                "phone_number": "+49 100 111111",
                "detour_from_route_time": {"hour": 0, "minutes": 6},
                "charging_plugs": [
                    {
                        "plug_id": "plug_dc",
                        "power_type": "DC",
                        "power_kw": 250,
                        "availability": "available",
                    }
                ],
            }
        ]
    }


def email_route_energy_tools() -> list[dict]:
    return email_tools() + [
        fake_tool(
            "get_current_navigation_state",
            {"detailed_information": {"type": "boolean"}},
        ),
        fake_tool(
            "search_poi_along_the_route",
            {
                "category_poi": {"type": "string"},
                "route_id": {"type": "string"},
                "at_kilometer": {"type": "number"},
            },
            ["category_poi", "route_id", "at_kilometer"],
        ),
    ]


def email_weather_tools() -> list[dict]:
    return email_tools() + [
        fake_tool(
            "get_location_id_by_location_name",
            {"location": {"type": "string"}},
            ["location"],
        ),
        fake_tool(
            "get_weather",
            {
                "location_or_poi_id": {"type": "string"},
                "month": {"type": "integer"},
                "day": {"type": "integer"},
                "time_hour_24hformat": {"type": "integer"},
            },
            ["location_or_poi_id", "month", "day", "time_hour_24hformat"],
        ),
    ]


def test_contact_lookup_refuses_when_search_tool_missing() -> None:
    controller = PolicyAwareController()

    action = controller.decide(
        context_id="ctx-missing-contact-search",
        messages=system_messages(),
        tools=email_tools(include_contact_lookup=False),
        latest_user_text="Find Avery.",
    )

    assert action is not None
    assert action.action == "respond"
    assert "can't look up contacts" in action.content


def test_email_controller_confirms_before_sending_late_meeting_email() -> None:
    controller = PolicyAwareController()
    tools = email_tools()
    messages = system_messages(include_secretary=True)

    controller.decide(
        context_id="ctx-late-email",
        messages=messages,
        tools=tools,
        latest_user_text="What's the status of my 'Partnership Discussion' meeting?",
    )
    controller.decide(
        context_id="ctx-late-email",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result("get_entries_from_calendar", partnership_calendar())
        ],
    )

    action = controller.decide(
        context_id="ctx-late-email",
        messages=messages,
        tools=tools,
        latest_user_text=(
            "Send an email to Frank Walker. I'm running late and apologize "
            "for the delay."
        ),
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_contact_id_by_contact_name",
            "arguments": {
                "contact_first_name": "Frank",
                "contact_last_name": "Walker",
            },
        }
    ]

    action = controller.decide(
        context_id="ctx-late-email",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_contact_id_by_contact_name",
                {"matches": {"con_1541": "frank walker"}},
            )
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_contact_information",
            "arguments": {"contact_ids": ["con_1541"]},
        }
    ]

    action = controller.decide(
        context_id="ctx-late-email",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_contact_information",
                {
                    "con_1541": {
                        "id": "con_1541",
                        "name": {"first_name": "Frank", "last_name": "Walker"},
                        "phone_number": "+49 486 408537",
                        "email": "frank.walker1219@andex.com",
                    }
                },
            )
        ],
    )
    assert action is not None
    assert action.action == "respond"
    assert "confirm" in action.content.lower()
    assert "frank.walker1219@andex.com" in action.content
    assert "emma.walker1132@outlook.com" in action.content
    assert "14:00" in action.content
    assert "15 minutes late" in action.content

    action = controller.decide(
        context_id="ctx-late-email",
        messages=messages,
        tools=tools,
        latest_user_text="Yes, please send it.",
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "send_email",
            "arguments": {
                "email_addresses": [
                    "frank.walker1219@andex.com",
                    "emma.walker1132@outlook.com",
                ],
                "content_message": (
                    "Hi Frank, I wanted to reach out regarding our Partnership "
                    "Discussion meeting that started at 14:00 today. I'm running "
                    "about 15 minutes late and apologize for the delay. I should be "
                    "there shortly. Thank you for your patience. Best regards"
                ),
            },
        }
    ]


def test_email_controller_sends_contact_details_to_original_recipient() -> None:
    controller = PolicyAwareController()
    tools = email_tools()
    messages = system_messages()

    action = controller.decide(
        context_id="ctx-share-contact",
        messages=messages,
        tools=tools,
        latest_user_text=(
            "Look up Rachel Walker's contact information. I want to send her an email."
        ),
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_contact_id_by_contact_name",
            "arguments": {
                "contact_first_name": "Rachel",
                "contact_last_name": "Walker",
            },
        }
    ]

    action = controller.decide(
        context_id="ctx-share-contact",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_contact_id_by_contact_name",
                {"matches": {"con_3692": "rachel walker"}},
            )
        ],
    )
    assert action is not None
    assert action.action == "respond"
    assert "whose contact details" in action.content.lower()

    action = controller.decide(
        context_id="ctx-share-contact",
        messages=messages,
        tools=tools,
        latest_user_text="Share David Harris's contact details.",
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_contact_id_by_contact_name",
            "arguments": {
                "contact_first_name": "David",
                "contact_last_name": "Harris",
            },
        }
    ]

    action = controller.decide(
        context_id="ctx-share-contact",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_contact_id_by_contact_name",
                {"matches": {"con_8528": "david harris"}},
            )
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_contact_information",
            "arguments": {"contact_ids": ["con_3692", "con_8528"]},
        }
    ]

    action = controller.decide(
        context_id="ctx-share-contact",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_contact_information",
                {
                    "con_3692": {
                        "id": "con_3692",
                        "name": {"first_name": "Rachel", "last_name": "Walker"},
                        "phone_number": "+49 913 182721",
                        "email": "rachel.walker1312@outlook.com",
                    },
                    "con_8528": {
                        "id": "con_8528",
                        "name": {"first_name": "David", "last_name": "Harris"},
                        "phone_number": "+49 550 435701",
                        "email": "david.harris3615@protonmail.com",
                    },
                },
            )
        ],
    )
    assert action is not None
    assert action.action == "respond"
    assert "rachel.walker1312@outlook.com" in action.content
    assert "David Harris" in action.content
    assert "confirm" in action.content.lower()

    action = controller.decide(
        context_id="ctx-share-contact",
        messages=messages,
        tools=tools,
        latest_user_text="Yes.",
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "send_email",
            "arguments": {
                "email_addresses": ["rachel.walker1312@outlook.com"],
                "content_message": (
                    "Hi Rachel,\n\n"
                    "I wanted to share David Harris's contact information with you:\n\n"
                    "Name: David Harris\n"
                    "Phone: +49 550 435701\n"
                    "Email: david.harris3615@protonmail.com\n\n"
                    "Best regards"
                ),
            },
        }
    ]


def test_email_controller_disambiguates_first_name_recipient_before_sending() -> None:
    controller = PolicyAwareController()
    tools = email_tools()
    messages = system_messages()

    action = controller.decide(
        context_id="ctx-share-contact-ambiguous",
        messages=messages,
        tools=tools,
        latest_user_text="Can you help me find Rachel's contact info? I want to send her an email.",
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_contact_id_by_contact_name",
            "arguments": {"contact_first_name": "Rachel"},
        }
    ]

    action = controller.decide(
        context_id="ctx-share-contact-ambiguous",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_contact_id_by_contact_name",
                {
                    "matches": {
                        "con_9881": "rachel mitchell",
                        "con_3692": "rachel walker",
                    }
                },
            )
        ],
    )
    assert action is not None
    assert action.action == "respond"
    assert "multiple matching contacts" in action.content.lower()
    assert "Rachel Walker" in action.content

    action = controller.decide(
        context_id="ctx-share-contact-ambiguous",
        messages=messages,
        tools=tools,
        latest_user_text="Rachel Walker.",
    )
    assert action is not None
    assert action.action == "respond"
    assert "whose contact details" in action.content.lower()


def test_email_controller_preserves_preference_recipients_after_confirmation() -> None:
    controller = PolicyAwareController()
    tools = email_tools(include_preferences=True)
    messages = system_messages()

    controller.decide(
        context_id="ctx-late-email-pref-tool",
        messages=messages,
        tools=tools,
        latest_user_text="What's the status of my 'Partnership Discussion' meeting?",
    )
    controller.decide(
        context_id="ctx-late-email-pref-tool",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result("get_entries_from_calendar", partnership_calendar())
        ],
    )
    controller.decide(
        context_id="ctx-late-email-pref-tool",
        messages=messages,
        tools=tools,
        latest_user_text=(
            "Can you send an email to Frank Walker for me? "
            "I need to let him know I'm running late and apologize."
        ),
    )
    controller.decide(
        context_id="ctx-late-email-pref-tool",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_contact_id_by_contact_name",
                {"matches": {"con_1541": "frank walker"}},
            )
        ],
    )
    action = controller.decide(
        context_id="ctx-late-email-pref-tool",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_contact_information",
                {
                    "con_1541": {
                        "id": "con_1541",
                        "name": {"first_name": "Frank", "last_name": "Walker"},
                        "phone_number": "+49 486 408537",
                        "email": "frank.walker1219@andex.com",
                    }
                },
            )
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_user_preferences",
            "arguments": {
                "preference_categories": {
                    "productivity_and_communication": {"email": True}
                }
            },
        }
    ]

    action = controller.decide(
        context_id="ctx-late-email-pref-tool",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_user_preferences",
                {
                    "productivity_and_communication": {
                        "email": [
                            "If the email is business related, user always wants "
                            "to send the mail in addition to original recipient "
                            "to secretary: 'emma.walker1132@outlook.com'"
                        ]
                    }
                },
            )
        ],
    )
    assert action is not None
    assert action.action == "respond"
    assert "emma.walker1132@outlook.com" in action.content

    action = controller.decide(
        context_id="ctx-late-email-pref-tool",
        messages=messages,
        tools=tools,
        latest_user_text="Yes.",
    )
    assert action is not None
    assert action.tool_calls[0]["tool_name"] == "send_email"
    assert action.tool_calls[0]["arguments"]["email_addresses"] == [
        "frank.walker1219@andex.com",
        "emma.walker1132@outlook.com",
    ]


def test_email_controller_confirms_before_sending_meeting_attendee_reminder() -> None:
    controller = PolicyAwareController()
    tools = email_tools()
    messages = system_messages()

    action = controller.decide(
        context_id="ctx-meeting-attendee-reminder",
        messages=messages,
        tools=tools,
        latest_user_text=(
            "Send an email reminder to the attendees of my Launch Review meeting."
        ),
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_entries_from_calendar",
            "arguments": {"month": 1, "day": 20},
        }
    ]

    action = controller.decide(
        context_id="ctx-meeting-attendee-reminder",
        messages=messages,
        tools=tools,
        latest_tool_results=[tool_result("get_entries_from_calendar", attendee_calendar())],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_contact_information",
            "arguments": {"contact_ids": ["con_alpha", "con_beta"]},
        }
    ]

    action = controller.decide(
        context_id="ctx-meeting-attendee-reminder",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result("get_contact_information", attendee_contact_info())
        ],
    )
    assert action is not None
    assert action.action == "respond"
    assert "what should the email say" in action.content.lower()

    action = controller.decide(
        context_id="ctx-meeting-attendee-reminder",
        messages=messages,
        tools=tools,
        latest_user_text=(
            "It should be a friendly reminder about the meeting today at 3:30 PM."
        ),
    )
    assert action is not None
    assert action.action == "respond"
    assert "please confirm these email details" in action.content.lower()
    assert "avery.stone@example.com" in action.content
    assert "blair.reed@example.com" in action.content
    assert "15:30" in action.content
    assert "3:30 PM" not in action.content

    action = controller.decide(
        context_id="ctx-meeting-attendee-reminder",
        messages=messages,
        tools=tools,
        latest_user_text="Yes, send it.",
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "send_email",
            "arguments": {
                "email_addresses": [
                    "avery.stone@example.com",
                    "blair.reed@example.com",
                ],
                "content_message": (
                    "A friendly reminder about the meeting today at 15:30"
                ),
            },
        }
    ]


def test_email_controller_confirms_charging_stop_email_to_meeting_attendees() -> None:
    controller = PolicyAwareController()
    tools = email_route_energy_tools()
    messages = system_messages()

    controller.decide(
        context_id="ctx-charging-stop-attendee-email",
        messages=messages,
        tools=tools,
        latest_tool_results=[tool_result("get_entries_from_calendar", attendee_calendar())],
    )
    controller.decide(
        context_id="ctx-charging-stop-attendee-email",
        messages=messages,
        tools=tools,
        latest_user_text="Find a charging station along my current route around 350 km.",
    )
    controller.decide(
        context_id="ctx-charging-stop-attendee-email",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result("get_current_navigation_state", active_route_result())
        ],
    )
    controller.decide(
        context_id="ctx-charging-stop-attendee-email",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result("search_poi_along_the_route", charging_stop_result())
        ],
    )

    action = controller.decide(
        context_id="ctx-charging-stop-attendee-email",
        messages=messages,
        tools=tools,
        latest_user_text=(
            "Please send an email to the attendees of my Launch Review meeting. "
            "Let them know I'll be slightly delayed because of a charging stop. "
            "Mention VoltHub and the detour."
        ),
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_contact_information",
            "arguments": {"contact_ids": ["con_alpha", "con_beta"]},
        }
    ]

    action = controller.decide(
        context_id="ctx-charging-stop-attendee-email",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result("get_contact_information", attendee_contact_info())
        ],
    )
    assert action is not None
    assert action.action == "respond"
    assert "please confirm these email details" in action.content.lower()
    assert "VoltHub" in action.content
    assert "6 minutes" in action.content

    action = controller.decide(
        context_id="ctx-charging-stop-attendee-email",
        messages=messages,
        tools=tools,
        latest_user_text="Yes, send it.",
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "send_email",
            "arguments": {
                "email_addresses": [
                    "avery.stone@example.com",
                    "blair.reed@example.com",
                ],
                "content_message": (
                    "Dear team,\n\n"
                    "I wanted to let you know that I need to make a charging "
                    "stop at VoltHub, which will add about 6 minutes to my "
                    "journey due to the detour. I may be slightly delayed for "
                    "the Launch Review meeting.\n\n"
                    "Best regards"
                ),
            },
        }
    ]


def test_email_controller_checks_weather_before_meeting_attendee_weather_email() -> None:
    controller = PolicyAwareController()
    tools = email_weather_tools()
    messages = system_messages()

    action = controller.decide(
        context_id="ctx-weather-attendee-email",
        messages=messages,
        tools=tools,
        latest_user_text=(
            "Can you check the weather for Harbor Bay at 3:30 PM today? "
            "After that, please send an email to the attendees of the Launch "
            "Review meeting about the weather and travel."
        ),
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_entries_from_calendar",
            "arguments": {"month": 1, "day": 20},
        }
    ]

    action = controller.decide(
        context_id="ctx-weather-attendee-email",
        messages=messages,
        tools=tools,
        latest_tool_results=[tool_result("get_entries_from_calendar", attendee_calendar())],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_contact_information",
            "arguments": {"contact_ids": ["con_alpha", "con_beta"]},
        }
    ]

    action = controller.decide(
        context_id="ctx-weather-attendee-email",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result("get_contact_information", attendee_contact_info())
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_location_id_by_location_name",
            "arguments": {"location": "Harbor Bay"},
        }
    ]

    action = controller.decide(
        context_id="ctx-weather-attendee-email",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result("get_location_id_by_location_name", {"id": "loc_harbor_bay"})
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_weather",
            "arguments": {
                "location_or_poi_id": "loc_harbor_bay",
                "month": 1,
                "day": 20,
                "time_hour_24hformat": 15,
            },
        }
    ]

    action = controller.decide(
        context_id="ctx-weather-attendee-email",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_weather",
                {"current_slot": {"condition": "cloudy_and_rain", "temperature": 6}},
            )
        ],
    )
    assert action is not None
    assert action.action == "respond"
    assert "please confirm these email details" in action.content.lower()
    assert "avery.stone@example.com" in action.content
    assert "blair.reed@example.com" in action.content
    assert "cloudy and rain" in action.content
    assert "6 C" in action.content

    action = controller.decide(
        context_id="ctx-weather-attendee-email",
        messages=messages,
        tools=tools,
        latest_user_text="Yes, send it.",
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "send_email",
            "arguments": {
                "email_addresses": [
                    "avery.stone@example.com",
                    "blair.reed@example.com",
                ],
                "content_message": (
                    "Hi team,\n\n"
                    "I wanted to share a weather update for our Launch Review "
                    "meeting in Harbor Bay today at 15:30. The forecast shows "
                    "cloudy and rain. The temperature is expected to be around "
                    "6 C.\n\n"
                    "Please plan accordingly for travel, allow extra time, and "
                    "bring suitable weather protection if needed.\n\n"
                    "Best regards"
                ),
            },
        }
    ]


def test_email_controller_uses_selected_meeting_location_for_weather_email() -> None:
    controller = PolicyAwareController()
    tools = email_weather_tools()
    messages = system_messages()

    controller.decide(
        context_id="ctx-weather-attendee-email-selected-meeting",
        messages=messages,
        tools=tools,
        latest_tool_results=[tool_result("get_entries_from_calendar", attendee_calendar())],
    )

    action = controller.decide(
        context_id="ctx-weather-attendee-email-selected-meeting",
        messages=messages,
        tools=tools,
        latest_user_text=(
            "Can you send an email to the attendees of that meeting, letting "
            "them know about the weather in Harbor Bay and how it might affect "
            "their travel and the meeting?"
        ),
    )
    assert action is not None
    assert action.action == "respond"
    assert "which meeting" in action.content.lower()

    action = controller.decide(
        context_id="ctx-weather-attendee-email-selected-meeting",
        messages=messages,
        tools=tools,
        latest_user_text="The Launch Review meeting at 3:30 PM.",
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_contact_information",
            "arguments": {"contact_ids": ["con_alpha", "con_beta"]},
        }
    ]

    action = controller.decide(
        context_id="ctx-weather-attendee-email-selected-meeting",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result("get_contact_information", attendee_contact_info())
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_location_id_by_location_name",
            "arguments": {"location": "Harbor Bay"},
        }
    ]


def test_email_controller_asks_which_meeting_before_attendee_reminder() -> None:
    controller = PolicyAwareController()
    tools = email_tools()
    messages = system_messages()

    controller.decide(
        context_id="ctx-meeting-attendee-disambiguation",
        messages=messages,
        tools=tools,
        latest_user_text="Send an email reminder to my meeting attendees today.",
    )
    action = controller.decide(
        context_id="ctx-meeting-attendee-disambiguation",
        messages=messages,
        tools=tools,
        latest_tool_results=[tool_result("get_entries_from_calendar", attendee_calendar())],
    )
    assert action is not None
    assert action.action == "respond"
    assert "which meeting" in action.content.lower()
    assert "Launch Review" in action.content
    assert "Budget Sync" in action.content

    action = controller.decide(
        context_id="ctx-meeting-attendee-disambiguation",
        messages=messages,
        tools=tools,
        latest_user_text="Launch Review.",
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "get_contact_information",
            "arguments": {"contact_ids": ["con_alpha", "con_beta"]},
        }
    ]
