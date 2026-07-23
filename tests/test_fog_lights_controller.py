import json

from carbench_agent_core import PolicyAwareController


def fake_tool(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }


def tool_result(tool_name: str, result: dict) -> dict:
    return {
        "tool_name": tool_name,
        "content": json.dumps({"status": "SUCCESS", "result": result}),
    }


def messages() -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                'CURRENT_LOCATION {"id": "loc_test"}\n'
                'DATETIME {"month": 1, "day": 2, "hour": 3, "minute": 4}'
            ),
        }
    ]


def fog_tools() -> list[dict]:
    return [
        fake_tool("get_weather"),
        fake_tool("get_exterior_lights_status"),
        fake_tool("set_head_lights_low_beams"),
        fake_tool("set_head_lights_high_beams"),
        fake_tool("set_fog_lights"),
    ]


def weather_result(condition: str = "cloudy_and_hail") -> dict:
    return {
        "current_slot": {
            "condition": condition,
            "temperature_c": 2,
            "wind_speed_kph": 12,
        },
        "next_slot": None,
    }


def exterior_result(*, low_beams: bool = True, high_beams: bool = False) -> dict:
    return {
        "fog_lights": False,
        "head_lights_low_beams": low_beams,
        "head_lights_high_beams": high_beams,
    }


def start_fog_flow(
    controller: PolicyAwareController,
    *,
    context_id: str,
    tools: list[dict] | None = None,
):
    return controller.decide(
        context_id=context_id,
        messages=messages(),
        tools=tools or fog_tools(),
        latest_user_text="Turn on the fog lights.",
    )


def test_fog_lights_controller_checks_weather_and_exterior_status() -> None:
    controller = PolicyAwareController()
    tools = fog_tools()

    action = start_fog_flow(controller, context_id="ctx-fog-basic", tools=tools)
    assert action is not None
    assert action.action == "tool_calls"
    assert action.tool_calls == [
        {
            "tool_name": "get_weather",
            "arguments": {
                "location_or_poi_id": "loc_test",
                "month": 1,
                "day": 2,
                "time_hour_24hformat": 3,
                "time_minutes": 4,
            },
        }
    ]

    action = controller.decide(
        context_id="ctx-fog-basic",
        messages=messages(),
        tools=tools,
        latest_tool_results=[tool_result("get_weather", weather_result())],
    )
    assert action is not None
    assert action.tool_calls == [
        {"tool_name": "get_exterior_lights_status", "arguments": {}}
    ]

    action = controller.decide(
        context_id="ctx-fog-basic",
        messages=messages(),
        tools=tools,
        latest_tool_results=[
            tool_result("get_exterior_lights_status", exterior_result())
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {"tool_name": "set_fog_lights", "arguments": {"on": True}}
    ]


def test_fog_lights_controller_enables_low_beams_before_fog_lights() -> None:
    controller = PolicyAwareController()
    tools = fog_tools()

    start_fog_flow(controller, context_id="ctx-fog-low", tools=tools)
    controller.decide(
        context_id="ctx-fog-low",
        messages=messages(),
        tools=tools,
        latest_tool_results=[tool_result("get_weather", weather_result())],
    )
    action = controller.decide(
        context_id="ctx-fog-low",
        messages=messages(),
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_exterior_lights_status",
                exterior_result(low_beams=False, high_beams=False),
            )
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {"tool_name": "set_head_lights_low_beams", "arguments": {"on": True}}
    ]

    action = controller.decide(
        context_id="ctx-fog-low",
        messages=messages(),
        tools=tools,
        latest_tool_results=[tool_result("set_head_lights_low_beams", {"on": True})],
    )
    assert action is not None
    assert action.tool_calls == [
        {"tool_name": "set_fog_lights", "arguments": {"on": True}}
    ]


def test_fog_lights_controller_confirms_before_disabling_high_beams() -> None:
    controller = PolicyAwareController()
    tools = fog_tools()

    start_fog_flow(controller, context_id="ctx-fog-high", tools=tools)
    controller.decide(
        context_id="ctx-fog-high",
        messages=messages(),
        tools=tools,
        latest_tool_results=[tool_result("get_weather", weather_result())],
    )
    action = controller.decide(
        context_id="ctx-fog-high",
        messages=messages(),
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_exterior_lights_status",
                exterior_result(low_beams=True, high_beams=True),
            )
        ],
    )
    assert action is not None
    assert action.action == "respond"
    assert "confirm" in action.content.lower()
    assert "high beams" in action.content.lower()

    action = controller.decide(
        context_id="ctx-fog-high",
        messages=messages(),
        tools=tools,
        latest_user_text="Yes, continue.",
    )
    assert action is not None
    assert action.tool_calls == [
        {"tool_name": "set_head_lights_high_beams", "arguments": {"on": False}}
    ]

    action = controller.decide(
        context_id="ctx-fog-high",
        messages=messages(),
        tools=tools,
        latest_tool_results=[tool_result("set_head_lights_high_beams", {"on": False})],
    )
    assert action is not None
    assert action.tool_calls == [
        {"tool_name": "set_fog_lights", "arguments": {"on": True}}
    ]


def test_fog_lights_controller_confirms_for_non_allowed_weather() -> None:
    controller = PolicyAwareController()
    tools = fog_tools()

    start_fog_flow(controller, context_id="ctx-fog-weather", tools=tools)
    action = controller.decide(
        context_id="ctx-fog-weather",
        messages=messages(),
        tools=tools,
        latest_tool_results=[tool_result("get_weather", weather_result("sunny"))],
    )
    assert action is not None
    assert action.action == "respond"
    assert "confirmation" in action.content.lower()

    action = controller.decide(
        context_id="ctx-fog-weather",
        messages=messages(),
        tools=tools,
        latest_user_text="Yes, I still want them on.",
    )
    assert action is not None
    assert action.tool_calls == [
        {"tool_name": "get_exterior_lights_status", "arguments": {}}
    ]


def test_fog_lights_controller_stops_when_fog_tool_is_missing() -> None:
    controller = PolicyAwareController()
    tools = [fake_tool("get_weather"), fake_tool("get_exterior_lights_status")]

    action = start_fog_flow(controller, context_id="ctx-fog-missing", tools=tools)

    assert action is not None
    assert action.action == "respond"
    assert "unavailable" in action.content.lower()


def test_fog_lights_controller_turns_off_without_weather_check() -> None:
    controller = PolicyAwareController()
    tools = fog_tools()

    action = controller.decide(
        context_id="ctx-fog-off",
        messages=messages(),
        tools=tools,
        latest_user_text="Turn off the fog lights.",
    )

    assert action is not None
    assert action.tool_calls == [
        {"tool_name": "set_fog_lights", "arguments": {"on": False}}
    ]
