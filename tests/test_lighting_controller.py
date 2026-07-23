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


def lighting_tools() -> list[dict]:
    return [
        fake_tool("get_exterior_lights_status"),
        fake_tool("set_head_lights_high_beams"),
    ]


def test_high_beam_controller_checks_fog_lights_confirms_and_turns_on() -> None:
    controller = PolicyAwareController()
    messages = [{"role": "system", "content": ""}]
    tools = lighting_tools()

    action = controller.decide(
        context_id="ctx-high-beam",
        messages=messages,
        tools=tools,
        latest_user_text="Turn on the high beam headlights.",
    )
    assert action is not None
    assert action.action == "tool_calls"
    assert action.tool_calls == [
        {"tool_name": "get_exterior_lights_status", "arguments": {}}
    ]

    action = controller.decide(
        context_id="ctx-high-beam",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_exterior_lights_status",
                {
                    "fog_lights": False,
                    "head_lights_low_beams": True,
                    "head_lights_high_beams": False,
                },
            )
        ],
    )
    assert action is not None
    assert action.action == "respond"
    assert "confirm" in action.content.lower()
    assert "on" in action.content.lower()

    action = controller.decide(
        context_id="ctx-high-beam",
        messages=messages,
        tools=tools,
        latest_user_text="Yes, confirm.",
    )
    assert action is not None
    assert action.action == "tool_calls"
    assert action.tool_calls == [
        {"tool_name": "set_head_lights_high_beams", "arguments": {"on": True}}
    ]

    action = controller.decide(
        context_id="ctx-high-beam",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result("set_head_lights_high_beams", {"on": True})
        ],
    )
    assert action is not None
    assert action.action == "respond"
    assert "high beam headlights are on" in action.content.lower()


def test_high_beam_controller_refuses_when_fog_lights_are_on() -> None:
    controller = PolicyAwareController()
    messages = [{"role": "system", "content": ""}]
    tools = lighting_tools()

    controller.decide(
        context_id="ctx-fog-conflict",
        messages=messages,
        tools=tools,
        latest_user_text="Activate the high beams.",
    )
    action = controller.decide(
        context_id="ctx-fog-conflict",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_exterior_lights_status",
                {
                    "fog_lights": True,
                    "head_lights_low_beams": True,
                    "head_lights_high_beams": False,
                },
            )
        ],
    )

    assert action is not None
    assert action.action == "respond"
    assert "fog lights" in action.content.lower()
    assert "can't turn on" in action.content.lower()


def test_high_beam_controller_requires_status_check_before_turning_on() -> None:
    controller = PolicyAwareController()
    messages = [{"role": "system", "content": ""}]
    tools = [fake_tool("set_head_lights_high_beams")]

    action = controller.decide(
        context_id="ctx-missing-status",
        messages=messages,
        tools=tools,
        latest_user_text="I need the high beams on.",
    )

    assert action is not None
    assert action.action == "respond"
    assert "status check is unavailable" in action.content.lower()


def test_high_beam_controller_confirms_before_turning_off() -> None:
    controller = PolicyAwareController()
    messages = [{"role": "system", "content": ""}]
    tools = lighting_tools()

    action = controller.decide(
        context_id="ctx-high-beam-off",
        messages=messages,
        tools=tools,
        latest_user_text="Turn off the high beams.",
    )
    assert action is not None
    assert action.action == "respond"
    assert "confirm" in action.content.lower()
    assert "off" in action.content.lower()

    action = controller.decide(
        context_id="ctx-high-beam-off",
        messages=messages,
        tools=tools,
        latest_user_text="Yes.",
    )
    assert action is not None
    assert action.tool_calls == [
        {"tool_name": "set_head_lights_high_beams", "arguments": {"on": False}}
    ]


def test_high_beam_controller_stops_when_confirmation_is_declined() -> None:
    controller = PolicyAwareController()
    messages = [{"role": "system", "content": ""}]
    tools = lighting_tools()

    controller.decide(
        context_id="ctx-high-beam-decline",
        messages=messages,
        tools=tools,
        latest_user_text="Turn off the high beams.",
    )
    action = controller.decide(
        context_id="ctx-high-beam-decline",
        messages=messages,
        tools=tools,
        latest_user_text="No, cancel that.",
    )

    assert action is not None
    assert action.action == "respond"
    assert "won't set" in action.content.lower()
