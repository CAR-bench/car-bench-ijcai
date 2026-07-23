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


def defrost_tools() -> list[dict]:
    return [
        fake_tool("get_climate_settings"),
        fake_tool("get_vehicle_window_positions"),
        fake_tool(
            "open_close_window",
            {"window": {"type": "string"}, "percentage": {"type": "number"}},
            ["window", "percentage"],
        ),
        fake_tool("set_fan_speed", {"level": {"type": "integer"}}, ["level"]),
        fake_tool(
            "set_fan_airflow_direction",
            {"direction": {"type": "string"}},
            ["direction"],
        ),
        fake_tool("set_air_conditioning", {"on": {"type": "boolean"}}, ["on"]),
        fake_tool(
            "set_window_defrost",
            {"on": {"type": "boolean"}, "defrost_window": {"type": "string"}},
            ["on", "defrost_window"],
        ),
    ]


def climate_result(
    *,
    fan_speed: int = 0,
    fan_airflow_direction: str = "HEAD",
    air_conditioning: bool = False,
) -> dict:
    return {
        "fan_speed": fan_speed,
        "fan_airflow_direction": fan_airflow_direction,
        "air_conditioning": air_conditioning,
        "air_circulation": "AUTO",
        "window_front_defrost": False,
        "window_rear_defrost": False,
    }


def window_positions(*, opened: int = 25) -> dict:
    return {
        "window_driver_position": opened,
        "window_passenger_position": 0,
        "window_driver_rear_position": 0,
        "window_passenger_rear_position": 0,
    }


def test_front_defrost_controller_orders_required_safety_actions() -> None:
    controller = PolicyAwareController()
    messages = [{"role": "system", "content": ""}]
    tools = defrost_tools()

    action = controller.decide(
        context_id="ctx-defrost-front",
        messages=messages,
        tools=tools,
        latest_user_text="Turn on the front window defrost.",
    )
    assert action is not None
    assert action.tool_calls == [{"tool_name": "get_climate_settings", "arguments": {}}]

    action = controller.decide(
        context_id="ctx-defrost-front",
        messages=messages,
        tools=tools,
        latest_tool_results=[tool_result("get_climate_settings", climate_result())],
    )
    assert action is not None
    assert action.tool_calls == [
        {"tool_name": "get_vehicle_window_positions", "arguments": {}}
    ]

    action = controller.decide(
        context_id="ctx-defrost-front",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result("get_vehicle_window_positions", window_positions())
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {"tool_name": "open_close_window", "arguments": {"window": "ALL", "percentage": 0}}
    ]

    action = controller.decide(
        context_id="ctx-defrost-front",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result("open_close_window", {"window": "ALL", "percentage": 0})
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {"tool_name": "set_fan_speed", "arguments": {"level": 2}}
    ]

    action = controller.decide(
        context_id="ctx-defrost-front",
        messages=messages,
        tools=tools,
        latest_tool_results=[tool_result("set_fan_speed", {"level": 2})],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "set_fan_airflow_direction",
            "arguments": {"direction": "WINDSHIELD"},
        }
    ]

    action = controller.decide(
        context_id="ctx-defrost-front",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result("set_fan_airflow_direction", {"direction": "WINDSHIELD"})
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {"tool_name": "set_air_conditioning", "arguments": {"on": True}}
    ]

    action = controller.decide(
        context_id="ctx-defrost-front",
        messages=messages,
        tools=tools,
        latest_tool_results=[tool_result("set_air_conditioning", {"on": True})],
    )
    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "set_window_defrost",
            "arguments": {"on": True, "defrost_window": "FRONT"},
        }
    ]


def test_front_defrost_controller_sets_defrost_when_climate_is_ready() -> None:
    controller = PolicyAwareController()
    messages = [{"role": "system", "content": ""}]
    tools = defrost_tools()

    controller.decide(
        context_id="ctx-defrost-ready",
        messages=messages,
        tools=tools,
        latest_user_text="Activate the windshield defrost.",
    )
    action = controller.decide(
        context_id="ctx-defrost-ready",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_climate_settings",
                climate_result(
                    fan_speed=3,
                    fan_airflow_direction="WINDSHIELD_HEAD",
                    air_conditioning=True,
                ),
            )
        ],
    )

    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "set_window_defrost",
            "arguments": {"on": True, "defrost_window": "FRONT"},
        }
    ]


def test_front_defrost_controller_treats_missing_ac_status_as_needing_ac() -> None:
    controller = PolicyAwareController()
    messages = [{"role": "system", "content": ""}]
    tools = defrost_tools()

    controller.decide(
        context_id="ctx-defrost-unknown-ac",
        messages=messages,
        tools=tools,
        latest_user_text="Turn on the front defrost.",
    )
    action = controller.decide(
        context_id="ctx-defrost-unknown-ac",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_climate_settings",
                {
                    "fan_speed": 2,
                    "fan_airflow_direction": "WINDSHIELD",
                    "air_circulation": "AUTO",
                },
            )
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {"tool_name": "get_vehicle_window_positions", "arguments": {}}
    ]

    action = controller.decide(
        context_id="ctx-defrost-unknown-ac",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result("get_vehicle_window_positions", window_positions(opened=0))
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {"tool_name": "set_air_conditioning", "arguments": {"on": True}}
    ]


def test_defrost_controller_asks_for_window_when_ambiguous() -> None:
    controller = PolicyAwareController()
    messages = [{"role": "system", "content": ""}]
    tools = defrost_tools()

    action = controller.decide(
        context_id="ctx-defrost-ambiguous",
        messages=messages,
        tools=tools,
        latest_user_text="Turn on window defrost.",
    )
    assert action is not None
    assert action.action == "respond"
    assert "front, rear, or all" in action.content.lower()

    action = controller.decide(
        context_id="ctx-defrost-ambiguous",
        messages=messages,
        tools=tools,
        latest_user_text="Front.",
    )
    assert action is not None
    assert action.tool_calls == [{"tool_name": "get_climate_settings", "arguments": {}}]


def test_rear_defrost_controller_does_not_require_front_safety_actions() -> None:
    controller = PolicyAwareController()
    messages = [{"role": "system", "content": ""}]
    tools = defrost_tools()

    action = controller.decide(
        context_id="ctx-defrost-rear",
        messages=messages,
        tools=tools,
        latest_user_text="Turn on the rear window defrost.",
    )

    assert action is not None
    assert action.tool_calls == [
        {
            "tool_name": "set_window_defrost",
            "arguments": {"on": True, "defrost_window": "REAR"},
        }
    ]


def test_front_defrost_controller_stops_when_ac_tool_is_missing() -> None:
    controller = PolicyAwareController()
    messages = [{"role": "system", "content": ""}]
    tools = [tool for tool in defrost_tools() if tool["function"]["name"] != "set_air_conditioning"]

    controller.decide(
        context_id="ctx-defrost-missing-ac",
        messages=messages,
        tools=tools,
        latest_user_text="Turn on the front defrost.",
    )
    controller.decide(
        context_id="ctx-defrost-missing-ac",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_climate_settings",
                climate_result(
                    fan_speed=2,
                    fan_airflow_direction="WINDSHIELD",
                    air_conditioning=False,
                ),
            )
        ],
    )
    action = controller.decide(
        context_id="ctx-defrost-missing-ac",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result("get_vehicle_window_positions", window_positions(opened=0))
        ],
    )

    assert action is not None
    assert action.action == "respond"
    assert "air conditioning control is unavailable" in action.content.lower()


def test_front_defrost_controller_stops_when_window_status_is_partial() -> None:
    controller = PolicyAwareController()
    messages = [{"role": "system", "content": ""}]
    tools = defrost_tools()

    controller.decide(
        context_id="ctx-defrost-partial-windows",
        messages=messages,
        tools=tools,
        latest_user_text="Turn on the front defrost.",
    )
    controller.decide(
        context_id="ctx-defrost-partial-windows",
        messages=messages,
        tools=tools,
        latest_tool_results=[tool_result("get_climate_settings", climate_result())],
    )
    action = controller.decide(
        context_id="ctx-defrost-partial-windows",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_vehicle_window_positions",
                {
                    "window_driver_rear_position": 0,
                    "window_passenger_rear_position": 0,
                },
            )
        ],
    )

    assert action is not None
    assert action.action == "respond"
    assert "couldn't determine all window positions" in action.content.lower()


def test_window_match_to_passenger_rear_then_front_defrost() -> None:
    controller = PolicyAwareController()
    messages = [{"role": "system", "content": ""}]
    tools = defrost_tools()

    action = controller.decide(
        context_id="ctx-window-match-front-defrost",
        messages=messages,
        tools=tools,
        latest_user_text=(
            "Close all windows to match the current position of the passenger "
            "rear window, then activate the front window defrost."
        ),
    )
    assert action is not None
    assert action.tool_calls == [{"tool_name": "get_vehicle_window_positions", "arguments": {}}]

    action = controller.decide(
        context_id="ctx-window-match-front-defrost",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_vehicle_window_positions",
                {
                    "window_driver_position": 25,
                    "window_passenger_position": 50,
                    "window_driver_rear_position": 25,
                    "window_passenger_rear_position": 5,
                },
            )
        ],
    )
    assert action is not None
    assert action.tool_calls == [{"tool_name": "get_climate_settings", "arguments": {}}]

    action = controller.decide(
        context_id="ctx-window-match-front-defrost",
        messages=messages,
        tools=tools,
        latest_tool_results=[tool_result("get_climate_settings", climate_result())],
    )
    assert action is not None
    assert action.tool_calls == [
        {"tool_name": "open_close_window", "arguments": {"window": "ALL", "percentage": 5}}
    ]

    action = controller.decide(
        context_id="ctx-window-match-front-defrost",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result("open_close_window", {"window": "ALL", "percentage": 5})
        ],
    )
    assert action is not None
    assert action.tool_calls == [
        {"tool_name": "set_fan_speed", "arguments": {"level": 2}}
    ]

    action = controller.decide(
        context_id="ctx-window-match-front-defrost-followup",
        messages=messages,
        tools=tools,
        latest_user_text=(
            "Were all the windows adjusted to match the passenger rear window?"
        ),
    )
    assert action is not None
    assert action.tool_calls == [{"tool_name": "get_vehicle_window_positions", "arguments": {}}]

    action = controller.decide(
        context_id="ctx-window-match-front-defrost-followup",
        messages=messages,
        tools=tools,
        latest_tool_results=[
            tool_result(
                "get_vehicle_window_positions",
                {
                    "window_driver_position": 5,
                    "window_passenger_position": 5,
                    "window_driver_rear_position": 5,
                    "window_passenger_rear_position": 5,
                },
            )
        ],
    )
    assert action is not None
    assert action.action == "respond"
    assert "all windows match the passenger rear window" in action.content.lower()
