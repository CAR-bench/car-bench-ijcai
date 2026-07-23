from carbench_agent_core import ToolIndex
from carbench_agent_core.tool_index import ValidationIssue, _compile_validator


def fake_tool(
    name: str,
    *,
    properties: dict | None = None,
    required: list[str] | None = None,
    additional_properties: bool | dict | None = False,
) -> dict:
    parameters = {
        "type": "object",
        "properties": properties or {},
        "required": required or [],
    }
    if additional_properties is not None:
        parameters["additionalProperties"] = additional_properties
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "",
            "parameters": parameters,
        },
    }


def test_tool_index_validation_uses_user_safe_missing_capability_template() -> None:
    index = ToolIndex([])

    error = index.validate_call("missing_tool", {})

    assert (
        error
        == "I don't currently have that capability, so I can't complete this request."
    )
    assert "missing_tool" not in error


def test_tool_index_ignores_non_function_tool_declarations() -> None:
    declaration = fake_tool("not_callable")
    declaration["type"] = "resource"

    index = ToolIndex([declaration])

    assert not index.has("not_callable")


def test_tool_index_validation_rejects_unknown_arguments_without_leaking_names() -> (
    None
):
    index = ToolIndex(
        [
            fake_tool(
                "set_control",
                properties={"on": {"type": "boolean"}},
            )
        ]
    )

    error = index.validate_call("set_control", {"on": True, "extra": "value"})

    assert error == "I can't complete this request with the available controls."
    assert "extra" not in error


def test_tool_index_validation_rejects_arguments_for_no_arg_tool() -> None:
    index = ToolIndex([fake_tool("get_status")])

    error = index.validate_call("get_status", {"": {}})

    assert error == "I can't complete this request with the available controls."


def test_tool_index_validation_rejects_missing_required_argument() -> None:
    index = ToolIndex(
        [
            fake_tool(
                "set_control",
                properties={"on": {"type": "boolean"}},
                required=["on"],
            )
        ]
    )

    error = index.validate_call("set_control", {})

    assert error == "I need more information before I can complete this request."


def test_tool_index_validation_rejects_enum_and_type_mismatch() -> None:
    index = ToolIndex(
        [
            fake_tool(
                "set_mode",
                properties={
                    "mode": {"type": "string", "enum": ["AUTO", "MANUAL"]},
                    "level": {"type": "integer"},
                },
            )
        ]
    )

    assert (
        index.validate_call("set_mode", {"mode": "SPORT", "level": 1})
        == "I can't complete this request with the available controls."
    )
    assert (
        index.validate_call("set_mode", {"mode": "AUTO", "level": "high"})
        == "I can't complete this request with the available controls."
    )


def test_tool_index_exposes_structured_validation_issue() -> None:
    index = ToolIndex(
        [
            fake_tool(
                "set_level",
                properties={"level": {"type": "integer"}},
                required=["level"],
            )
        ]
    )

    issue = index.validation_issue("set_level", {})

    assert isinstance(issue, ValidationIssue)
    assert issue.code == "missing_required_argument"
    assert issue.instance_path == ("level",)
    assert issue.constraint == "required"
    assert issue.repair_hint == "Provide the required argument at $.level."
    assert index.validate_call("set_level", {}) == issue.user_message


def test_tool_index_validates_numeric_draft_2020_12_constraints() -> None:
    index = ToolIndex(
        [
            fake_tool(
                "set_level",
                properties={
                    "level": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 3,
                        "multipleOf": 0.5,
                    }
                },
                required=["level"],
            )
        ]
    )

    assert index.validation_issue("set_level", {"level": 2.5}) is None
    assert index.validation_issue("set_level", {"level": -0.5}).code == "out_of_range"
    assert index.validation_issue("set_level", {"level": 3.5}).code == "out_of_range"
    issue = index.validation_issue("set_level", {"level": 1.25})
    assert issue.code == "out_of_range"
    assert issue.constraint == "multipleOf"
    assert issue.instance_path == ("level",)


def test_tool_index_validates_nested_objects_and_arrays() -> None:
    index = ToolIndex(
        [
            fake_tool(
                "configure_zones",
                properties={
                    "config": {
                        "type": "object",
                        "required": ["zones"],
                        "properties": {
                            "zones": {
                                "type": "array",
                                "minItems": 1,
                                "uniqueItems": True,
                                "items": {
                                    "type": "string",
                                    "enum": ["DRIVER", "PASSENGER"],
                                },
                            }
                        },
                        "additionalProperties": False,
                    }
                },
                required=["config"],
            )
        ]
    )

    assert (
        index.validation_issue(
            "configure_zones",
            {"config": {"zones": ["DRIVER", "PASSENGER"]}},
        )
        is None
    )
    missing = index.validation_issue("configure_zones", {"config": {}})
    assert missing.code == "missing_required_argument"
    assert missing.instance_path == ("config",)
    invalid_item = index.validation_issue(
        "configure_zones",
        {"config": {"zones": ["REAR"]}},
    )
    assert invalid_item.code == "unsupported_value"
    assert invalid_item.instance_path == ("config", "zones", 0)
    duplicate = index.validation_issue(
        "configure_zones",
        {"config": {"zones": ["DRIVER", "DRIVER"]}},
    )
    assert duplicate.code == "schema_violation"
    assert duplicate.constraint == "uniqueItems"
    nested_extra = index.validation_issue(
        "configure_zones",
        {"config": {"zones": ["DRIVER"], "extra": True}},
    )
    assert nested_extra.code == "unexpected_argument"
    assert nested_extra.instance_path == ("config",)


def test_tool_index_validates_composed_schema_and_format() -> None:
    index = ToolIndex(
        [
            fake_tool(
                "send_to_target",
                properties={
                    "target": {
                        "oneOf": [
                            {"const": "AUTO"},
                            {"type": "integer", "minimum": 1},
                        ]
                    },
                    "email": {"type": "string", "format": "email"},
                },
                required=["target", "email"],
            )
        ]
    )

    assert (
        index.validation_issue(
            "send_to_target",
            {"target": "AUTO", "email": "driver@example.com"},
        )
        is None
    )
    assert (
        index.validation_issue(
            "send_to_target",
            {"target": 2, "email": "driver@example.com"},
        )
        is None
    )
    invalid_target = index.validation_issue(
        "send_to_target",
        {"target": False, "email": "driver@example.com"},
    )
    assert invalid_target is not None
    assert invalid_target.instance_path[:1] == ("target",)
    invalid_email = index.validation_issue(
        "send_to_target",
        {"target": "AUTO", "email": "not-an-email"},
    )
    assert invalid_email.code == "invalid_format"
    assert invalid_email.constraint == "format"


def test_tool_index_caches_compiled_dynamic_schemas() -> None:
    _compile_validator.cache_clear()
    tool = fake_tool(
        "set_mode",
        properties={"mode": {"type": "string", "enum": ["AUTO", "MANUAL"]}},
        required=["mode"],
    )

    assert ToolIndex([tool]).validation_issue("set_mode", {"mode": "AUTO"}) is None
    after_first = _compile_validator.cache_info()
    assert ToolIndex([tool]).validation_issue("set_mode", {"mode": "MANUAL"}) is None
    after_second = _compile_validator.cache_info()

    assert after_first.misses == 1
    assert after_second.misses == 1
    assert after_second.hits == 1


def test_tool_index_rejects_invalid_dynamic_schema_without_crashing() -> None:
    tool = fake_tool(
        "broken_tool",
        properties={"level": {"type": "not-a-json-schema-type"}},
    )

    issue = ToolIndex([tool]).validation_issue("broken_tool", {"level": 1})

    assert issue.code == "invalid_tool_schema"
    assert issue.constraint == "schema"
    assert (
        issue.user_message
        == "I can't complete this request with the available controls."
    )


def test_json_schema_omitted_additional_properties_allows_extra_fields() -> None:
    permissive = ToolIndex(
        [
            fake_tool(
                "configure",
                properties={"mode": {"type": "string"}},
                required=["mode"],
                additional_properties=None,
            )
        ]
    )
    strict = ToolIndex(
        [
            fake_tool(
                "configure",
                properties={"mode": {"type": "string"}},
                required=["mode"],
                additional_properties=False,
            )
        ]
    )

    arguments = {"mode": "eco", "extension": {"level": 2}}

    assert permissive.validation_issue("configure", arguments) is None
    issue = strict.validation_issue("configure", arguments)
    assert issue is not None
    assert issue.code == "unexpected_argument"
    assert issue.constraint == "additionalProperties"
