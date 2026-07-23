"""Utilities for validating live tool availability and JSON arguments."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker  # type: ignore[import-untyped]
from jsonschema.exceptions import (  # type: ignore[import-untyped]
    SchemaError,
    ValidationError,
    best_match,
)

from .response_renderer import (
    render_malformed_tool_arguments,
    render_missing_capability,
    render_missing_required_information,
    render_unavailable_control,
)


@dataclass(frozen=True)
class ValidationIssue:
    """Structured, non-sensitive explanation of a rejected tool call.

    ``user_message`` is safe to return to the requesting user. The other
    fields are intended for logging and bounded model repair without retaining
    argument values from the request.
    """

    code: str
    user_message: str
    instance_path: tuple[str | int, ...] = ()
    schema_path: tuple[str | int, ...] = ()
    constraint: str | None = None
    repair_hint: str | None = None


@dataclass(frozen=True)
class ToolIndex:
    """Small read-only index over the currently available tool schema."""

    tools: list[dict[str, Any]]
    _by_name: dict[str, dict[str, Any]] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        by_name: dict[str, dict[str, Any]] = {}
        for tool in self.tools:
            if not isinstance(tool, dict):
                continue
            if tool.get("type") != "function":
                continue
            function = tool.get("function")
            if not isinstance(function, dict):
                continue
            name = function.get("name")
            if isinstance(name, str) and name:
                by_name[name] = tool
        object.__setattr__(
            self,
            "_by_name",
            by_name,
        )

    @property
    def names(self) -> set[str]:
        return set(self._by_name)

    def has(self, name: str) -> bool:
        return name in self._by_name

    def get(self, name: str) -> dict[str, Any] | None:
        return self._by_name.get(name)

    def required_args(self, name: str) -> set[str]:
        params = self.parameter_schema(name)
        required = params.get("required") or []
        return {arg for arg in required if isinstance(arg, str)}

    def arg_names(self, name: str) -> set[str]:
        params = self.parameter_schema(name)
        properties = params.get("properties") or {}
        return {arg for arg in properties if isinstance(arg, str)}

    def arg_schema(self, name: str, arg: str) -> dict[str, Any]:
        params = self.parameter_schema(name)
        properties = params.get("properties") or {}
        schema = properties.get(arg)
        return schema if isinstance(schema, dict) else {}

    def parameter_schema(self, name: str) -> dict[str, Any]:
        """Return the evaluator-provided JSON Schema for one tool's arguments."""

        tool = self.get(name)
        if not tool:
            return {}
        parameters = tool.get("function", {}).get("parameters", {})
        return parameters if isinstance(parameters, dict) else {}

    def validate_call(self, name: str, arguments: dict[str, Any]) -> str | None:
        """Return a user-safe validation error or None if the call is valid.

        This compatibility API intentionally keeps the original string return
        type. New callers that need a machine-readable cause should use
        :meth:`validation_issue`.
        """

        issue = self.validation_issue(name, arguments)
        return issue.user_message if issue is not None else None

    def validation_issue(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> ValidationIssue | None:
        """Validate a dynamic tool call with the full Draft 2020-12 schema."""

        if not self.has(name):
            return ValidationIssue(
                code="missing_capability",
                user_message=render_missing_capability(),
                repair_hint="Choose only a tool exposed in the current tool list.",
            )

        if not isinstance(arguments, dict):
            return ValidationIssue(
                code="malformed_arguments",
                user_message=render_malformed_tool_arguments(),
                constraint="type",
                repair_hint="Return the tool arguments as one JSON object.",
            )

        tool = self.get(name) or {}
        raw_parameters = tool.get("function", {}).get("parameters", {})
        if not isinstance(raw_parameters, dict):
            return ValidationIssue(
                code="invalid_tool_schema",
                user_message=render_unavailable_control(),
                constraint="schema",
                repair_hint="Do not call this tool because its schema is invalid.",
            )

        required_missing = sorted(self.required_args(name) - set(arguments))
        if required_missing:
            missing = required_missing[0]
            return ValidationIssue(
                code="missing_required_argument",
                user_message=render_missing_required_information(),
                instance_path=(missing,),
                constraint="required",
                repair_hint=f"Provide the required argument at {_json_path((missing,))}.",
            )

        params = raw_parameters
        try:
            validator = _validator_for_schema(params)
        except (SchemaError, TypeError, ValueError) as exc:
            return ValidationIssue(
                code="invalid_tool_schema",
                user_message=render_unavailable_control(),
                schema_path=tuple(getattr(exc, "path", ())),
                constraint="schema",
                repair_hint="Do not call this tool because its schema is invalid.",
            )

        error = best_match(validator.iter_errors(arguments))
        if error is not None:
            return _issue_from_schema_error(error)

        return None


def parse_tool_arguments(raw_arguments: Any) -> dict[str, Any] | None:
    """Parse provider-specific tool-call arguments into a dictionary."""

    if isinstance(raw_arguments, dict):
        return raw_arguments
    if raw_arguments is None:
        return {}
    if not isinstance(raw_arguments, str):
        return None
    try:
        parsed = json.loads(raw_arguments)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


_FORMAT_CHECKER = FormatChecker()


def _validator_for_schema(schema: dict[str, Any]) -> Draft202012Validator:
    canonical_schema = json.dumps(
        schema,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return _compile_validator(canonical_schema)


@lru_cache(maxsize=512)
def _compile_validator(canonical_schema: str) -> Draft202012Validator:
    """Compile and cache an immutable validator for a dynamic tool schema."""

    schema = json.loads(canonical_schema)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, format_checker=_FORMAT_CHECKER)


def _issue_from_schema_error(error: ValidationError) -> ValidationIssue:
    instance_path = tuple(error.absolute_path)
    schema_path = tuple(error.absolute_schema_path)
    constraint = str(error.validator) if error.validator is not None else None

    if constraint == "required":
        code = "missing_required_argument"
        user_message = render_missing_required_information()
        repair_hint = (
            f"Provide all required information below {_json_path(instance_path)}."
        )
    elif constraint == "additionalProperties":
        code = "unexpected_argument"
        user_message = render_unavailable_control()
        repair_hint = f"Remove unavailable fields below {_json_path(instance_path)}."
    elif constraint in {"enum", "const"}:
        code = "unsupported_value"
        user_message = render_unavailable_control()
        repair_hint = (
            f"Choose a value allowed by the schema at {_json_path(instance_path)}."
        )
    elif constraint in {
        "exclusiveMaximum",
        "exclusiveMinimum",
        "maximum",
        "minimum",
        "multipleOf",
    }:
        code = "out_of_range"
        user_message = render_unavailable_control()
        repair_hint = (
            f"Use a value within the allowed range at {_json_path(instance_path)}."
        )
    elif constraint in {
        "format",
        "maxLength",
        "minLength",
        "pattern",
    }:
        code = "invalid_format"
        user_message = render_unavailable_control()
        repair_hint = f"Use the required format at {_json_path(instance_path)}."
    elif constraint == "type":
        code = "type_mismatch"
        user_message = render_unavailable_control()
        repair_hint = f"Use the required JSON type at {_json_path(instance_path)}."
    else:
        code = "schema_violation"
        user_message = render_unavailable_control()
        repair_hint = f"Satisfy the tool schema at {_json_path(instance_path)}."

    return ValidationIssue(
        code=code,
        user_message=user_message,
        instance_path=instance_path,
        schema_path=schema_path,
        constraint=constraint,
        repair_hint=repair_hint,
    )


def _json_path(path: tuple[str | int, ...]) -> str:
    rendered = "$"
    for part in path:
        if isinstance(part, int):
            rendered += f"[{part}]"
        elif part.isidentifier():
            rendered += f".{part}"
        else:
            rendered += f"[{json.dumps(part, ensure_ascii=False)}]"
    return rendered
