"""CAR-bench Track 2 agent using direct Cerebras SDK inference."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from a2a.helpers.proto_helpers import new_data_part, new_message, new_text_part
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import Role
from google.protobuf.json_format import MessageToDict

sys.path.insert(0, str(Path(__file__).parent.parent))
from carbench_agent_core import NextAction, PolicyAwareController, ToolIndex
from carbench_agent_core.response_renderer import (
    clean_user_content as _clean_user_content,
    render_malformed_action,
    render_malformed_tool_arguments,
    render_malformed_tool_call,
)
from logging_utils import configure_logger
from tool_call_types import ToolCall, ToolCallsData
from turn_metrics import (
    AVG_LLM_CALL_TIME_MS,
    COMPLETION_TOKENS,
    COST,
    MODEL,
    NUM_LLM_CALLS,
    NUM_PASSES,
    PROMPT_TOKENS,
    QUOTA_WAIT_TIME_MS,
    THINKING_TOKENS,
    TURN_METRICS_KEY,
)

sys.path.pop(0)

if __package__:
    from .cerebras_client import (
        DEFAULT_CEREBRAS_API_BASE,
        DEFAULT_EXECUTOR_MODEL,
        DEFAULT_EXECUTOR_REASONING_EFFORT,
        CerebrasCompletionClient,
        CerebrasTemplateError,
        MalformedModelResponseError,
        TokenUsage,
        add_token_usage,
    )
else:
    from cerebras_client import (
        DEFAULT_CEREBRAS_API_BASE,
        DEFAULT_EXECUTOR_MODEL,
        DEFAULT_EXECUTOR_REASONING_EFFORT,
        CerebrasCompletionClient,
        CerebrasTemplateError,
        MalformedModelResponseError,
        TokenUsage,
        add_token_usage,
    )


logger = configure_logger(role="agent_under_test", context="-")


def _payload_from_next_action(action: NextAction) -> dict[str, Any]:
    if action.action == "respond":
        return {"action": "respond", "content": action.content, "tool_calls": []}
    return {
        "action": "tool_calls",
        "content": "",
        "tool_calls": [
            {
                "tool_name": tool_call["tool_name"],
                "arguments": tool_call.get("arguments") or {},
            }
            for tool_call in action.tool_calls
        ],
    }


def _validated_next_action(
    next_action: dict[str, Any],
    *,
    tool_index: ToolIndex,
) -> dict[str, Any]:
    if next_action.get("action") == "respond":
        return {
            "action": "respond",
            "content": _clean_user_content(next_action.get("content") or ""),
            "tool_calls": [],
        }

    if next_action.get("action") != "tool_calls":
        return {
            "action": "respond",
            "content": render_malformed_action(),
            "tool_calls": [],
        }

    tool_calls = next_action.get("tool_calls")
    if not isinstance(tool_calls, list) or not tool_calls:
        return {
            "action": "respond",
            "content": render_malformed_tool_call(),
            "tool_calls": [],
        }

    validated_calls = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            return {
                "action": "respond",
                "content": render_malformed_tool_call(),
                "tool_calls": [],
            }
        name = tool_call.get("tool_name")
        arguments = tool_call.get("arguments") or {}
        if not isinstance(name, str) or not name:
            return {
                "action": "respond",
                "content": render_malformed_tool_call(),
                "tool_calls": [],
            }
        if not isinstance(arguments, dict):
            return {
                "action": "respond",
                "content": render_malformed_tool_arguments(),
                "tool_calls": [],
            }

        validation_error = tool_index.validate_call(name, arguments)
        if validation_error:
            return {
                "action": "respond",
                "content": validation_error,
                "tool_calls": [],
            }
        validated_calls.append({"tool_name": name, "arguments": arguments})

    return {"action": "tool_calls", "content": "", "tool_calls": validated_calls}


@dataclass
class AgentInferenceResult:
    """Internal result for one benchmark-visible assistant step."""

    next_action: dict[str, Any]
    elapsed_ms: float
    token_usage: TokenUsage | None = None
    cost: float = 0.0
    internal_calls: int = 1
    quota_wait_ms: float = 0.0


class CARBenchAgentExecutor(AgentExecutor):
    """A2A executor that asks a Cerebras model for one CAR-bench next action."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_EXECUTOR_MODEL,
        api_base: str = DEFAULT_CEREBRAS_API_BASE,
        service_tier: str | None = None,
        temperature: float | None = None,
        reasoning_effort: str | None = DEFAULT_EXECUTOR_REASONING_EFFORT,
        max_completion_tokens: int = 1024,
        malformed_retries: int = 1,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.max_completion_tokens = max_completion_tokens
        self.malformed_retries = malformed_retries
        self.client = CerebrasCompletionClient(
            api_base=api_base,
            service_tier=service_tier,
            logger=logger.bind(role="agent_under_test", context="cerebras"),
        )
        self.ctx_id_to_messages: dict[str, list[dict[str, Any]]] = {}
        self.ctx_id_to_tools: dict[str, list[dict[str, Any]]] = {}
        self.ctx_id_to_turn_metrics: dict[str, dict[str, Any]] = {}
        self.policy_controller = PolicyAwareController()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        inbound_message = context.message
        ctx_logger = logger.bind(
            role="agent_under_test",
            context=f"ctx:{context.context_id[:8]}",
        )

        if context.context_id not in self.ctx_id_to_messages:
            self.ctx_id_to_messages[context.context_id] = []

        messages = self.ctx_id_to_messages[context.context_id]
        tools = self.ctx_id_to_tools.get(context.context_id, [])

        try:
            user_message_text, incoming_tool_results = self._parse_inbound_parts(
                inbound_message,
                context,
                messages,
            )
            tools_from_msg = self._extract_tools(inbound_message)
            if tools_from_msg is not None:
                tools = tools_from_msg
                self.ctx_id_to_tools[context.context_id] = tools

            ctx_logger.info(
                "Received message",
                turn=len(messages) + 1,
                has_tools=bool(tools),
                num_tools=len(tools) if tools else 0,
                has_tool_results=bool(incoming_tool_results),
                message_preview=(
                    user_message_text[:100]
                    if user_message_text
                    else f"[{len(incoming_tool_results)} tool results]"
                    if incoming_tool_results
                    else ""
                ),
            )

            self._append_inbound_to_history(
                messages=messages,
                user_message_text=user_message_text,
                incoming_tool_results=incoming_tool_results,
            )

            tool_index = ToolIndex(tools)
            controlled_action = self.policy_controller.decide(
                context_id=context.context_id,
                messages=messages,
                tools=tools,
                latest_user_text=(
                    user_message_text if not incoming_tool_results else None
                ),
                latest_tool_results=incoming_tool_results,
            )

            if controlled_action is not None:
                inference_result = AgentInferenceResult(
                    next_action=_validated_next_action(
                        _payload_from_next_action(controlled_action),
                        tool_index=tool_index,
                    ),
                    elapsed_ms=0.0,
                    token_usage=None,
                    cost=0.0,
                    internal_calls=0,
                    quota_wait_ms=0.0,
                )
                ctx_logger.info(
                    "Policy controller selected action",
                    action=controlled_action.action,
                    reason=controlled_action.reason,
                    num_tool_calls=len(controlled_action.tool_calls),
                    pending_obligations=self.policy_controller.pending_obligations(
                        context.context_id
                    ),
                )
                self._on_policy_controlled_action(
                    context.context_id,
                    controlled_action,
                )
            else:
                inference_result = await asyncio.to_thread(
                    self._call_model_with_retries,
                    context_id=context.context_id,
                    messages=messages,
                    tools=tools,
                    ctx_logger=ctx_logger,
                )
                inference_result.next_action = _validated_next_action(
                    inference_result.next_action,
                    tool_index=tool_index,
                )

            parts, assistant_message_for_history = self._build_a2a_response_parts(
                inference_result.next_action
            )
            messages.append(assistant_message_for_history)

            self._record_turn_metrics(
                context.context_id,
                inference_result.elapsed_ms,
                token_usage=inference_result.token_usage,
                cost=inference_result.cost,
                internal_calls=inference_result.internal_calls,
                quota_wait_ms=inference_result.quota_wait_ms,
            )
            response_message = new_message(
                parts=parts,
                context_id=context.context_id,
                role=Role.ROLE_AGENT,
            )

            has_tool_calls = bool(assistant_message_for_history.get("tool_calls"))
            if not has_tool_calls and context.context_id in self.ctx_id_to_turn_metrics:
                metrics = self._public_turn_metrics(
                    self.ctx_id_to_turn_metrics.pop(context.context_id)
                )
                response_message.metadata.update({TURN_METRICS_KEY: metrics})

            await event_queue.enqueue_event(response_message)

        except Exception as exc:
            ctx_logger.error("Cerebras agent error", error=str(exc), exc_info=True)
            response_message = new_message(
                parts=[
                    new_text_part(
                        "I'm sorry, I couldn't process that request safely."
                    )
                ],
                context_id=context.context_id,
                role=Role.ROLE_AGENT,
            )
            await event_queue.enqueue_event(response_message)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        logger.bind(
            role="agent_under_test",
            context=f"ctx:{context.context_id[:8]}",
        ).info("Canceling context")
        self.ctx_id_to_messages.pop(context.context_id, None)
        self.ctx_id_to_tools.pop(context.context_id, None)
        self.ctx_id_to_turn_metrics.pop(context.context_id, None)
        self.policy_controller.reset(context.context_id)

    def _call_model_with_retries(
        self,
        *,
        context_id: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        ctx_logger,
    ) -> AgentInferenceResult:
        last_error: Exception | None = None
        correction = None
        total_duration_ms = 0.0
        total_token_usage: TokenUsage | None = None
        total_cost = 0.0
        total_quota_wait_ms = 0.0
        internal_calls = 0

        for attempt in range(self.malformed_retries + 1):
            prompt = build_next_action_prompt(
                messages=messages,
                tools=tools,
                correction=correction,
            )
            prompt_sizes = prompt_component_char_counts(messages=messages, tools=tools)
            ctx_logger.debug(
                "Calling Cerebras executor",
                attempt=attempt + 1,
                model=self.model,
                num_messages=len(messages),
                num_tools=len(tools),
                prompt_chars=len(prompt),
                system_prompt_chars=prompt_sizes["system_prompt_chars"],
                tool_schema_prompt_chars=prompt_sizes["tool_schema_prompt_chars"],
                transcript_prompt_chars=prompt_sizes["transcript_prompt_chars"],
                max_completion_tokens=self.max_completion_tokens,
                reasoning_effort=self.reasoning_effort,
                tool_names=[
                    tool.get("function", {}).get("name", "<unknown>")
                    for tool in tools[:10]
                ],
            )
            try:
                result = self.client.generate(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": CEREBRAS_DEVELOPER_INSTRUCTIONS,
                        },
                        {"role": "user", "content": prompt},
                    ],
                    response_schema=NEXT_ACTION_OUTPUT_SCHEMA,
                    response_schema_name="next_action",
                    max_completion_tokens=self.max_completion_tokens,
                    temperature=self.temperature,
                    reasoning_effort=self.reasoning_effort,
                )
                internal_calls += 1
                total_duration_ms += result.duration_ms
                total_cost += result.cost
                total_token_usage = add_token_usage(
                    total_token_usage,
                    result.token_usage,
                )
                total_quota_wait_ms += result.quota_wait_ms
                parsed = parse_next_action(result.text)
                ctx_logger.info(
                    "Cerebras response received",
                    action=parsed["action"],
                    num_tool_calls=len(parsed.get("tool_calls") or []),
                    model=result.model,
                    inference_ms=round(result.duration_ms, 1),
                    total_inference_ms=round(total_duration_ms, 1),
                    estimated_request_tokens=result.estimated_request_tokens,
                    cerebras_rate_limit_headers=(
                        result.rate_limit_headers.as_dict()
                        if result.rate_limit_headers is not None
                        else None
                    ),
                    input_tokens=(
                        total_token_usage.input_tokens
                        if total_token_usage is not None
                        else 0
                    ),
                    cached_input_tokens=(
                        total_token_usage.cached_input_tokens
                        if total_token_usage is not None
                        else 0
                    ),
                    output_tokens=(
                        total_token_usage.output_tokens
                        if total_token_usage is not None
                        else 0
                    ),
                    reasoning_tokens=(
                        total_token_usage.reasoning_output_tokens
                        if total_token_usage is not None
                        else 0
                    ),
                    attempt=attempt + 1,
                    quota_wait_ms=round(result.quota_wait_ms, 1),
                )
                return AgentInferenceResult(
                    next_action=parsed,
                    elapsed_ms=total_duration_ms,
                    token_usage=total_token_usage,
                    cost=total_cost,
                    internal_calls=max(internal_calls, 1),
                    quota_wait_ms=total_quota_wait_ms,
                )
            except (MalformedModelResponseError, json.JSONDecodeError) as exc:
                last_error = exc
                correction = (
                    "The previous model output was invalid. Return one JSON "
                    f"object matching the schema. Error: {exc}"
                )
                ctx_logger.warning(
                    "Malformed Cerebras response",
                    attempt=attempt + 1,
                    retrying=attempt < self.malformed_retries,
                    error=str(exc),
                )
            except CerebrasTemplateError:
                raise

        raise MalformedModelResponseError(
            f"Cerebras did not produce a valid next-action JSON object: {last_error}"
        )

    def _on_policy_controlled_action(
        self,
        context_id: str,
        controlled_action: NextAction,
    ) -> None:
        """Hook for subclasses that keep model-side per-context state."""
        return None

    def _parse_inbound_parts(
        self,
        inbound_message,
        context: RequestContext,
        messages: list[dict[str, Any]],
    ) -> tuple[str | None, list[dict[str, Any]] | None]:
        user_message_text = None
        incoming_tool_results = None

        for part in inbound_message.parts:
            content_type = part.WhichOneof("content")
            if content_type == "text":
                text = part.text
                if "System:" in text and "\n\nUser:" in text:
                    parts_split = text.split("\n\nUser:", 1)
                    system_prompt = parts_split[0].replace("System:", "").strip()
                    user_message_text = parts_split[1].strip()
                    if not messages:
                        messages.append({"role": "system", "content": system_prompt})
                else:
                    user_message_text = text
            elif content_type == "data":
                data = MessageToDict(part.data)
                if "tool_results" in data:
                    incoming_tool_results = data["tool_results"]

        if not user_message_text and not incoming_tool_results:
            user_message_text = context.get_user_input()

        return user_message_text, incoming_tool_results

    @staticmethod
    def _extract_tools(inbound_message) -> list[dict[str, Any]] | None:
        for part in inbound_message.parts:
            if part.WhichOneof("content") != "data":
                continue
            data = MessageToDict(part.data)
            if "tools" in data:
                return data["tools"]
        return None

    @staticmethod
    def _append_inbound_to_history(
        *,
        messages: list[dict[str, Any]],
        user_message_text: str | None,
        incoming_tool_results: list[dict[str, Any]] | None,
    ) -> None:
        if (
            messages
            and messages[-1].get("role") == "assistant"
            and messages[-1].get("tool_calls")
        ):
            prev_tool_calls = messages[-1]["tool_calls"]
            tool_results = _format_tool_results(
                prev_tool_calls=prev_tool_calls,
                incoming_tool_results=incoming_tool_results,
                fallback_text=user_message_text,
            )
            messages.extend(tool_results)
        else:
            messages.append({"role": "user", "content": user_message_text or ""})

    @staticmethod
    def _build_a2a_response_parts(
        assistant_content: dict[str, Any],
    ) -> tuple[list[Any], dict[str, Any]]:
        action = assistant_content["action"]
        if action == "respond":
            content = assistant_content.get("content", "")
            return [new_text_part(content)], {"role": "assistant", "content": content}

        tool_calls_for_history = []
        tool_calls_data = []
        for tool_call in assistant_content["tool_calls"]:
            call_id = f"call_{uuid4().hex[:12]}"
            name = tool_call["tool_name"]
            arguments = tool_call.get("arguments") or {}
            argument_json = json.dumps(arguments, separators=(",", ":"))
            tool_calls_for_history.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": argument_json,
                    },
                }
            )
            tool_calls_data.append(ToolCall(tool_name=name, arguments=arguments))

        parts = [new_data_part(ToolCallsData(tool_calls=tool_calls_data).model_dump())]
        return parts, {
            "role": "assistant",
            "content": None,
            "tool_calls": tool_calls_for_history,
        }

    def _record_turn_metrics(
        self,
        context_id: str,
        elapsed_ms: float,
        *,
        token_usage: TokenUsage | None = None,
        cost: float = 0.0,
        internal_calls: int = 1,
        quota_wait_ms: float = 0.0,
    ) -> None:
        metrics = self.ctx_id_to_turn_metrics.setdefault(
            context_id,
            {
                PROMPT_TOKENS: 0,
                COMPLETION_TOKENS: 0,
                COST: 0.0,
                MODEL: self.model,
                THINKING_TOKENS: 0,
                NUM_LLM_CALLS: 0,
                NUM_PASSES: 1,
                QUOTA_WAIT_TIME_MS: 0.0,
                "_total_llm_time_ms": 0.0,
            },
        )
        internal_calls = max(internal_calls, 0)
        metrics[NUM_LLM_CALLS] += internal_calls
        if token_usage is not None:
            metrics[PROMPT_TOKENS] += token_usage.input_tokens
            metrics[COMPLETION_TOKENS] += token_usage.output_tokens
            metrics[THINKING_TOKENS] += token_usage.reasoning_output_tokens
        metrics[COST] += cost
        metrics["_total_llm_time_ms"] += elapsed_ms
        metrics[QUOTA_WAIT_TIME_MS] += quota_wait_ms
        num_calls = metrics[NUM_LLM_CALLS]
        metrics[AVG_LLM_CALL_TIME_MS] = (
            round(metrics["_total_llm_time_ms"] / num_calls, 1) if num_calls else 0.0
        )
        metrics[NUM_PASSES] = max(metrics.get(NUM_PASSES, 1), max(internal_calls, 1))

    @staticmethod
    def _public_turn_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
        public_metrics = dict(metrics)
        public_metrics.pop("_total_llm_time_ms", None)
        return public_metrics


def _format_tool_results(
    *,
    prev_tool_calls: list[dict[str, Any]],
    incoming_tool_results: list[dict[str, Any]] | None,
    fallback_text: str | None,
) -> list[dict[str, Any]]:
    if not incoming_tool_results:
        return [
            {
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": fallback_text or "",
            }
            for tc in prev_tool_calls
        ]

    tool_call_by_name: dict[str, list[dict[str, Any]]] = {}
    for tc in prev_tool_calls:
        name = tc["function"]["name"]
        tool_call_by_name.setdefault(name, []).append(tc)

    tool_results = []
    for tr in incoming_tool_results:
        if not isinstance(tr, dict):
            tr = MessageToDict(tr)
        tr_name = tr.get("tool_name", tr.get("toolName", ""))
        matching_calls = tool_call_by_name.get(tr_name, [])
        if matching_calls:
            matched_tc = matching_calls.pop(0)
            tool_call_id = matched_tc["id"]
        else:
            tool_call_id = tr.get(
                "tool_call_id",
                tr.get("toolCallId", f"unknown_{tr_name}"),
            )
        tool_results.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": tr_name,
                "content": tr.get("content", ""),
            }
        )
    return tool_results


def build_next_action_prompt(
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    correction: str | None = None,
) -> str:
    prompt = {
        "task": "Choose exactly one next assistant action for this CAR-bench turn.",
        "available_tools": compact_tools_for_prompt(tools),
        "conversation_transcript": _messages_for_prompt(messages),
        "output_contract": {
            "respond": "Use when speaking naturally to the user.",
            "tool_calls": "Use one or more supplied CAR-bench environment tools.",
        },
        "rules": [
            "Use only the tool definitions in available_tools.",
            "Do not invent tool observations.",
            "If a capability or parameter is unavailable, respond to the user transparently.",
            "If choosing tool_calls, keep content empty and do not explain the tool call.",
            "If asking the user, ask exactly one concrete question.",
            (
                "Before responding, verify that every requested subtask and required "
                "confirmation is complete; otherwise continue with the next tool call "
                "or one clarifying question."
            ),
            "If confirming completion, mention only actions actually completed by tool results.",
            "Do not mention schemas, parameter names, evaluator behavior, or hidden implementation details.",
            "Keep user-facing responses short and TTS-friendly.",
            "Respect all policies in the system/wiki message inside the transcript.",
        ],
    }
    if correction:
        prompt["correction"] = correction
    return json.dumps(prompt, ensure_ascii=False, indent=2)


def prompt_component_char_counts(
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> dict[str, int]:
    """Return safe prompt-size telemetry without retaining user content."""

    transcript = _messages_for_prompt(messages)
    return {
        "system_prompt_chars": sum(
            len(item.get("content") or "")
            for item in transcript
            if item.get("role") == "system"
        ),
        "tool_schema_prompt_chars": len(
            json.dumps(compact_tools_for_prompt(tools), ensure_ascii=False)
        ),
        "transcript_prompt_chars": len(json.dumps(transcript, ensure_ascii=False)),
    }


def compact_tools_for_prompt(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the tool-call contract while removing verbose schema-only metadata.

    The evaluator provides rich tool schemas, including examples and long output
    descriptions. The agent validates model output against the original schema,
    so the prompt needs only the callable function names and input constraints.
    """

    compacted: list[dict[str, Any]] = []
    for tool in tools:
        function = tool.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if not isinstance(name, str) or not name:
            continue
        compact_function: dict[str, Any] = {"name": name}
        if description := _compact_description(function.get("description"), 120):
            compact_function["description"] = description
        parameters = function.get("parameters")
        if isinstance(parameters, dict):
            compact_function["parameters"] = _compact_json_schema(parameters)
        compacted.append({"type": "function", "function": compact_function})
    return compacted


def _compact_json_schema(schema: dict[str, Any], *, depth: int = 0) -> dict[str, Any]:
    """Retain the input constraints required to form a valid tool call."""

    if depth >= 4:
        schema_type = schema.get("type")
        return {"type": schema_type} if isinstance(schema_type, str) else {}

    compact: dict[str, Any] = {}
    for key in ("type", "enum", "minimum", "maximum", "multipleOf", "format"):
        value = schema.get(key)
        if value is not None:
            compact[key] = value
    required = schema.get("required")
    if isinstance(required, list):
        compact["required"] = [item for item in required if isinstance(item, str)]
    if isinstance(schema.get("additionalProperties"), bool):
        compact["additionalProperties"] = schema["additionalProperties"]
    properties = schema.get("properties")
    if isinstance(properties, dict):
        compact["properties"] = {
            str(name): _compact_json_schema(value, depth=depth + 1)
            for name, value in properties.items()
            if isinstance(value, dict)
        }
    items = schema.get("items")
    if isinstance(items, dict):
        compact["items"] = _compact_json_schema(items, depth=depth + 1)
    for key in ("anyOf", "oneOf", "allOf"):
        variants = schema.get(key)
        if isinstance(variants, list):
            compact[key] = [
                _compact_json_schema(value, depth=depth + 1)
                for value in variants
                if isinstance(value, dict)
            ]
    return compact


def _compact_description(value: Any, limit: int) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = " ".join(value.split())
    if not normalized:
        return None
    return normalized[:limit]


def _messages_for_prompt(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Preserve policy and task origin while bounding growing tool transcripts."""

    max_messages = _prompt_history_max_messages()
    selected_messages = messages
    if len(messages) > max_messages:
        selected_indices = {0}
        if len(messages) > 1 and messages[1].get("role") == "user":
            selected_indices.add(1)
        tail_count = max(max_messages - len(selected_indices), 0)
        selected_indices.update(range(max(len(messages) - tail_count, 0), len(messages)))
        selected_messages = [messages[index] for index in sorted(selected_indices)]

    rendered = []
    for message in selected_messages:
        item: dict[str, Any] = {
            "role": message.get("role"),
            "content": message.get("content"),
        }
        if message.get("tool_calls"):
            item["tool_calls"] = [
                {
                    "tool_name": tc.get("function", {}).get("name"),
                    "arguments": _parse_arguments(
                        tc.get("function", {}).get("arguments", {})
                    ),
                }
                for tc in message["tool_calls"]
            ]
        if message.get("role") == "tool":
            item["tool_call_id"] = message.get("tool_call_id")
            item["name"] = message.get("name")
        rendered.append(item)
    return rendered


def _prompt_history_max_messages() -> int:
    value = os.getenv("TRACK2_PROMPT_HISTORY_MAX_MESSAGES", "24")
    try:
        return max(4, int(value))
    except ValueError:
        return 24


def _parse_arguments(arguments: Any) -> Any:
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except json.JSONDecodeError:
            return arguments
    return arguments


def parse_next_action(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise MalformedModelResponseError(f"No JSON object found in: {text[:200]}")
        payload = json.loads(text[start : end + 1])

    if payload.get("action") == "respond":
        content = payload.get("content")
        if not isinstance(content, str):
            raise MalformedModelResponseError("respond action requires string content")
        return {"action": "respond", "content": content}

    if payload.get("action") == "tool_calls":
        tool_calls = payload.get("tool_calls")
        if not isinstance(tool_calls, list) or not tool_calls:
            raise MalformedModelResponseError(
                "tool_calls action requires non-empty tool_calls"
            )
        normalized = []
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                raise MalformedModelResponseError("each tool call must be an object")
            tool_name = tool_call.get("tool_name")
            arguments = tool_call.get("arguments")
            if arguments is None and "arguments_json" in tool_call:
                arguments = _parse_tool_arguments_json(tool_call["arguments_json"])
            if arguments is None:
                arguments = {}
            if not isinstance(tool_name, str) or not tool_name:
                raise MalformedModelResponseError("each tool call requires tool_name")
            if not isinstance(arguments, dict):
                raise MalformedModelResponseError(
                    "tool call arguments must be an object"
                )
            normalized.append({"tool_name": tool_name, "arguments": arguments})
        return {"action": "tool_calls", "tool_calls": normalized}

    raise MalformedModelResponseError("action must be either respond or tool_calls")


def _parse_tool_arguments_json(arguments_json: Any) -> dict[str, Any]:
    if not isinstance(arguments_json, str):
        raise MalformedModelResponseError("tool call arguments_json must be a string")
    if not arguments_json.strip():
        return {}
    try:
        parsed = json.loads(arguments_json)
    except json.JSONDecodeError as exc:
        raise MalformedModelResponseError(
            f"tool call arguments_json is not valid JSON: {exc}"
        ) from exc
    if not isinstance(parsed, dict):
        raise MalformedModelResponseError(
            "tool call arguments_json must decode to an object"
        )
    return parsed


NEXT_ACTION_OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["action", "content", "tool_calls"],
    "properties": {
        "action": {"type": "string", "enum": ["respond", "tool_calls"]},
        "content": {
            "type": "string",
            "description": (
                "Natural user-facing assistant text when action is respond; "
                "otherwise empty."
            ),
        },
        "tool_calls": {
            "type": "array",
            "description": (
                "CAR-bench tool calls when action is tool_calls; otherwise empty."
            ),
            "items": {
                "type": "object",
                "required": ["tool_name", "arguments_json"],
                "properties": {
                    "tool_name": {"type": "string"},
                    "arguments_json": {
                        "type": "string",
                        "description": (
                            "JSON object string containing the tool arguments, "
                            'for example "{}" or "{\\"position\\":50}".'
                        ),
                    },
                },
                "additionalProperties": False,
            },
        },
    },
    "additionalProperties": False,
}


CEREBRAS_DEVELOPER_INSTRUCTIONS = """You are an in-car assistant reasoning layer for CAR-bench.
Use only the supplied CAR-bench tool definitions.
Return only JSON matching the requested schema.
Never invent unavailable tools, parameters, or tool results.
For tool calls, put arguments in arguments_json as a JSON object string.
For tool-call actions, keep content empty and do not explain the tool call.
For missing capability or missing information, tell the user transparently.
When asking the user, ask exactly one concrete question.
When confirming completion, mention only actions actually completed by tool results.
Do not mention schemas, parameter names, evaluator behavior, or hidden implementation details.
Keep spoken responses short, natural, and TTS-friendly.
Respect confirmation and disambiguation policy from the wiki/system prompt."""
