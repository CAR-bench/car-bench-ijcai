"""
CAR-bench Agent - Agent under test that solves CAR-bench tasks.

This is the agent being tested. It:
1. Receives task descriptions with available tools from the evaluator
2. Decides which tool to call or how to respond
3. Returns responses in the expected JSON format wrapped in <json>...</json> tags
"""

import json
import time
from pathlib import Path
import sys
from dotenv import load_dotenv

load_dotenv()

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.helpers.proto_helpers import new_message, new_text_part, new_data_part
from a2a.types import Role
from google.protobuf.json_format import MessageToDict
from litellm import completion
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).parent.parent))
from logging_utils import configure_logger
from carbench_agent_core import NextAction, PolicyAwareController, ToolIndex
from carbench_agent_core.prompting import model_messages_with_competition_prompt
from carbench_agent_core.response_renderer import (
    clean_user_content as _clean_user_content,
    render_malformed_tool_arguments,
    render_malformed_tool_call,
)
from carbench_agent_core.tool_index import parse_tool_arguments
from tool_call_types import ToolCall, ToolCallsData
from turn_metrics import (
    TURN_METRICS_KEY,
    PROMPT_TOKENS,
    COMPLETION_TOKENS,
    COST,
    MODEL,
    THINKING_TOKENS,
    NUM_LLM_CALLS,
    AVG_LLM_CALL_TIME_MS,
    NUM_PASSES,
)

sys.path.pop(0)

logger = configure_logger(role="agent_under_test", context="-")


def _assistant_content_from_next_action(action: NextAction) -> dict:
    if action.action == "respond":
        return {"content": action.content}

    tool_calls = []
    for tool_call in action.tool_calls:
        call_id = f"call_{uuid4().hex[:12]}"
        arguments = tool_call.get("arguments") or {}
        tool_calls.append(
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": tool_call["tool_name"],
                    "arguments": json.dumps(arguments, separators=(",", ":")),
                },
            }
        )
    return {"content": None, "tool_calls": tool_calls}


def _parts_from_assistant_content(assistant_content: dict) -> list:
    parts = []

    if assistant_content.get("content"):
        parts.append(new_text_part(assistant_content["content"]))

    if assistant_content.get("tool_calls"):
        tool_calls_list = []
        for tc in assistant_content["tool_calls"]:
            arguments = parse_tool_arguments(tc["function"].get("arguments"))
            if arguments is None:
                arguments = {}
            tool_calls_list.append(
                ToolCall(
                    tool_name=tc["function"]["name"],
                    arguments=arguments,
                )
            )
        parts.append(
            new_data_part(ToolCallsData(tool_calls=tool_calls_list).model_dump())
        )

    if not parts:
        parts.append(new_text_part(assistant_content.get("content", "") or ""))

    return parts


def _validated_assistant_content(
    assistant_content: dict,
    *,
    tool_index: ToolIndex,
) -> dict:
    tool_calls = assistant_content.get("tool_calls")
    if not tool_calls:
        return {"content": _clean_user_content(assistant_content.get("content") or "")}

    validated_calls = []
    for tc in tool_calls:
        function = tc.get("function") or {}
        name = function.get("name")
        arguments = parse_tool_arguments(function.get("arguments"))
        if not isinstance(name, str) or not name:
            return {"content": render_malformed_tool_call()}
        if arguments is None:
            return {"content": render_malformed_tool_arguments()}
        validation_error = tool_index.validate_call(name, arguments)
        if validation_error:
            return {"content": validation_error}
        validated_calls.append(
            {
                "id": tc.get("id") or f"call_{uuid4().hex[:12]}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments, separators=(",", ":")),
                },
            }
        )

    return {"content": None, "tool_calls": validated_calls}


class CARBenchAgentExecutor(AgentExecutor):
    """Executor for the CAR-bench agent under test using native tool calling."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        thinking: bool = False,
        reasoning_effort: str = "medium",
        interleaved_thinking: bool = False,
    ):
        self.model = model
        self.temperature = temperature
        self.thinking = thinking
        self.reasoning_effort = reasoning_effort  # Can be 'none', 'disable', 'low', 'medium', 'high', or integer token budget
        self.interleaved_thinking = (
            interleaved_thinking  # Whether to use interleaved thinking
        )
        self.ctx_id_to_messages: dict[str, list[dict]] = {}
        self.ctx_id_to_tools: dict[str, list[dict]] = {}
        # Per-context turn metrics accumulation (reset when final response is sent)
        self.ctx_id_to_turn_metrics: dict[str, dict] = {}
        self.policy_controller = PolicyAwareController()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        inbound_message = context.message
        ctx_logger = logger.bind(
            role="agent_under_test", context=f"ctx:{context.context_id[:8]}"
        )

        # Initialize or get conversation history
        if context.context_id not in self.ctx_id_to_messages:
            self.ctx_id_to_messages[context.context_id] = []

        messages = self.ctx_id_to_messages[context.context_id]
        tools = self.ctx_id_to_tools.get(context.context_id, [])

        # Parse the incoming A2A Message with Parts (now protobuf)
        user_message_text = None
        incoming_tool_results = None  # Structured tool results from evaluator

        try:
            for part in inbound_message.parts:
                content_type = part.WhichOneof("content")
                if content_type == "text":
                    text = part.text
                    # Parse system prompt and user message from formatted text
                    if "System:" in text and "\n\nUser:" in text:
                        # First message with system prompt
                        parts_split = text.split("\n\nUser:", 1)
                        system_prompt = parts_split[0].replace("System:", "").strip()
                        user_message_text = parts_split[1].strip()
                        if not messages:  # Only add system prompt once
                            messages.append(
                                {"role": "system", "content": system_prompt}
                            )
                    else:
                        # Regular user message
                        user_message_text = text

                elif content_type == "data":
                    # Extract tools or tool results from data Part
                    data = MessageToDict(part.data)
                    if "tools" in data:
                        tools = data["tools"]
                        self.ctx_id_to_tools[context.context_id] = tools
                    elif "tool_results" in data:
                        # Structured tool results from the evaluator
                        incoming_tool_results = data["tool_results"]

            # Fallback if no text part and no structured tool results found
            if not user_message_text and not incoming_tool_results:
                user_message_text = context.get_user_input()

            ctx_logger.info(
                "Received user message",
                context_id=context.context_id[:8],
                turn=len(messages) + 1,
                message_preview=(
                    user_message_text[:100]
                    if user_message_text
                    else f"[{len(incoming_tool_results)} tool results]"
                    if incoming_tool_results
                    else ""
                ),
            )
            ctx_logger.debug(
                "Message details",
                context_id=context.context_id[:8],
                message=user_message_text,
                num_parts=len(inbound_message.parts),
                has_tools=bool(tools),
                num_tools=len(tools) if tools else 0,
                has_tool_results=bool(incoming_tool_results),
                num_tool_results=len(incoming_tool_results)
                if incoming_tool_results
                else 0,
            )

        except Exception as e:
            logger.warning(f"Failed to parse message parts: {e}, using fallback")
            user_message_text = context.get_user_input()

        # Check if previous message had tool calls - if so, format as tool results
        if (
            messages
            and messages[-1].get("role") == "assistant"
            and messages[-1].get("tool_calls")
        ):
            prev_tool_calls = messages[-1]["tool_calls"]

            if incoming_tool_results:
                # Structured tool results from evaluator — match each result
                # to its corresponding tool_call_id by tool name
                tool_call_by_name = {}
                for tc in prev_tool_calls:
                    name = tc["function"]["name"]
                    # If multiple calls to the same tool, use a list
                    tool_call_by_name.setdefault(name, []).append(tc)

                tool_results = []
                for tr in incoming_tool_results:
                    tr_name = (
                        tr.get("tool_name", "")
                        if isinstance(tr, dict)
                        else tr.get("toolName", "")
                    )
                    matching_calls = tool_call_by_name.get(tr_name, [])
                    if matching_calls:
                        # Pop the first matching call to handle duplicate tool names
                        matched_tc = matching_calls.pop(0)
                        tool_results.append(
                            {
                                "role": "tool",
                                "tool_call_id": matched_tc["id"],
                                "name": tr_name,
                                "content": tr.get("content", ""),
                            }
                        )
                    else:
                        # Fallback: no matching tool_call found, use first unmatched
                        ctx_logger.warning(
                            "No matching tool_call_id for tool result",
                            tool_name=tr_name,
                        )
                        tool_results.append(
                            {
                                "role": "tool",
                                "tool_call_id": tr.get(
                                    "tool_call_id",
                                    tr.get("toolCallId", f"unknown_{tr_name}"),
                                ),
                                "name": tr_name,
                                "content": tr.get("content", ""),
                            }
                        )
            else:
                # Fallback: no structured tool results, use the text message
                # for all tool calls (legacy behavior)
                tool_results = []
                for tc in prev_tool_calls:
                    tool_results.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "name": tc.get("function", {}).get("name", ""),
                            "content": user_message_text or "",
                        }
                    )

            # Add all tool result messages
            messages.extend(tool_results)

            ctx_logger.debug(
                "Formatted tool results",
                num_tools=len(tool_results),
                tool_call_ids=[tr["tool_call_id"] for tr in tool_results],
            )
        else:
            # Regular user message
            messages.append({"role": "user", "content": user_message_text})

        controlled_action = self.policy_controller.decide(
            context_id=context.context_id,
            messages=messages,
            tools=tools,
            latest_user_text=user_message_text if not incoming_tool_results else None,
            latest_tool_results=incoming_tool_results,
        )

        if controlled_action is not None:
            assistant_content = _assistant_content_from_next_action(controlled_action)
            assistant_content = _validated_assistant_content(
                assistant_content,
                tool_index=ToolIndex(tools),
            )
            parts = _parts_from_assistant_content(assistant_content)
            ctx_logger.info(
                "Policy controller selected action",
                action=controlled_action.action,
                reason=controlled_action.reason,
                num_tool_calls=len(controlled_action.tool_calls),
            )
        else:
            # Call LLM with native tool calling
            try:
                # Configure prompt caching (guard against empty lists)
                if tools:
                    tools[-1]["function"]["cache_control"] = {"type": "ephemeral"}
                if messages:
                    messages[0]["cache_control"] = {"type": "ephemeral"}

                completion_kwargs = {
                    "model": self.model,
                    "tools": tools if tools else None,
                }

                if self.temperature is not None:
                    completion_kwargs["temperature"] = self.temperature

                # Configure reasoning effort / thinking
                if self.thinking:
                    if self.model == "claude-opus-4-6":
                        completion_kwargs["thinking"] = {"type": "adaptive"}
                    else:
                        if self.reasoning_effort in [
                            "none",
                            "disable",
                            "low",
                            "medium",
                            "high",
                        ]:
                            completion_kwargs["reasoning_effort"] = (
                                self.reasoning_effort
                            )
                        else:
                            try:
                                thinking_budget = int(self.reasoning_effort)
                            except ValueError:
                                raise ValueError(
                                    "reasoning_effort must be 'none', 'disable', 'low', 'medium', 'high', or an integer value"
                                )
                            completion_kwargs["thinking"] = {
                                "type": "enabled",
                                "budget_tokens": thinking_budget,
                            }
                    if self.interleaved_thinking:
                        completion_kwargs["extra_headers"] = {
                            "anthropic-beta": "interleaved-thinking-2025-05-14"
                        }

                call_start_time = time.perf_counter()
                response = completion(
                    messages=model_messages_with_competition_prompt(messages),
                    **completion_kwargs,
                )

                # Accumulate turn metrics for this LLM call
                call_end_time = time.perf_counter()
                call_elapsed_ms = (call_end_time - call_start_time) * 1000.0

                if context.context_id not in self.ctx_id_to_turn_metrics:
                    self.ctx_id_to_turn_metrics[context.context_id] = {
                        PROMPT_TOKENS: 0,
                        COMPLETION_TOKENS: 0,
                        THINKING_TOKENS: 0,
                        COST: 0.0,
                        NUM_LLM_CALLS: 0,
                        "_total_llm_time_ms": 0.0,
                    }

                turn_m = self.ctx_id_to_turn_metrics[context.context_id]
                usage = getattr(response, "usage", None)
                if usage:
                    turn_m[PROMPT_TOKENS] += getattr(usage, "prompt_tokens", 0) or 0
                    turn_m[COMPLETION_TOKENS] += (
                        getattr(usage, "completion_tokens", 0) or 0
                    )
                    # Some providers report thinking/reasoning tokens in completion_tokens_details
                    details = getattr(usage, "completion_tokens_details", None)
                    if details:
                        turn_m[THINKING_TOKENS] += (
                            getattr(details, "reasoning_tokens", 0) or 0
                        )
                turn_m[COST] += (
                    getattr(response, "_hidden_params", {}).get("response_cost", 0.0)
                    or 0.0
                )
                turn_m[NUM_LLM_CALLS] += 1
                turn_m["_total_llm_time_ms"] += call_elapsed_ms

                # Get the message from LLM
                llm_message = response.choices[0].message
                assistant_content = llm_message.model_dump(exclude_unset=True)
                assistant_content = _validated_assistant_content(
                    assistant_content,
                    tool_index=ToolIndex(tools),
                )

                # Extract tool calls from assistant content
                tool_calls = assistant_content.get("tool_calls")

                ctx_logger.info(
                    "LLM response received",
                    has_tool_calls=bool(tool_calls),
                    num_tool_calls=len(tool_calls) if tool_calls else 0,
                    has_content=bool(assistant_content.get("content")),
                    content_length=len(assistant_content.get("content") or ""),
                    has_thinking=bool(
                        assistant_content.get("thinking_blocks")
                        or assistant_content.get("reasoning_content")
                    ),
                )
                ctx_logger.debug(
                    "LLM response details",
                    context_id=context.context_id[:8],
                    content=assistant_content.get("content"),
                    tool_calls=[
                        {
                            "name": tc["function"]["name"],
                            "args": tc["function"]["arguments"],
                        }
                        for tc in tool_calls
                    ]
                    if tool_calls
                    else None,
                    reasoning_content=assistant_content.get("reasoning_content"),
                )

                parts = _parts_from_assistant_content(assistant_content)

                ctx_logger.debug(
                    "Sending response",
                    context_id=context.context_id[:8],
                    num_parts=len(parts),
                )

            except Exception as e:
                logger.error(f"LLM error: {e}")
                # Error response as Parts
                parts = [new_text_part(f"Error processing request: {str(e)}")]
                # Create a simple assistant_content for error case
                assistant_content = {"content": f"Error processing request: {str(e)}"}

        # Add to history - preserve complete assistant message including thinking blocks
        # Store the full assistant_content to preserve thinking blocks and reasoning_content
        assistant_message_for_history = {
            "role": "assistant",
            "content": assistant_content.get("content"),
        }

        # Preserve tool calls in proper format for LLM API
        if assistant_content.get("tool_calls"):
            assistant_message_for_history["tool_calls"] = assistant_content[
                "tool_calls"
            ]

        messages.append(assistant_message_for_history)

        # Always return a Message — the agent under test is a conversational participant
        # in a multi-turn exchange. The evaluator decides when the task is done.
        response_message = new_message(
            parts=parts,
            context_id=context.context_id,
            role=Role.ROLE_AGENT,
        )

        # Attach turn_metrics on final response (no tool calls = turn complete)
        has_tool_calls = bool(assistant_content.get("tool_calls"))
        if not has_tool_calls and context.context_id in self.ctx_id_to_turn_metrics:
            turn_m = self.ctx_id_to_turn_metrics.pop(context.context_id)
            num_calls = turn_m[NUM_LLM_CALLS]
            avg_time = (
                (turn_m["_total_llm_time_ms"] / num_calls) if num_calls > 0 else 0.0
            )
            metrics_data = {
                PROMPT_TOKENS: turn_m[PROMPT_TOKENS],
                COMPLETION_TOKENS: turn_m[COMPLETION_TOKENS],
                COST: turn_m[COST],
                MODEL: self.model,
                THINKING_TOKENS: turn_m[THINKING_TOKENS],
                NUM_LLM_CALLS: num_calls,
                AVG_LLM_CALL_TIME_MS: round(avg_time, 1),
                NUM_PASSES: 1,
            }
            response_message.metadata.update({TURN_METRICS_KEY: metrics_data})
            ctx_logger.info(
                "Attached turn_metrics to final response",
                num_llm_calls=num_calls,
                avg_llm_call_time_ms=round(avg_time, 1),
                prompt_tokens=turn_m[PROMPT_TOKENS],
                completion_tokens=turn_m[COMPLETION_TOKENS],
            )

        await event_queue.enqueue_event(response_message)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel the current execution."""
        logger.bind(
            role="agent_under_test", context=f"ctx:{context.context_id[:8]}"
        ).info("Canceling context", context_id=context.context_id[:8])
        if context.context_id in self.ctx_id_to_messages:
            del self.ctx_id_to_messages[context.context_id]
        if context.context_id in self.ctx_id_to_tools:
            del self.ctx_id_to_tools[context.context_id]
        if context.context_id in self.ctx_id_to_turn_metrics:
            del self.ctx_id_to_turn_metrics[context.context_id]
        self.policy_controller.reset(context.context_id)
