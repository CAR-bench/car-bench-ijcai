"""
CAR-bench Agent - Agent under test that solves CAR-bench tasks.

This is the agent being tested. It:
1. Receives task descriptions with available tools from the evaluator
2. Decides which tool to call or how to respond
3. Returns responses in the expected JSON format wrapped in <json>...</json> tags
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
import sys
import diskcache
import uvicorn
from dotenv import load_dotenv

load_dotenv()

_EVPI_CACHE = diskcache.Cache(
    Path(__file__).parent.parent.parent / ".cache" / "evpi"
)

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.helpers.proto_helpers import new_message, new_text_part, new_data_part, new_task_from_user_message
from a2a.types import Role, TaskState
from google.protobuf.json_format import MessageToDict
from litellm import completion
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).parent.parent))
from logging_utils import configure_logger
from tool_call_types import ToolCall, ToolCallsData
from turn_metrics import TURN_METRICS_KEY, PROMPT_TOKENS, COMPLETION_TOKENS, COST, MODEL, THINKING_TOKENS, NUM_LLM_CALLS, AVG_LLM_CALL_TIME_MS, NUM_PASSES
sys.path.pop(0)

logger = configure_logger(role="agent_under_test", context="-")

SYSTEM_PROMPT = """You are a helpful car voice assistant. Follow the policy and tool instructions provided."""

# ── Prerequisite table: what must happen before calling each tool ─────────────
# Derived directly from wiki.md policies (AUT-POL / LLM-POL).
# auto=True  → inject the prerequisite call automatically (no user input needed)
# auto=False → must ask user; question_hint drives a natural clarification via LLM re-call
PREREQUISITE_TABLE: dict[str, list[dict]] = {
    # AUT-POL:005 + LLM-POL:008/AUT-POL:009
    "open_close_sunroof": [
        {
            "requires_tool": "get_weather",
            "auto": False,
            "prereq_args": {},
            "question_hint": (
                "The driver wants to open the sunroof but weather hasn't been checked yet. "
                "Sound like you're genuinely looking out for them — frame the question around "
                "current conditions outside, not around policy compliance."
            ),
        },
        {
            "requires_tool": "open_close_sunshade",
            "auto": True,
            "prereq_args": {"percentage": 100},
        },
    ],
    # LLM-POL:008 — fog lights need weather check + AUT-POL:013 — lights status check
    "set_fog_lights": [
        {
            "requires_tool": "get_exterior_lights_status",
            "auto": True,
            "prereq_args": {},
        },
        {
            "requires_tool": "get_weather",
            "auto": False,
            "prereq_args": {},
            "question_hint": (
                "Fog lights are being requested but conditions haven't been checked. "
                "Ask the driver naturally about current visibility or weather outside."
            ),
        },
    ],
    # AUT-POL:013/014 — must know current light state before changing beams
    "set_head_lights_high_beams": [
        {
            "requires_tool": "get_exterior_lights_status",
            "auto": True,
            "prereq_args": {},
        },
    ],
    "set_head_lights_low_beams": [
        {
            "requires_tool": "get_exterior_lights_status",
            "auto": True,
            "prereq_args": {},
        },
    ],
    # AUT-POL:010 — defrost requires knowing current climate state
    "set_window_defrost": [
        {
            "requires_tool": "get_vehicle_window_positions",
            "auto": True,
            "prereq_args": {},
        },
        {
            "requires_tool": "get_climate_settings",
            "auto": True,
            "prereq_args": {},
        },
    ],
    # AUT-POL:011 — AC requires knowing window positions and current climate
    "set_air_conditioning": [
        {
            "requires_tool": "get_vehicle_window_positions",
            "auto": True,
            "prereq_args": {},
        },
        {
            "requires_tool": "get_climate_settings",
            "auto": True,
            "prereq_args": {},
        },
    ],
    # AUT-POL:017/018 — nav edit tools only valid when navigation is active
    "navigation_replace_final_destination": [
        {
            "requires_tool": "get_current_navigation_state",
            "auto": True,
            "prereq_args": {"detailed_information": False},
        },
    ],
    "navigation_delete_final_destination": [
        {
            "requires_tool": "get_current_navigation_state",
            "auto": True,
            "prereq_args": {"detailed_information": False},
        },
    ],
    "navigation_add_one_waypoint": [
        {
            "requires_tool": "get_current_navigation_state",
            "auto": True,
            "prereq_args": {"detailed_information": False},
        },
    ],
    "navigation_replace_one_waypoint": [
        {
            "requires_tool": "get_current_navigation_state",
            "auto": True,
            "prereq_args": {"detailed_information": False},
        },
    ],
    "navigation_delete_one_waypoint": [
        {
            "requires_tool": "get_current_navigation_state",
            "auto": True,
            "prereq_args": {"detailed_information": False},
        },
    ],
    # Ambient lights — check current status and user preferences before changing color
    "set_ambient_lights": [
        {
            "requires_tool": "get_ambient_light_status_and_color",
            "auto": True,
            "prereq_args": {},
        },
        {
            "requires_tool": "get_user_preferences",
            "auto": True,
            "prereq_args": {"preference_categories": {"vehicle_settings": {"vehicle_settings": True}}},
        },
    ],
    # EV calculations need charging specs first
    "calculate_charging_time_by_soc": [
        {
            "requires_tool": "get_charging_specs_and_status",
            "auto": True,
            "prereq_args": {},
        },
    ],
    "get_distance_by_soc": [
        {
            "requires_tool": "get_charging_specs_and_status",
            "auto": True,
            "prereq_args": {},
        },
    ],
}

# ── Expected response fields per tool (source: get_output_info() in each tool) ─
REQUIRED_RESPONSE_FIELDS: dict[str, set[str]] = {
    "get_exterior_lights_status": {
        "fog_lights", "head_lights_low_beams", "head_lights_high_beams",
    },
    "get_vehicle_window_positions": {
        "window_driver_position", "window_passenger_position",
        "window_driver_rear_position", "window_passenger_rear_position",
    },
    "get_climate_settings": {
        "fan_speed", "fan_airflow_direction", "air_conditioning",
        "air_circulation", "window_front_defrost",
    },
    "get_temperature_inside_car": {
        "climate_temperature_driver", "climate_temperature_passenger",
    },
    "get_sunroof_and_sunshade_position": {"sunroof_position", "sunshade_position"},
    "get_seat_heating_level": {"seat_heating_driver", "seat_heating_passenger"},
    "get_steering_wheel_heating_level": {"steering_wheel_heating"},
    "get_ambient_light_status_and_color": {"ambient_light"},
    "get_trunk_door_position": {"trunk_door_position"},
    "get_seats_occupancy": {"seats_occupied"},
}

# ── Preference prerequisite table: tools that need user prefs fetched first ────
# Maps set_* tool → preference_categories arg for get_user_preferences.
# Agent auto-injects get_user_preferences before these tools if not yet fetched.
PREFERENCE_PREREQUISITE_TABLE: dict[str, dict] = {
    "set_ambient_lights":            {"vehicle_settings": {"vehicle_settings": True}},
    "set_climate_temperature":       {"vehicle_settings": {"climate_control": True}},
    "set_fan_speed":                 {"vehicle_settings": {"climate_control": True}},
    "set_fan_airflow_direction":     {"vehicle_settings": {"climate_control": True}},
    "set_air_conditioning":          {"vehicle_settings": {"climate_control": True}},
    "set_air_circulation":           {"vehicle_settings": {"climate_control": True}},
    "set_seat_heating":              {"vehicle_settings": {"vehicle_settings": True}},
    "set_steering_wheel_heating":    {"vehicle_settings": {"vehicle_settings": True}},
    "set_window_defrost":            {"vehicle_settings": {"climate_control": True}},
    "set_new_navigation":            {"navigation_and_routing": {"route_selection": True}},
    "navigation_replace_final_destination": {"navigation_and_routing": {"route_selection": True}},
    "navigation_add_one_waypoint":   {"navigation_and_routing": {"route_selection": True}},
}

# ── Info-state requirements: fields from prior calls needed before each action ─
# key = "source_tool.field_name"; value 0 in info_state means the field was missing
REQUIRED_FIELDS_FOR_ACTION: dict[str, set[str]] = {
    "set_fog_lights": {
        "get_exterior_lights_status.fog_lights",
    },
    "set_head_lights_high_beams": {
        "get_exterior_lights_status.head_lights_high_beams",
    },
    "set_head_lights_low_beams": {
        "get_exterior_lights_status.head_lights_low_beams",
    },
    "set_window_defrost": {
        "get_vehicle_window_positions.window_driver_position",
        "get_vehicle_window_positions.window_passenger_position",
    },
}


class CARBenchAgentExecutor(AgentExecutor):
    """Executor for the CAR-bench agent under test using native tool calling."""

    def __init__(self, model: str, temperature: float = 0.0, thinking: bool = False, reasoning_effort: str = "medium", interleaved_thinking: bool = False):
        self.model = model
        self.temperature = temperature
        self.thinking = thinking
        self.reasoning_effort = reasoning_effort  # Can be 'none', 'disable', 'low', 'medium', 'high', or integer token budget
        self.interleaved_thinking = interleaved_thinking  # Whether to use interleaved thinking
        self.ctx_id_to_messages: dict[str, list[dict]] = {}
        self.ctx_id_to_tools: dict[str, list[dict]] = {}
        # Per-context turn metrics accumulation (reset when final response is sent)
        self.ctx_id_to_turn_metrics: dict[str, dict] = {}
        # Policy layer state (Points 1-5)
        self.ctx_id_to_tool_history: dict[str, dict] = {}    # tool_name → raw result content
        self.ctx_id_to_pending_intent: dict[str, dict] = {}  # deferred call waiting on prereq
        self.ctx_id_to_info_state: dict[str, dict] = {}      # "tool.field" → 0|1 binary vector

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        inbound_message = context.message
        ctx_logger = logger.bind(role="agent_under_test", context=f"ctx:{context.context_id[:8]}")

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
                            messages.append({"role": "system", "content": system_prompt})
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
                        # Update tool history and info state vector
                        for tr in incoming_tool_results:
                            tr_name = tr.get("tool_name", "") if isinstance(tr, dict) else tr.get("toolName", "")
                            if tr_name:
                                self._update_info_state(context.context_id, tr_name, tr.get("content", ""))

            # Fallback if no text part and no structured tool results found
            if not user_message_text and not incoming_tool_results:
                user_message_text = context.get_user_input()

            ctx_logger.info(
                "Received user message",
                context_id=context.context_id[:8],
                turn=len(messages) + 1,
                message_preview=(user_message_text[:100] if user_message_text else
                                 f"[{len(incoming_tool_results)} tool results]" if incoming_tool_results else "")
            )
            ctx_logger.debug(
                "Message details",
                context_id=context.context_id[:8],
                message=user_message_text,
                num_parts=len(inbound_message.parts),
                has_tools=bool(tools),
                num_tools=len(tools) if tools else 0,
                has_tool_results=bool(incoming_tool_results),
                num_tool_results=len(incoming_tool_results) if incoming_tool_results else 0
            )

        except Exception as e:
            logger.warning(f"Failed to parse message parts: {e}, using fallback")
            user_message_text = context.get_user_input()

        # Check if previous message had tool calls - if so, format as tool results
        if messages and messages[-1].get("role") == "assistant" and messages[-1].get("tool_calls"):
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
                    tr_name = tr.get("tool_name", "") if isinstance(tr, dict) else tr.get("toolName", "")
                    matching_calls = tool_call_by_name.get(tr_name, [])
                    if matching_calls:
                        # Pop the first matching call to handle duplicate tool names
                        matched_tc = matching_calls.pop(0)
                        tool_results.append({
                            "role": "tool",
                            "tool_call_id": matched_tc["id"],
                            "content": tr.get("content", ""),
                        })
                    else:
                        # Fallback: no matching tool_call found, use first unmatched
                        ctx_logger.warning(
                            "No matching tool_call_id for tool result",
                            tool_name=tr_name,
                        )
                        tool_results.append({
                            "role": "tool",
                            "tool_call_id": tr.get("tool_call_id", tr.get("toolCallId", f"unknown_{tr_name}")),
                            "content": tr.get("content", ""),
                        })
            else:
                # Fallback: no structured tool results, use the text message
                # for all tool calls (legacy behavior)
                tool_results = []
                for tc in prev_tool_calls:
                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": user_message_text or "",
                    })

            # Add all tool result messages
            messages.extend(tool_results)

            ctx_logger.debug(
                "Formatted tool results",
                num_tools=len(tool_results),
                tool_call_ids=[tr["tool_call_id"] for tr in tool_results]
            )
        else:
            # Regular user message
            messages.append({"role": "user", "content": user_message_text})

        # ── Pending intent: if we auto-injected a prereq last turn, fire the original ──
        pending = self.ctx_id_to_pending_intent.get(context.context_id)
        if pending and incoming_tool_results:
            tool_history = self.ctx_id_to_tool_history.get(context.context_id, {})
            if pending.get("prereq_tool") in tool_history:
                del self.ctx_id_to_pending_intent[context.context_id]
                ctx_logger.info("Policy: firing pending intent", tool=pending["tool_name"])
                pending_tc = {
                    "id": str(uuid4()),
                    "type": "function",
                    "function": {
                        "name": pending["tool_name"],
                        "arguments": json.dumps(pending["args"]),
                    },
                }
                pending_tool_calls_list = [
                    ToolCall(tool_name=pending["tool_name"], arguments=pending["args"])
                ]
                pending_parts = [new_data_part(ToolCallsData(tool_calls=pending_tool_calls_list).model_dump())]
                assistant_msg = {"role": "assistant", "content": None, "tool_calls": [pending_tc]}
                messages.append(assistant_msg)
                response_message = new_message(
                    parts=pending_parts,
                    context_id=context.context_id,
                    role=Role.ROLE_AGENT,
                )
                await event_queue.enqueue_event(response_message)
                return

        # Call LLM with native tool calling
        try:
            is_anthropic = "claude" in self.model.lower() or "anthropic" in self.model.lower()

            # Prompt caching is Anthropic-only — skip for Ollama/Groq/Gemini etc.
            if is_anthropic:
                if tools:
                    tools[-1]["function"]["cache_control"] = {"type": "ephemeral"}
                if messages:
                    messages[0]["cache_control"] = {"type": "ephemeral"}

            completion_kwargs = {
                "model": self.model,
                "tools": tools if tools else None
            }

            # Ollama needs a larger context window to fit 57 tool schemas (~10K tokens)
            if self.model.startswith("ollama/"):
                completion_kwargs["num_ctx"] = 16384

            if self.temperature is not None:
                completion_kwargs["temperature"] = self.temperature

            # Configure reasoning effort / thinking
            if self.thinking:
                    if self.model == "claude-opus-4-6":
                        completion_kwargs["thinking"] = {
                            "type": "adaptive"
                        }
                    else:
                        if self.reasoning_effort in [
                            "none",
                            "disable",
                            "low",
                            "medium",
                            "high",
                        ]:
                            completion_kwargs["reasoning_effort"] = self.reasoning_effort
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
                messages=messages,
                **completion_kwargs
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
                turn_m[COMPLETION_TOKENS] += getattr(usage, "completion_tokens", 0) or 0
                # Some providers report thinking/reasoning tokens in completion_tokens_details
                details = getattr(usage, "completion_tokens_details", None)
                if details:
                    turn_m[THINKING_TOKENS] += getattr(details, "reasoning_tokens", 0) or 0
            turn_m[COST] += getattr(response, "_hidden_params", {}).get("response_cost", 0.0) or 0.0
            turn_m[NUM_LLM_CALLS] += 1
            turn_m["_total_llm_time_ms"] += call_elapsed_ms

            # Get the message from LLM
            llm_message = response.choices[0].message
            assistant_content = llm_message.model_dump(exclude_unset=True)

            # Extract tool calls from assistant content
            tool_calls = assistant_content.get("tool_calls")

            # ── Policy layer: intercept and validate tool calls ───────────
            if tool_calls and tools:
                policy_result = self._apply_policy_layer(
                    ctx_id=context.context_id,
                    tool_calls=tool_calls,
                    tools=tools,
                    messages=messages,
                    completion_kwargs=completion_kwargs,
                    ctx_logger=ctx_logger,
                )
                if policy_result is not None:
                    if policy_result["type"] == "replace_with_text":
                        pol_resp = policy_result["response"]
                        pol_msg = pol_resp.choices[0].message
                        assistant_content = pol_msg.model_dump(exclude_unset=True)
                        tool_calls = None
                    elif policy_result["type"] == "replace_with_calls":
                        tool_calls = policy_result["tool_calls"]
                        assistant_content = {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": tool_calls,
                        }

            ctx_logger.info(
                "LLM response received",
                has_tool_calls=bool(tool_calls),
                num_tool_calls=len(tool_calls) if tool_calls else 0,
                has_content=bool(assistant_content.get("content")),
                content_length=len(assistant_content.get("content") or ""),
                has_thinking=bool(assistant_content.get("thinking_blocks") or assistant_content.get("reasoning_content"))
            )
            ctx_logger.debug(
                "LLM response details",
                context_id=context.context_id[:8],
                content=assistant_content.get("content"),
                tool_calls=[{"name": tc["function"]["name"], "args": tc["function"]["arguments"]} for tc in tool_calls] if tool_calls else None,
                reasoning_content=assistant_content.get("reasoning_content")
            )

            # Build proper A2A Message with Parts (protobuf)
            parts = []

            # Add text Part if there's content
            if assistant_content.get("content"):
                parts.append(new_text_part(assistant_content["content"]))

            # Add data Part if there are tool calls
            if assistant_content.get("tool_calls"):
                tool_calls_list = [
                    ToolCall(
                        tool_name=tc["function"]["name"],
                        arguments=json.loads(tc["function"]["arguments"]),
                    )
                    for tc in assistant_content["tool_calls"]
                ]
                tool_calls_data = ToolCallsData(tool_calls=tool_calls_list)
                parts.append(new_data_part(tool_calls_data.model_dump()))

            # Add reasoning_content as data Part for debugging (if present)
            if assistant_content.get("reasoning_content"):
                parts.append(new_data_part({"reasoning_content": assistant_content["reasoning_content"]}))

            # If no parts, add empty text
            if not parts:
                parts.append(new_text_part(assistant_content.get("content", "")))

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
            assistant_message_for_history["tool_calls"] = assistant_content["tool_calls"]

        # Preserve thinking blocks and reasoning content for Claude extended thinking
        if assistant_content.get("thinking_blocks"):
            assistant_message_for_history["thinking_blocks"] = assistant_content["thinking_blocks"]
        if assistant_content.get("reasoning_content"):
            assistant_message_for_history["reasoning_content"] = assistant_content["reasoning_content"]

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
            avg_time = (turn_m["_total_llm_time_ms"] / num_calls) if num_calls > 0 else 0.0
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
        logger.bind(role="agent_under_test", context=f"ctx:{context.context_id[:8]}").info(
            "Canceling context",
            context_id=context.context_id[:8]
        )
        if context.context_id in self.ctx_id_to_messages:
            del self.ctx_id_to_messages[context.context_id]
        if context.context_id in self.ctx_id_to_tools:
            del self.ctx_id_to_tools[context.context_id]
        if context.context_id in self.ctx_id_to_turn_metrics:
            del self.ctx_id_to_turn_metrics[context.context_id]
        if context.context_id in self.ctx_id_to_tool_history:
            del self.ctx_id_to_tool_history[context.context_id]
        if context.context_id in self.ctx_id_to_pending_intent:
            del self.ctx_id_to_pending_intent[context.context_id]
        if context.context_id in self.ctx_id_to_info_state:
            del self.ctx_id_to_info_state[context.context_id]

    # ── Policy layer helper methods ───────────────────────────────────────────

    def _update_info_state(self, ctx_id: str, tool_name: str, result_content: str) -> None:
        """Parse a tool result and record which expected fields are present (1) or missing (0)."""
        if ctx_id not in self.ctx_id_to_tool_history:
            self.ctx_id_to_tool_history[ctx_id] = {}
        self.ctx_id_to_tool_history[ctx_id][tool_name] = result_content

        expected = REQUIRED_RESPONSE_FIELDS.get(tool_name)
        if not expected:
            return

        if ctx_id not in self.ctx_id_to_info_state:
            self.ctx_id_to_info_state[ctx_id] = {}
        state = self.ctx_id_to_info_state[ctx_id]

        try:
            parsed = json.loads(result_content) if isinstance(result_content, str) else result_content
            result_dict = parsed.get("result", parsed) if isinstance(parsed, dict) else {}
            for field in expected:
                state[f"{tool_name}.{field}"] = 1 if field in result_dict else 0
        except (json.JSONDecodeError, AttributeError):
            for field in expected:
                state[f"{tool_name}.{field}"] = 0

    def _check_missing_response_fields(self, ctx_id: str, action_name: str) -> set[str]:
        """Return field keys required for action_name that are flagged absent (0) in info_state."""
        required = REQUIRED_FIELDS_FOR_ACTION.get(action_name, set())
        if not required:
            return set()
        state = self.ctx_id_to_info_state.get(ctx_id, {})
        # Default to 1 (assume present) if field was never tracked — only flag explicit 0s
        return {f for f in required if state.get(f, 1) == 0}

    def _generate_clarification(self, messages: list[dict], hint: str, completion_kwargs: dict):
        """Re-call LLM with a temporary hint to generate a natural clarification response."""
        temp_messages = messages + [{"role": "user", "content": f"[AGENT CONTEXT: {hint}]"}]
        kwargs = {k: v for k, v in completion_kwargs.items() if k != "tools"}
        return completion(messages=temp_messages, **kwargs)

    # Keywords that indicate the user named a specific car system → not tool-level ambiguous
    _SPECIFIC_CAR_TERMS = {
        "temperature", "temp", "fan", "speed", "ac", "air conditioning", "heat", "cool",
        "defrost", "airflow", "circulation", "climate",
        "light", "lights", "beam", "beams", "fog", "ambient", "reading",
        "window", "sunroof", "sunshade", "trunk",
        "navigation", "route", "destination", "waypoint", "navigate",
        "seat", "steering", "wheel",
        "call", "phone", "contact", "email", "send",
        "battery", "charge", "range", "distance",
        "weather",
    }

    def _get_missing_required_params(self, tool_name: str, tc_args: dict, tools: list[dict]) -> list[str]:
        """Return required params from the tool schema that are absent or None in tc_args."""
        tool_schema = next((t for t in tools if t["function"]["name"] == tool_name), None)
        if not tool_schema:
            return []
        required = tool_schema["function"].get("parameters", {}).get("required", [])
        return [p for p in required if p not in tc_args or tc_args[p] is None]

    def _is_vague_request(self, user_msg: str) -> bool:
        """True if message is short and doesn't name any specific car system."""
        if len(user_msg.split()) >= 15:
            return False
        msg_lower = user_msg.lower()
        return not any(term in msg_lower for term in self._SPECIFIC_CAR_TERMS)

    def _score_question_eig(self, question: str, interpretations: list[str], completion_kwargs: dict) -> int:
        """Score a question by how evenly it splits interpretations (lower = better split)."""
        prompt = (
            f"For each interpretation, would the answer to this question be YES or NO?\n"
            f"Question: {question}\n"
            f"Interpretations:\n"
            + "\n".join(f"{i+1}. {interp}" for i, interp in enumerate(interpretations))
            + "\nReply with only a JSON array of YES or NO strings, one per interpretation in order."
        )
        kwargs = {k: v for k, v in completion_kwargs.items() if k != "tools"}
        kwargs["temperature"] = 0.0
        try:
            resp = completion(messages=[{"role": "user", "content": prompt}], **kwargs)
            answers = json.loads(resp.choices[0].message.content.strip())
            yes_count = sum(1 for a in answers if str(a).upper().startswith("Y"))
            no_count = len(answers) - yes_count
            return abs(yes_count - no_count)
        except Exception:
            return len(interpretations)

    def _generate_evpi_clarification(self, user_request: str, messages: list[dict], completion_kwargs: dict):
        """Generate the most informative clarifying question when tool-level ambiguity is detected."""
        cache_key = "evpi:" + hashlib.md5(user_request.strip().lower().encode()).hexdigest()
        if cache_key in _EVPI_CACHE:
            cached_question = _EVPI_CACHE[cache_key]
            phrase_messages = messages + [{
                "role": "user",
                "content": f"[AGENT CONTEXT: Ask the driver this one clarifying question naturally and briefly: {cached_question}]"
            }]
            kwargs = {k: v for k, v in completion_kwargs.items() if k != "tools"}
            kwargs["temperature"] = 0.0
            return completion(messages=phrase_messages, **kwargs)

        kwargs = {k: v for k, v in completion_kwargs.items() if k != "tools"}

        # Step 1: generate diverse interpretations referencing different car systems
        kwargs["temperature"] = 0.7
        interp_prompt = (
            f'The car driver said: "{user_request}"\n'
            "List 3-4 distinct possible interpretations of what they might want. "
            "Each must reference a specific car system (AC, seat heating, windows, navigation, etc.).\n"
            "Reply with only a JSON array of strings."
        )
        try:
            interp_resp = completion(messages=[{"role": "user", "content": interp_prompt}], **kwargs)
            interpretations = json.loads(interp_resp.choices[0].message.content.strip())
            if not isinstance(interpretations, list) or len(interpretations) < 2:
                return None
        except Exception:
            return None

        # Step 2: generate candidate yes/no clarifying questions
        kwargs["temperature"] = 0.7
        q_prompt = (
            f'The car driver said: "{user_request}"\n'
            f"Possible interpretations: {json.dumps(interpretations)}\n"
            "Generate 3 short yes/no clarifying questions to identify which interpretation is correct. "
            "Questions must be natural, brief, and safe for a driver to answer.\n"
            "Reply with only a JSON array of 3 question strings."
        )
        try:
            q_resp = completion(messages=[{"role": "user", "content": q_prompt}], **kwargs)
            candidates = json.loads(q_resp.choices[0].message.content.strip())
            if not isinstance(candidates, list) or len(candidates) == 0:
                return None
        except Exception:
            return None

        # Step 3: pick question with best info gain (most balanced YES/NO split)
        kwargs["temperature"] = 0.0
        scored = [(self._score_question_eig(q, interpretations, kwargs), q) for q in candidates]
        best_question = min(scored, key=lambda x: x[0])[1]

        # Step 4: phrase it naturally in the agent's voice
        kwargs["temperature"] = 0.0
        phrase_messages = messages + [{
            "role": "user",
            "content": f"[AGENT CONTEXT: Ask the driver this one clarifying question naturally and briefly: {best_question}]"
        }]
        _EVPI_CACHE.set(cache_key, best_question, expire=86400)  # 24h TTL
        return completion(messages=phrase_messages, **kwargs)

    def _apply_policy_layer(
        self,
        ctx_id: str,
        tool_calls: list[dict],
        tools: list[dict],
        messages: list[dict],
        completion_kwargs: dict,
        ctx_logger,
    ) -> dict | None:
        """
        Validate proposed tool calls against policy rules.
        Returns action dict or None if all checks pass.

        Return shapes:
          {"type": "replace_with_text",  "response": <litellm response>}
          {"type": "replace_with_calls", "tool_calls": [...]}
        """
        available_tool_names = {t["function"]["name"] for t in tools}
        tool_history = self.ctx_id_to_tool_history.get(ctx_id, {})

        for tc in tool_calls:
            tc_name = tc["function"]["name"]
            try:
                tc_args = json.loads(tc["function"]["arguments"]) if isinstance(tc["function"]["arguments"], str) else (tc["function"]["arguments"] or {})
            except (json.JSONDecodeError, TypeError):
                tc_args = {}

            # ── Check A: Tool existence (E5b fix) ────────────────────────
            if tc_name not in available_tool_names:
                ctx_logger.info("Policy[A]: tool not available", tool=tc_name)
                hint = (
                    f"The driver asked for something that requires '{tc_name}', "
                    "but this capability isn't available in their car. "
                    "Let them know honestly and briefly — warm but direct, no over-apologizing."
                )
                return {"type": "replace_with_text", "response": self._generate_clarification(messages, hint, completion_kwargs)}

            # ── Check B: Prerequisites (E1 fix) ──────────────────────────
            for prereq in PREREQUISITE_TABLE.get(tc_name, []):
                prereq_tool = prereq["requires_tool"]
                if prereq_tool not in tool_history:
                    if prereq["auto"]:
                        ctx_logger.info("Policy[B]: auto-injecting prereq", prereq=prereq_tool, for_tool=tc_name)
                        self.ctx_id_to_pending_intent[ctx_id] = {
                            "tool_name": tc_name,
                            "args": tc_args,
                            "prereq_tool": prereq_tool,
                        }
                        prereq_tc = {
                            "id": str(uuid4()),
                            "type": "function",
                            "function": {
                                "name": prereq_tool,
                                "arguments": json.dumps(prereq.get("prereq_args", {})),
                            },
                        }
                        return {"type": "replace_with_calls", "tool_calls": [prereq_tc]}
                    else:
                        ctx_logger.info("Policy[B]: need user info", prereq=prereq_tool, for_tool=tc_name)
                        return {"type": "replace_with_text", "response": self._generate_clarification(messages, prereq["question_hint"], completion_kwargs)}

            # ── Check C: Missing response fields (hallucination_missing_tool_response fix) ──
            missing = self._check_missing_response_fields(ctx_id, tc_name)
            if missing:
                ctx_logger.info("Policy[C]: missing response fields", tool=tc_name, missing=list(missing))
                field_labels = [f.split(".", 1)[-1] for f in missing]
                hint = (
                    f"You need to use '{tc_name}' but a required piece of information is missing "
                    f"from an earlier tool response ({', '.join(field_labels)}). "
                    "Tell the driver honestly that you couldn't get all the info needed. Be brief."
                )
                return {"type": "replace_with_text", "response": self._generate_clarification(messages, hint, completion_kwargs)}

            # ── Check E: Preference auto-injection ────────────────────────────────
            # If agent is about to call a preference-sensitive tool but hasn't
            # fetched user preferences yet, inject get_user_preferences first.
            if tc_name in PREFERENCE_PREREQUISITE_TABLE and "get_user_preferences" not in tool_history:
                pref_categories = PREFERENCE_PREREQUISITE_TABLE[tc_name]
                ctx_logger.info("Policy[E]: auto-injecting get_user_preferences", for_tool=tc_name)
                self.ctx_id_to_pending_intent[ctx_id] = {
                    "tool_name": tc_name,
                    "args": tc_args,
                    "prereq_tool": "get_user_preferences",
                }
                pref_tc = {
                    "id": str(uuid4()),
                    "type": "function",
                    "function": {
                        "name": "get_user_preferences",
                        "arguments": json.dumps({"preference_categories": pref_categories}),
                    },
                }
                return {"type": "replace_with_calls", "tool_calls": [pref_tc]}

        # ── Check D: Disambiguation ───────────────────────────────────────────
        # Two-stage check:
        #   D1 — Params missing → ask one targeted question about the missing param(s).
        #   D2 — All params present but message is vague (short + no specific car system named)
        #        → run EVPI to detect tool-level ambiguity and ask the best question.
        if len(tool_calls) == 1:
            tc = tool_calls[0]
            tc_name_d = tc["function"]["name"]
            try:
                tc_args_d = json.loads(tc["function"]["arguments"]) if isinstance(tc["function"]["arguments"], str) else (tc["function"]["arguments"] or {})
            except (json.JSONDecodeError, TypeError):
                tc_args_d = {}

            missing_params = self._get_missing_required_params(tc_name_d, tc_args_d, tools)

            if missing_params:
                # D1: required param(s) genuinely missing — ask targeted question
                ctx_logger.info("Policy[D1]: missing required params", tool=tc_name_d, missing=missing_params)
                hint = (
                    f"You are about to call '{tc_name_d}' but the following required information "
                    f"is missing: {', '.join(missing_params)}. "
                    "Ask the driver for exactly this information — one short, natural question."
                )
                return {"type": "replace_with_text", "response": self._generate_clarification(messages, hint, completion_kwargs)}

            else:
                # D2: all params filled — check for tool-level ambiguity on vague messages
                user_msgs = [m for m in messages if m.get("role") == "user"]
                last_user_msg = user_msgs[-1].get("content", "") if user_msgs else ""
                # Skip D2 if this message is answering a question the agent just asked
                asst_msgs = [m for m in messages if m.get("role") == "assistant" and m.get("content")]
                last_asst_content = asst_msgs[-1].get("content", "") if asst_msgs else ""
                is_followup_answer = bool(last_asst_content and last_asst_content.strip().endswith("?"))
                if last_user_msg and not is_followup_answer and self._is_vague_request(last_user_msg):
                    ctx_logger.info("Policy[D2]: vague request, running EVPI", tool=tc_name_d, msg=last_user_msg[:60])
                    evpi_resp = self._generate_evpi_clarification(last_user_msg, messages, completion_kwargs)
                    if evpi_resp is not None:
                        return {"type": "replace_with_text", "response": evpi_resp}

        return None  # all checks passed
