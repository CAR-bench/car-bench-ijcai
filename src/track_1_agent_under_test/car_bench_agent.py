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
import re
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
from pydantic import BaseModel
from uuid import uuid4


# ── Structured-output schemas for judge/verifier LLM calls ────────────────────
# Passed as `response_format=<Model>` to litellm's completion() — the same
# pattern the organizer's own policy_evaluator.py uses. This constrains the
# provider's output at generation time instead of hoping a "reply with only
# JSON" instruction is followed, which is what caused the fence-wrapped/
# malformed JSON that made Check O/P's judge calls fail intermittently.
class GenericPrereqJudgment(BaseModel):
    missing_prereq: bool
    hint: str = ""


class PolicyCheckResult(BaseModel):
    id: str
    followed: bool
    reason: str = ""


class PolicyVerifierResult(BaseModel):
    results: list[PolicyCheckResult]


class EVPICandidate(BaseModel):
    question: str
    answers: list[str]


class EVPIResponse(BaseModel):
    interpretations: list[str]
    candidates: list[EVPICandidate]

sys.path.insert(0, str(Path(__file__).parent.parent))
from logging_utils import configure_logger
from tool_call_types import ToolCall, ToolCallsData
from turn_metrics import TURN_METRICS_KEY, PROMPT_TOKENS, COMPLETION_TOKENS, COST, MODEL, THINKING_TOKENS, NUM_LLM_CALLS, AVG_LLM_CALL_TIME_MS, NUM_PASSES
sys.path.pop(0)

logger = configure_logger(role="agent_under_test", context="-")

SYSTEM_PROMPT = """You are a helpful car voice assistant. Follow the policy and tool instructions provided."""

# Appended to the evaluator's own wiki system prompt. Re-emphasizes rules that
# are already stated in the wiki (LLM-POL:002, LLM-POL:022) but were observed
# to be missed in practice — this is salience reinforcement, not new policy.
FORMAT_REMINDER = """

REMINDERS (already required by the policy above, called out because they are
easy to miss):
- ALWAYS state times in 24-hour format (e.g. "14:30", never "2:30 PM"). This
  applies even when a tool result gives you a 24h time — do not convert it to
  12h in your response.
- ALWAYS state temperatures with the explicit unit "degrees Celsius" (or
  "°C"), and distances in kilometers — never state a bare number.
- When you proactively pick the fastest route for a multi-stop request because
  the user didn't specify a route preference, you MUST say in your response
  that you chose the fastest option, and ask if they want details on
  alternative routes.
- Before acting, check whether the request actually matches a real capability
  of this car and your available tools. If the user asks for a tool,
  argument, value, or capability that does not exist among what you were
  given (e.g. a setting, sensor, or action with no matching tool, or a tool
  that has been removed), do NOT invent a workaround, guess at an alternative
  tool, or claim you performed the action anyway. Say plainly that it isn't
  something you can do, and stop there — do not pretend it succeeded.

  Example: User: "Can you close the sunshade halfway, to 50%?" (no sunshade
  tool is available this turn, only open_close_sunroof). Don't do this: call
  open_close_sunroof instead, since it sounds similar. Do this: "I don't have
  a way to control the sunshade right now — only the sunroof. Want me to
  adjust the sunroof instead?"

  A tool call can also succeed while a specific piece of data inside its
  result comes back unknown or missing. Never state a specific value, or
  claim a change was confirmed, for a field you never actually received.

  Example: User: "Are my rear windows fully open?" (get_vehicle_window_positions
  ran, but its result came back unknown for the rear window fields specifically).
  Don't do this: state a percentage or confirm the rear windows are at any
  position, since that data was never actually returned. Do this: "I got some
  window info back, but the rear window positions came back unknown right
  now — I can't confirm their exact state."

  When a needed value is missing this way, do not ask the user to manually
  supply it so you can proceed anyway — that is routing around the missing
  capability, not acknowledging it. State plainly that you can't check it.

  Example: User: "How long will it take to charge to full?" (get_charging_specs_and_status
  ran, but battery_capacity_kwh and state_of_charge both came back unknown).
  Don't do this: ask the user to tell you their current charge level so you
  can calculate it anyway. Do this: "I'm not able to check your battery's
  current charge or capacity right now, so I can't calculate charging time."

  Example: User: "Set the ambient lighting to a warm orange color." (the
  set_ambient_lights tool only accepts RED, BLUE, GREEN, CYAN, or OFF). Don't
  do this: call set_ambient_lights with color=ORANGE anyway, or silently
  substitute RED. Do this: "Orange isn't one of the ambient light colors I
  can set — the options are red, blue, green, or cyan. Want one of those?\""""

# ── Retry/timeout bounds for every LLM call in this agent ─────────────────────
# Without this, a single transient provider error (e.g. Gemini 503 "high
# demand") can hang a task for a very long time under litellm's own default
# retry/backoff. Configurable via env vars so dev vs. hidden-set runs can tune
# without a code change.
LLM_NUM_RETRIES = int(os.getenv("AGENT_LLM_NUM_RETRIES", "3"))
LLM_TIMEOUT_SECONDS = float(os.getenv("AGENT_LLM_TIMEOUT_SECONDS", "30"))

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

# ── Window enum → get_vehicle_window_positions field (for AC side-effect check) ─
WINDOW_POSITION_FIELDS: dict[str, str] = {
    "DRIVER": "window_driver_position",
    "PASSENGER": "window_passenger_position",
    "DRIVER_REAR": "window_driver_rear_position",
    "PASSENGER_REAR": "window_passenger_rear_position",
}

# ── Nav-editing tools that must never be called in parallel (TECH-AUT-POL:018) ─
NAVIGATION_EDIT_TOOLS: set[str] = {
    "navigation_add_one_waypoint",
    "navigation_delete_final_destination",
    "navigation_delete_one_waypoint",
    "navigation_replace_final_destination",
    "navigation_replace_one_waypoint",
}

# ── Tools requiring explicit user confirmation before calling (LLM-POL:004) ────
CONFIRMATION_REQUIRED_TOOLS: set[str] = {
    "send_email",
    "open_close_trunk_door",
    "set_head_lights_high_beams",
}

# Short affirmative replies that count as explicit confirmation for the above.
_AFFIRMATIVE_PATTERNS: set[str] = {
    "yes", "yeah", "yep", "yup", "sure", "confirmed", "confirm",
    "go ahead", "please do", "do it", "correct", "affirmative", "ok", "okay",
}

# Name prefixes treated as state-changing (candidates for Check O's generic
# prereq judge when not already covered by PREREQUISITE_TABLE).
STATE_CHANGING_PREFIXES = ("set_", "navigation_", "call_", "send_", "open_close_")

# ── Policy-compliance verifier rules (semantic-only wiki policies that can't ──
# be reduced to a deterministic table — Check P). Trigger tools mirror the
# organizer's own evaluator gating (enhance_policy_line_with_context in
# reward_calculators.py) so this only fires on the subset of tasks that
# actually touch these domains, not every turn. Policy text is the exact
# wording from wiki.md.
POLICY_VERIFIER_RULES: list[dict] = [
    {
        "id": "LLM-POL:007",
        "trigger_tools": {"open_close_window"},
        "policy_text": (
            "If windows are requested by the user to open more than 25% (absolute position) "
            "and AC is ON in that moment, prompt for confirmation and warn about energy inefficiency."
        ),
    },
    {
        "id": "LLM-POL:012",
        "trigger_tools": {"set_climate_temperature"},
        "policy_text": (
            "If the user sets the temperature to a single seat zone and the resulting temperature "
            "difference after execution to the other seat zones is more than 3 degrees Celsius, "
            "then the user must be informed about it."
        ),
    },
    {
        "id": "LLM-POL:021",
        "trigger_tools": {"get_routes_from_start_to_destination"},
        "policy_text": (
            "If a route is presented in detail (fastest route, shortest route, or upon user detail "
            "request), and the route includes toll roads, then the user must be informed about it."
        ),
    },
    {
        "id": "LLM-POL:022",
        "trigger_tools": {"get_routes_from_start_to_destination", "set_new_navigation"},
        "policy_text": (
            "If the user asks for a multi-stop route and does not specify the route selection, then "
            "take the fastest route proactively per route segment. Inform the user that you took the "
            "fastest alternative and ask if he wants more information on alternative routes. Still "
            "inform the user if one route segment of the chosen routes includes a toll road."
        ),
    },
]

# Matches "2 PM", "1:30pm", "11:00 A.M." etc. for deterministic 24h conversion.
_TIME_12H_PATTERN = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*([AaPp])\.?[Mm]\.?\b")

# JSON Schema keywords not supported by Vertex's Gemini function-calling
# Schema proto -- must be stripped before sending tools to a fine-tuned
# endpoint, since LiteLLM's normal OpenAI->Gemini conversion (which already
# strips these) never runs for a bare numeric model id.
_UNSUPPORTED_SCHEMA_KEYS = {"additionalProperties", "multipleOf", "exclusiveMinimum", "exclusiveMaximum", "const", "$schema", "title", "examples"}


def _strip_unsupported_schema_keys(obj):
    """Recursively remove JSON Schema keys Vertex's function-calling Schema proto doesn't accept."""
    if isinstance(obj, dict):
        return {
            k: _strip_unsupported_schema_keys(v)
            for k, v in obj.items()
            if k not in _UNSUPPORTED_SCHEMA_KEYS
        }
    if isinstance(obj, list):
        return [_strip_unsupported_schema_keys(v) for v in obj]
    return obj


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
        self.ctx_id_to_pending_intent: dict[str, dict] = {}  # deferred call waiting on prereqs (list, batched)
        self.ctx_id_to_info_state: dict[str, dict] = {}      # "tool.field" → 0|1 binary vector
        self.ctx_id_to_judged_tools: dict[str, set[str]] = {}  # tools already sent through the generic prereq judge (Check O), judged once per conversation
        self.ctx_id_to_verifier_checked: dict[str, set[str]] = {}  # policy rule ids already checked by Check P, once per conversation

        # Reflexion-style lessons-learned log — deliberately INSTANCE-level, not
        # per-context_id, since it's meant to persist across different tasks
        # within the same running benchmark process (this executor instance
        # lives for the whole run). Populated whenever Check O/P catch a real
        # violation; a short summary is injected into the system prompt for
        # subsequent tasks so the agent doesn't repeat the same mistake
        # category. Capped to keep prompt growth bounded.
        self._reflection_log: list[str] = []
        self._REFLECTION_LOG_MAX = 5

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
                            reflection_block = ""
                            if self._reflection_log:
                                lessons = "\n".join(f"- {note}" for note in self._reflection_log)
                                reflection_block = (
                                    "\n\nLESSONS FROM EARLIER TASKS THIS SESSION (avoid repeating these):\n"
                                    + lessons
                                )
                            messages.append({
                                "role": "system",
                                "content": system_prompt + FORMAT_REMINDER + reflection_block,
                            })
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

        # Built early — needed both for the main LLM call below and to re-run
        # the policy layer on a refired pending intent (see below), which must
        # not bypass policy checks like Check F side effects.
        completion_kwargs = self._build_completion_kwargs(messages, tools)

        # Set when falling through from a preference-reconsideration pending
        # intent, so the grounding check below knows to verify the LLM either
        # acted or asked, instead of just claiming completion in text.
        reconsidering_tool_name: str | None = None

        # ── Pending intent: if we auto-injected prereq(s) last turn, fire the original ──
        pending = self.ctx_id_to_pending_intent.get(context.context_id)
        if pending and incoming_tool_results:
            tool_history = self.ctx_id_to_tool_history.get(context.context_id, {})
            if all(pt in tool_history for pt in pending.get("prereq_tools", [])):
                del self.ctx_id_to_pending_intent[context.context_id]
                if pending.get("needs_llm_reconsideration"):
                    # Preference text just arrived in `messages` as a tool
                    # result — don't replay the stale pre-preference args.
                    # Fall through to the normal LLM call below so the model
                    # can decide fresh with the preference text visible.
                    ctx_logger.info("Policy: preferences resolved, letting LLM reconsider", tool=pending["tool_name"])
                    reconsidering_tool_name = pending["tool_name"]
                else:
                    ctx_logger.info("Policy: firing pending intent", tool=pending["tool_name"])
                    pending_tc = {
                        "id": str(uuid4()),
                        "type": "function",
                        "function": {
                            "name": pending["tool_name"],
                            "arguments": json.dumps(pending["args"]),
                        },
                    }
                    # Route the refired call back through the policy layer — it
                    # may still need e.g. Check F side effects injected
                    # (AC/defrost/fog lights), not just a raw replay.
                    policy_result = self._apply_policy_layer(
                        ctx_id=context.context_id,
                        tool_calls=[pending_tc],
                        tools=tools,
                        messages=messages,
                        completion_kwargs=completion_kwargs,
                        ctx_logger=ctx_logger,
                    )
                    final_tool_calls: list[dict] | None = [pending_tc]
                    final_text: str | None = None
                    if policy_result is not None:
                        if policy_result["type"] == "replace_with_calls":
                            final_tool_calls = policy_result["tool_calls"]
                        elif policy_result["type"] == "replace_with_text":
                            final_tool_calls = None
                            final_text = policy_result["response"].choices[0].message.content
                        elif policy_result["type"] == "replace_with_text_literal":
                            final_tool_calls = None
                            final_text = policy_result["text"]
                    final_text = self._enforce_24h_format(final_text)

                    pending_parts = []
                    if final_tool_calls is not None:
                        pending_tool_calls_list = [
                            ToolCall(tool_name=tc["function"]["name"], arguments=json.loads(tc["function"]["arguments"]))
                            for tc in final_tool_calls
                        ]
                        pending_parts.append(new_data_part(ToolCallsData(tool_calls=pending_tool_calls_list).model_dump()))
                        assistant_msg = {"role": "assistant", "content": None, "tool_calls": final_tool_calls}
                    else:
                        pending_parts.append(new_text_part(final_text or ""))
                        assistant_msg = {"role": "assistant", "content": final_text}
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
                    elif policy_result["type"] == "replace_with_text_literal":
                        # No LLM re-call — deterministic template text, saves a
                        # full round-trip for checks whose response doesn't need
                        # natural phrasing to be correct.
                        assistant_content = {"role": "assistant", "content": policy_result["text"]}
                        tool_calls = None
                    elif policy_result["type"] == "replace_with_calls":
                        tool_calls = policy_result["tool_calls"]
                        assistant_content = {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": tool_calls,
                        }

            # ── Zero-tool-call fallback detector ───────────────────────────
            # Narrowly scoped to two precise triggers, to avoid misfiring on
            # legitimate no-tool-call replies (e.g. a correct final answer to
            # an informational question):
            #   1. The very first assistant turn — a flat non-answer to the
            #      user's opening request.
            #   2. Right after a preference-reconsideration pending intent
            #      falls through (see `reconsidering_tool_name` above) — the
            #      LLM was specifically given a chance to act on fresh
            #      preference info and instead just claimed completion in
            #      text with no tool call and no question. This is the
            #      documented "tool-use hallucination" pattern (base_7).
            content_text = (assistant_content.get("content") or "").strip()
            is_first_turn_nonanswer = len(messages) == 2  # [system, first user message]
            if (
                not tool_calls
                and content_text
                and not content_text.endswith("?")
                and (is_first_turn_nonanswer or reconsidering_tool_name)
            ):
                ctx_logger.info(
                    "Policy[fallback]: non-answer with no tool call, redirecting",
                    reason="reconsideration" if reconsidering_tool_name else "first_turn",
                    tool=reconsidering_tool_name,
                )
                if reconsidering_tool_name:
                    hint = (
                        f"You were reconsidering whether to call '{reconsidering_tool_name}' now "
                        "that preference information is available. Your last reply neither called "
                        "that tool (or an appropriate alternative) nor asked a specific question — "
                        "it just claimed the action was done without actually doing it. Either call "
                        "the tool now, or ask one specific question."
                    )
                else:
                    hint = (
                        "Your last reply neither called a tool nor asked the driver a specific "
                        "question — it was just a non-committal statement. Either call the "
                        "appropriate tool now to fulfill the request, or ask one specific "
                        "clarifying question if something is genuinely missing."
                    )
                # Direct re-call (not _generate_clarification) — tools must stay
                # enabled here since the whole point is letting the LLM call one.
                redo_messages = messages + [{"role": "user", "content": f"[AGENT CONTEXT: {hint}]"}]
                redo_resp = completion(messages=redo_messages, **completion_kwargs)
                redo_msg = redo_resp.choices[0].message
                assistant_content = redo_msg.model_dump(exclude_unset=True)
                tool_calls = assistant_content.get("tool_calls")
                if tool_calls:
                    # The redo may itself need policy checks (prereqs, etc.)
                    policy_result = self._apply_policy_layer(
                        ctx_id=context.context_id,
                        tool_calls=tool_calls,
                        tools=tools,
                        messages=messages,
                        completion_kwargs=completion_kwargs,
                        ctx_logger=ctx_logger,
                    )
                    if policy_result is not None:
                        if policy_result["type"] in ("replace_with_text", "replace_with_text_literal"):
                            tool_calls = None
                            assistant_content = (
                                {"role": "assistant", "content": policy_result["text"]}
                                if policy_result["type"] == "replace_with_text_literal"
                                else policy_result["response"].choices[0].message.model_dump(exclude_unset=True)
                            )
                        elif policy_result["type"] == "replace_with_calls":
                            tool_calls = policy_result["tool_calls"]
                            assistant_content = {"role": "assistant", "content": None, "tool_calls": tool_calls}

            # ── Check P: policy-compliance verifier ────────────────────────
            # Runs on every final text response (no tool calls) — generalized
            # (not gated by turn number or which tool was used) to catch any
            # response that falsely claims a completed action, plus the
            # semantic wiki policies (LLM-POL:007/012/021/022). If the redo
            # produces a real tool call, it's routed through the full policy
            # layer, same as any freshly proposed call. Redoes at most once
            # per turn by construction (a single if-block, not a loop), so it
            # cannot compound with the zero-tool-call fallback above into
            # repeated redo cycles.
            if not tool_calls and (assistant_content.get("content") or "").strip():
                verifier_hint = self._run_policy_verifier(
                    ctx_id=context.context_id,
                    content_text=assistant_content["content"],
                    completion_kwargs=completion_kwargs,
                    ctx_logger=ctx_logger,
                )
                if verifier_hint:
                    # Keep tools enabled — if a real action needs to happen, the
                    # redo should be able to call it, not just rephrase the text.
                    # Sample twice (self-consistency): the flagged issue is
                    # usually "claimed action with no backing tool call", so a
                    # candidate that actually emits a tool call is the more
                    # trustworthy fix than one that just rephrases the text —
                    # a single redo can land on either by chance from trial to
                    # trial, which is exactly what hurts Pass^3 without hurting
                    # Pass@1.
                    redo_messages = messages + [{"role": "user", "content": f"[AGENT CONTEXT: {verifier_hint}]"}]
                    redo_candidates = []
                    for _ in range(2):
                        try:
                            redo_candidates.append(completion(messages=redo_messages, **completion_kwargs))
                        except Exception:
                            continue
                    redo_resp = next(
                        (r for r in redo_candidates if r.choices[0].message.model_dump(exclude_unset=True).get("tool_calls")),
                        redo_candidates[0] if redo_candidates else None,
                    )
                    if redo_resp is None:
                        redo_resp = completion(messages=redo_messages, **completion_kwargs)
                    assistant_content = redo_resp.choices[0].message.model_dump(exclude_unset=True)
                    tool_calls = assistant_content.get("tool_calls")
                    if tool_calls:
                        # Route through the full policy layer, same as any newly
                        # proposed tool call — it may need Checks A-O applied.
                        policy_result = self._apply_policy_layer(
                            ctx_id=context.context_id,
                            tool_calls=tool_calls,
                            tools=tools,
                            messages=messages,
                            completion_kwargs=completion_kwargs,
                            ctx_logger=ctx_logger,
                        )
                        if policy_result is not None:
                            if policy_result["type"] in ("replace_with_text", "replace_with_text_literal"):
                                tool_calls = None
                                assistant_content = (
                                    {"role": "assistant", "content": policy_result["text"]}
                                    if policy_result["type"] == "replace_with_text_literal"
                                    else policy_result["response"].choices[0].message.model_dump(exclude_unset=True)
                                )
                            elif policy_result["type"] == "replace_with_calls":
                                tool_calls = policy_result["tool_calls"]
                                assistant_content = {"role": "assistant", "content": None, "tool_calls": tool_calls}

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

            # Deterministic safety net for LLM-POL:002 — rewrite any 12h AM/PM
            # mentions to 24h before the response is sent or stored in history.
            if assistant_content.get("content"):
                assistant_content["content"] = self._enforce_24h_format(assistant_content["content"])

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
        if context.context_id in self.ctx_id_to_judged_tools:
            del self.ctx_id_to_judged_tools[context.context_id]
        if context.context_id in self.ctx_id_to_verifier_checked:
            del self.ctx_id_to_verifier_checked[context.context_id]

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

    def _generate_clarification_consistent(self, messages: list[dict], hint: str, completion_kwargs: dict, n: int = 2):
        """Self-consistency variant of _generate_clarification: sample n
        independent redo attempts instead of trusting a single one, and pick
        the first non-empty response. Targets Pass^3 (trial-to-trial
        consistency) rather than Pass@1 -- a single redo can happen to land
        on an empty/weak response by chance on one trial and a good one on
        another for the exact same flagged issue; sampling reduces that
        variance without needing a new judging mechanism."""
        candidates = []
        for _ in range(n):
            try:
                resp = self._generate_clarification(messages, hint, completion_kwargs)
                candidates.append(resp)
                content = getattr(resp.choices[0].message, "content", None)
                if content and content.strip():
                    return resp
            except Exception:
                continue
        # None were clearly non-empty -- fall back to the first candidate we
        # got (even if empty), or None if every attempt raised.
        return candidates[0] if candidates else None

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

    def _last_user_message_is_affirmative(self, messages: list[dict]) -> bool:
        """True if the most recent user message reads as an explicit 'yes' confirmation."""
        user_msgs = [m for m in messages if m.get("role") == "user" and m.get("content")]
        if not user_msgs:
            return False
        last = user_msgs[-1]["content"].strip().lower()
        if len(last.split()) > 6:
            # Longer replies only count if they clearly open with confirmation.
            return any(last.startswith(p) for p in _AFFIRMATIVE_PATTERNS)
        return any(p in last for p in _AFFIRMATIVE_PATTERNS)

    def _build_completion_kwargs(self, messages: list[dict], tools: list[dict]) -> dict:
        """Build the litellm completion() kwargs. Shared by the main LLM call and
        by the pending-intent refire path, which must reuse the same kwargs when
        routing a refired call back through the policy layer."""
        is_anthropic = "claude" in self.model.lower() or "anthropic" in self.model.lower()

        # Prompt caching is Anthropic-only — skip for Ollama/Groq/Gemini etc.
        if is_anthropic:
            if tools:
                tools[-1]["function"]["cache_control"] = {"type": "ephemeral"}
            if messages:
                messages[0]["cache_control"] = {"type": "ephemeral"}

        completion_kwargs = {
            "model": self.model,
            "tools": tools if tools else None,
            # Bounded retry/timeout — prevents a single transient provider
            # error (e.g. Gemini 503 "high demand") from hanging a task for
            # an unbounded amount of time via litellm's own default backoff.
            "num_retries": LLM_NUM_RETRIES,
            "timeout": LLM_TIMEOUT_SECONDS,
        }

        # A Vertex AI fine-tuned endpoint is referenced by a bare numeric ID
        # (e.g. "vertex_ai/1947235642547109888"). LiteLLM only recognizes
        # this as a Gemini-family model (routing to generateContent) if
        # base_model is given explicitly -- otherwise it defaults to the
        # legacy non-Gemini Predict/RawPredict path, which fails outright
        # for a tuned Gemini model with a 400 error.
        model_id_part = self.model.split("/")[-1]
        if model_id_part.isdigit():
            completion_kwargs["base_model"] = "vertex_ai/gemini-2.5-flash"
            # LiteLLM's OpenAI-tools -> Gemini functionDeclarations conversion
            # keys off the model string containing "gemini" -- a bare numeric
            # endpoint ID never matches, so tools are otherwise passed through
            # untranslated and Vertex rejects the raw OpenAI shape outright.
            # Convert to Gemini's native tool schema ourselves before calling.
            # Vertex's function-calling Schema proto only supports a subset of
            # JSON Schema -- keys like "additionalProperties"/"multipleOf" are
            # rejected outright. LiteLLM's own conversion path strips these,
            # but that path never runs for a bare numeric model, so do it here.
            if completion_kwargs["tools"]:
                completion_kwargs["tools"] = [{
                    "function_declarations": [
                        _strip_unsupported_schema_keys(t["function"])
                        for t in completion_kwargs["tools"]
                    ]
                }]

        # Ollama needs a larger context window to fit 57 tool schemas (~10K tokens)
        if self.model.startswith("ollama/"):
            completion_kwargs["num_ctx"] = 16384

        if self.temperature is not None:
            completion_kwargs["temperature"] = self.temperature

        # Configure reasoning effort / thinking
        if self.thinking:
            if self.model == "claude-opus-4-6":
                completion_kwargs["thinking"] = {"type": "adaptive"}
            else:
                if self.reasoning_effort in ["none", "disable", "low", "medium", "high"]:
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

        return completion_kwargs

    def _enforce_24h_format(self, text: str | None) -> str | None:
        """Deterministically rewrite 12h AM/PM time mentions to 24h format (LLM-POL:002).
        Safety net alongside the prompt reminder — zero extra LLM calls."""
        if not text:
            return text

        def _convert(match: "re.Match") -> str:
            hour = int(match.group(1))
            minute = match.group(2) or "00"
            meridiem = match.group(3).lower()
            if meridiem.startswith("p") and hour != 12:
                hour += 12
            elif meridiem.startswith("a") and hour == 12:
                hour = 0
            return f"{hour:02d}:{minute}"

        return _TIME_12H_PATTERN.sub(_convert, text)

    def _strip_json_fences(self, text: str) -> str:
        """Strip markdown code fences (```json ... ``` or ``` ... ```) that
        providers sometimes wrap structured JSON output in, even when
        explicitly told to reply with only JSON. Safe no-op if none present.
        Applied before every json.loads() on a raw completion response in
        this file — a bare .strip() alone leaves fenced output unparseable."""
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[: -3]
        return text.strip()

    def _add_reflection(self, note: str) -> None:
        """Reflexion-style: record a short, generalized lesson from a caught
        violation (Check O/P), for injection into subsequent tasks' system
        prompt this run. Deduplicated and capped — see _REFLECTION_LOG_MAX."""
        if note in self._reflection_log:
            return
        self._reflection_log.append(note)
        if len(self._reflection_log) > self._REFLECTION_LOG_MAX:
            self._reflection_log.pop(0)

    def _describe_action(self, tc_name: str, tc_args: dict) -> str:
        """Template-based (no LLM call) human-readable description of a pending action."""
        if tc_name == "set_head_lights_high_beams":
            return "turn the high beams " + ("on" if tc_args.get("on") else "off")
        readable = tc_name.replace("_", " ")
        if not tc_args:
            return readable
        details = ", ".join(f"{k}={v}" for k, v in tc_args.items())
        return f"{readable} ({details})"

    def _judge_generic_prereq(self, messages: list[dict], tc_name: str, tc_args: dict, ctx_logger) -> str | None:
        """AgentGuard-style fallback for tools outside PREREQUISITE_TABLE: ask a
        lightweight, policy-grounded judge whether this call skips a
        prerequisite the wiki requires. Judged once per tool per conversation
        (see ctx_id_to_judged_tools) to bound cost — this only fires for the
        long tail of tools our hardcoded tables don't cover.
        Returns a hint string if a likely-missing prerequisite is found, else None."""
        wiki_text = next((m["content"] for m in messages if m.get("role") == "system"), "")
        prompt = (
            f"Policy excerpt:\n{wiki_text[:4000]}\n\n"
            f"The assistant is about to call '{tc_name}' with arguments {json.dumps(tc_args)}.\n"
            "Based ONLY on the policy above, does this call skip a prerequisite the policy "
            "requires (e.g. checking current state, weather, or confirmation first)? "
            "If the policy doesn't mention this tool or has no applicable prerequisite, answer false.\n"
            'Reply with only a JSON object: {"missing_prereq": true/false, "hint": "<short reason or empty>"}'
        )
        try:
            resp = completion(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.0,
                num_retries=LLM_NUM_RETRIES,
                timeout=LLM_TIMEOUT_SECONDS,
                response_format=GenericPrereqJudgment,
            )
            parsed = json.loads(self._strip_json_fences(resp.choices[0].message.content))
            if parsed.get("missing_prereq"):
                ctx_logger.info("Policy[O]: generic judge flagged missing prereq", tool=tc_name, hint=parsed.get("hint"))
                self._add_reflection(
                    f"Before calling '{tc_name}' (or similar tools not in the standard prereq "
                    "table), check whether the policy requires confirming current state first."
                )
                return parsed.get("hint") or f"There may be a missing prerequisite before calling {tc_name}."
        except Exception as e:
            ctx_logger.warning("Policy[O]: judge call failed, skipping", tool=tc_name, error=str(e))
        return None

    def _run_policy_verifier(
        self, ctx_id: str, content_text: str, completion_kwargs: dict, ctx_logger
    ) -> str | None:
        """Check P — policy-compliance verifier. Runs on every final text-only
        response (not gated by which tool was used) and checks two things in
        one call:
        (1) a general rule that applies to ANY response — does it falsely
        claim a state-changing action was completed with no tool call in this
        same response to back it up (the documented "tool-use hallucination"
        pattern — e.g. saying "I'm calling her now" with no matching tool
        call). Purely reporting already-retrieved information is NOT a
        violation of this rule.
        (2) the semantic-only wiki policies that can't be reduced to a
        deterministic table (LLM-POL:007/012/021/022), each checked at most
        once per conversation (scoped like the organizer's own evaluator —
        only checked if its trigger tool was actually used) to bound cost.
        Fails open on any error — never blocks a response over a judge
        failure. Returns a hint to redo the response if a violation is found,
        else None."""
        if not content_text:
            return None
        tool_history = self.ctx_id_to_tool_history.get(ctx_id, {})
        checked = self.ctx_id_to_verifier_checked.setdefault(ctx_id, set())
        applicable = [
            rule for rule in POLICY_VERIFIER_RULES
            if rule["id"] not in checked and rule["trigger_tools"] & tool_history.keys()
        ]
        for rule in applicable:
            checked.add(rule["id"])  # checked at most once per conversation, regardless of outcome

        general_rule = (
            "- [COMPLETION_CLAIM] The response must not claim, imply, or describe a "
            "state-changing action (turning something on/off, setting a value, calling "
            "someone, sending something, etc.) as already done unless a matching tool "
            "call actually accompanies this exact response. Purely reporting information "
            "already retrieved earlier (e.g. summarizing a prior GET result) is NOT a "
            "violation of this rule."
        )
        table_rules = "\n".join(f"- [{r['id']}] {r['policy_text']}" for r in applicable)
        policy_lines = general_rule + ("\n" + table_rules if table_rules else "")

        # Ground the judgment in actual evidence (which tools genuinely executed
        # this conversation) rather than judging the text in isolation — a
        # claim can only be "backed up" by a real tool call that actually
        # happened, not by how confidently the response is phrased.
        executed_tools = sorted(tool_history.keys()) if tool_history else []
        evidence_line = f"Tools actually executed so far in this conversation: {executed_tools}\n"

        prompt = (
            f"Policy lines to check:\n{policy_lines}\n\n"
            f"{evidence_line}\n"
            f'The car assistant is about to send this response to the driver, with NO tool '
            f'calls attached to this exact response:\n"{content_text}"\n\n'
            "For each policy line, was it followed by this response? Ground your answer in the "
            "executed-tools evidence above, not just the wording of the response. If a policy "
            "doesn't apply given what's known, or there isn't enough information to tell, answer "
            "true (followed).\n"
            'Reply with only a JSON object: {"results": [{"id": "...", "followed": true/false, "reason": "..."}]}'
        )
        try:
            kwargs = {k: v for k, v in completion_kwargs.items() if k != "tools"}
            kwargs["temperature"] = 0.0
            kwargs["response_format"] = PolicyVerifierResult
            resp = completion(messages=[{"role": "user", "content": prompt}], **kwargs)
            parsed = json.loads(self._strip_json_fences(resp.choices[0].message.content))
            results = parsed.get("results", [])
            violations = [r for r in results if not r.get("followed", True)]
            if violations:
                reasons = "; ".join(f"{v.get('id')}: {v.get('reason', '')}" for v in violations)
                ctx_logger.info("Policy[P]: compliance issue flagged", violations=reasons)
                for v in violations:
                    if v.get("id") == "COMPLETION_CLAIM":
                        self._add_reflection(
                            "Never describe a state-changing action as done unless the matching "
                            "tool call is actually included in that same response."
                        )
                    else:
                        self._add_reflection(f"Watch for {v.get('id')}: {v.get('reason', '')[:100]}")
                return (
                    f"Before finalizing, address this: {reasons}. If an action actually needs "
                    "to happen, call the appropriate tool now instead of just describing it. "
                    "Otherwise, revise your reply to the driver, keeping it natural and brief."
                )
        except Exception as e:
            ctx_logger.warning("Policy[P]: verifier call failed, skipping", error=str(e))
        return None

    def _generate_evpi_clarification(self, user_request: str, messages: list[dict], completion_kwargs: dict):
        """Generate the most informative clarifying question when tool-level ambiguity
        is detected. Collapsed to 2 LLM calls (was up to 6): one call produces
        interpretations, candidate questions, AND each candidate's predicted
        yes/no answer per interpretation in a single structured response; the
        actual expected-information-gain scoring (how balanced each candidate's
        yes/no split is) is then computed deterministically in Python — no LLM
        call per candidate. A final call phrases the winner naturally."""
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

        # Step 1: one call producing interpretations + candidate questions +
        # each candidate's predicted yes/no answer per interpretation.
        kwargs["temperature"] = 0.7
        combined_prompt = (
            f'The car driver said: "{user_request}"\n'
            "1. List 3-4 distinct possible interpretations of what they might want. "
            "Each must reference a specific car system (AC, seat heating, windows, navigation, etc.).\n"
            "2. Generate 3 short yes/no clarifying questions that could help distinguish between those "
            "interpretations. Questions must be natural, brief, and safe for a driver to answer.\n"
            "3. For each of the 3 questions, predict whether the answer would be YES or NO under each "
            "interpretation, in the same order as the interpretations list.\n"
            "Reply with only a JSON object of this exact shape:\n"
            '{"interpretations": ["...", "..."], '
            '"candidates": [{"question": "...", "answers": ["YES", "NO", ...]}, ...]}'
        )
        try:
            resp = completion(
                messages=[{"role": "user", "content": combined_prompt}],
                response_format=EVPIResponse,
                **kwargs,
            )
            data = json.loads(self._strip_json_fences(resp.choices[0].message.content))
            interpretations = data["interpretations"]
            candidates = data["candidates"]
            if not isinstance(interpretations, list) or len(interpretations) < 2 or not candidates:
                return None
        except Exception:
            return None

        # Step 2: pick the candidate with the most balanced YES/NO split (best
        # expected information gain) — computed deterministically, no LLM call.
        def _split_score(candidate: dict) -> int:
            answers = candidate.get("answers", [])
            yes_count = sum(1 for a in answers if str(a).upper().startswith("Y"))
            no_count = len(answers) - yes_count
            return abs(yes_count - no_count)

        best_candidate = min(candidates, key=_split_score)
        best_question = best_candidate.get("question", "")
        if not best_question:
            return None

        # Step 3: phrase it naturally in the agent's voice
        kwargs["temperature"] = 0.0
        phrase_messages = messages + [{
            "role": "user",
            "content": f"[AGENT CONTEXT: Ask the driver this one clarifying question naturally and briefly: {best_question}]"
        }]
        _EVPI_CACHE.set(cache_key, best_question, expire=86400)  # 24h TTL
        return completion(messages=phrase_messages, **kwargs)

    # ── Post-action side-effect policies (AUT-POL:010/011/013) ────────────────
    # These fire *after* a trigger call is accepted, using state already fetched
    # via PREREQUISITE_TABLE (get_vehicle_window_positions / get_climate_settings /
    # get_exterior_lights_status are mandatory prereqs for these same trigger
    # tools, so their results are already cached in tool_history by this point).

    def _get_tool_result(self, ctx_id: str, tool_name: str) -> dict:
        """Parse a cached tool_history[tool_name] JSON string into its result dict."""
        raw = self.ctx_id_to_tool_history.get(ctx_id, {}).get(tool_name)
        if not raw:
            return {}
        try:
            parsed = json.loads(raw) if isinstance(raw, str) else raw
            return parsed.get("result", parsed) if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, AttributeError):
            return {}

    def _side_effects_air_conditioning(self, ctx_id: str, tc_args: dict) -> list[tuple[str, dict]]:
        """AUT-POL:011 — AC on must close windows >20% open and raise fan speed off 0."""
        if not tc_args.get("on"):
            return []
        effects: list[tuple[str, dict]] = []
        windows = self._get_tool_result(ctx_id, "get_vehicle_window_positions")
        open_windows = [
            enum for enum, field in WINDOW_POSITION_FIELDS.items()
            if (windows.get(field) or 0) > 20
        ]
        if open_windows and len(open_windows) == len(WINDOW_POSITION_FIELDS):
            effects.append(("open_close_window", {"window": "ALL", "percentage": 0}))
        else:
            effects.extend(("open_close_window", {"window": w, "percentage": 0}) for w in open_windows)
        climate = self._get_tool_result(ctx_id, "get_climate_settings")
        if (climate.get("fan_speed") or 0) == 0:
            effects.append(("set_fan_speed", {"level": 1}))
        return effects

    def _side_effects_window_defrost(self, ctx_id: str, tc_args: dict) -> list[tuple[str, dict]]:
        """AUT-POL:010 — defrost on must raise fan speed, include WINDSHIELD airflow, and turn on AC."""
        if not tc_args.get("on"):
            return []
        effects: list[tuple[str, dict]] = []
        climate = self._get_tool_result(ctx_id, "get_climate_settings")
        if (climate.get("fan_speed") or 0) < 2:
            effects.append(("set_fan_speed", {"level": 2}))
        if "WINDSHIELD" not in (climate.get("fan_airflow_direction") or ""):
            effects.append(("set_fan_airflow_direction", {"direction": "WINDSHIELD"}))
        if not climate.get("air_conditioning", False):
            effects.append(("set_air_conditioning", {"on": True}))
        return effects

    def _side_effects_fog_lights(self, ctx_id: str, tc_args: dict) -> list[tuple[str, dict]]:
        """AUT-POL:013 — fog lights on must ensure low beams on and high beams off."""
        if not tc_args.get("on"):
            return []
        effects: list[tuple[str, dict]] = []
        lights = self._get_tool_result(ctx_id, "get_exterior_lights_status")
        if lights.get("head_lights_low_beams") is False:
            effects.append(("set_head_lights_low_beams", {"on": True}))
        if lights.get("head_lights_high_beams") is True:
            effects.append(("set_head_lights_high_beams", {"on": False}))
        return effects

    def _compute_side_effects(self, ctx_id: str, tc_name: str, tc_args: dict) -> list[tuple[str, dict]]:
        if tc_name == "set_air_conditioning":
            return self._side_effects_air_conditioning(ctx_id, tc_args)
        if tc_name == "set_window_defrost":
            return self._side_effects_window_defrost(ctx_id, tc_args)
        if tc_name == "set_fog_lights":
            return self._side_effects_fog_lights(ctx_id, tc_args)
        return []

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
                text = "Sorry, I'm not able to do that with this car — that feature isn't available here."
                return {"type": "replace_with_text_literal", "text": text}

            # ── Check I: explicit confirmation required (LLM-POL:004) ─────
            if tc_name in CONFIRMATION_REQUIRED_TOOLS:
                if not self._last_user_message_is_affirmative(messages):
                    ctx_logger.info("Policy[I]: confirmation required, not yet given", tool=tc_name)
                    text = f"Just to confirm — I'm about to {self._describe_action(tc_name, tc_args)}. Should I go ahead?"
                    return {"type": "replace_with_text_literal", "text": text}

            # ── Check B: Prerequisites (E1 fix) ──────────────────────────
            # Collect ALL missing auto-injectable prereqs at once instead of one
            # per turn — cuts N sequential evaluator round-trips to 1 for tools
            # like set_air_conditioning that need multiple independent GET calls.
            missing_auto: list[dict] = []
            missing_manual: dict | None = None
            for prereq in PREREQUISITE_TABLE.get(tc_name, []):
                prereq_tool = prereq["requires_tool"]
                if prereq_tool not in tool_history:
                    if prereq["auto"]:
                        missing_auto.append(prereq)
                    elif missing_manual is None:
                        missing_manual = prereq

            if missing_auto:
                prereq_tools = [p["requires_tool"] for p in missing_auto]
                ctx_logger.info("Policy[B]: auto-injecting prereqs (batched)", prereqs=prereq_tools, for_tool=tc_name)
                self.ctx_id_to_pending_intent[ctx_id] = {
                    "tool_name": tc_name,
                    "args": tc_args,
                    "prereq_tools": prereq_tools,
                }
                prereq_tcs = [
                    {
                        "id": str(uuid4()),
                        "type": "function",
                        "function": {
                            "name": p["requires_tool"],
                            "arguments": json.dumps(p.get("prereq_args", {})),
                        },
                    }
                    for p in missing_auto
                ]
                return {"type": "replace_with_calls", "tool_calls": prereq_tcs}
            elif missing_manual is not None:
                ctx_logger.info("Policy[B]: need user info", prereq=missing_manual["requires_tool"], for_tool=tc_name)
                return {"type": "replace_with_text", "response": self._generate_clarification(messages, missing_manual["question_hint"], completion_kwargs)}

            # ── Check G: mutual exclusion — high beams blocked while fog lights on (AUT-POL:014) ──
            if tc_name == "set_head_lights_high_beams" and tc_args.get("on"):
                lights = self._get_tool_result(ctx_id, "get_exterior_lights_status")
                if lights.get("fog_lights") is True:
                    ctx_logger.info("Policy[G]: high beams blocked, fog lights on")
                    text = "I can't turn on the high beams while the fog lights are on — want me to turn those off first?"
                    return {"type": "replace_with_text_literal", "text": text}

            # ── Check C: Missing response fields (hallucination_missing_tool_response fix) ──
            missing = self._check_missing_response_fields(ctx_id, tc_name)
            if missing:
                ctx_logger.info("Policy[C]: missing response fields", tool=tc_name, missing=list(missing))
                field_labels = [f.split(".", 1)[-1] for f in missing]
                text = (
                    f"I wasn't able to confirm {', '.join(field_labels)} from the car, "
                    "so I can't safely do that right now."
                )
                return {"type": "replace_with_text_literal", "text": text}

            # ── Check E: Preference auto-injection ────────────────────────────────
            # If agent is about to call a preference-sensitive tool but hasn't
            # fetched user preferences yet, inject get_user_preferences first —
            # but ONLY when the proposed call is actually incomplete. Explicit
            # user request outranks learned preferences (wiki Priority 1 > 2):
            # if the LLM already has a fully-specified call (e.g. the user just
            # said "level 3"), there's nothing left for a preference to resolve
            # — checking anyway wastes a full LLM round-trip for no behavioral
            # difference in the outcome.
            call_is_incomplete = bool(self._get_missing_required_params(tc_name, tc_args, tools))
            if (
                tc_name in PREFERENCE_PREREQUISITE_TABLE
                and call_is_incomplete
                and "get_user_preferences" not in tool_history
            ):
                pref_categories = PREFERENCE_PREREQUISITE_TABLE[tc_name]
                ctx_logger.info("Policy[E]: auto-injecting get_user_preferences", for_tool=tc_name)
                self.ctx_id_to_pending_intent[ctx_id] = {
                    "tool_name": tc_name,
                    "args": tc_args,
                    "prereq_tools": ["get_user_preferences"],
                    # Preferences are free-text (List[str], not structured
                    # key/value data) — only an LLM can interpret them, so the
                    # original stale args must not be blindly replayed. Let the
                    # LLM re-decide once it can see the preference text.
                    "needs_llm_reconsideration": True,
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

            # ── Check O: generic prereq fallback for tools outside PREREQUISITE_TABLE ──
            # AgentGuard-style safety net: PREREQUISITE_TABLE only covers a fixed
            # ~13-tool list. For any other state-changing call, ask a lightweight
            # policy-grounded judge once per tool per conversation whether it
            # skips a wiki-required prerequisite, instead of leaving the long
            # tail of uncovered tools with zero prereq checking.
            if (
                tc_name.startswith(STATE_CHANGING_PREFIXES)
                and tc_name not in PREREQUISITE_TABLE
                and tc_name not in self.ctx_id_to_judged_tools.get(ctx_id, set())
            ):
                self.ctx_id_to_judged_tools.setdefault(ctx_id, set()).add(tc_name)
                verdict_hint = self._judge_generic_prereq(messages, tc_name, tc_args, ctx_logger)
                if verdict_hint:
                    return {"type": "replace_with_text", "response": self._generate_clarification_consistent(messages, verdict_hint, completion_kwargs)}

        # ── Check H: nav-edit tools must not run in parallel (TECH-AUT-POL:018) ──
        nav_edit_calls = [tc for tc in tool_calls if tc["function"]["name"] in NAVIGATION_EDIT_TOOLS]
        if len(nav_edit_calls) > 1:
            ctx_logger.info(
                "Policy[H]: multiple nav-edit tools in parallel, keeping first only",
                tools=[tc["function"]["name"] for tc in nav_edit_calls],
            )
            non_nav_calls = [tc for tc in tool_calls if tc["function"]["name"] not in NAVIGATION_EDIT_TOOLS]
            return {"type": "replace_with_calls", "tool_calls": non_nav_calls + [nav_edit_calls[0]]}

        # ── Check F: post-action side effects (AUT-POL:010/011/013) ──────────────
        # All tool_calls in this batch passed A/B/C/E/G above, so any prereq state
        # these side effects depend on is already cached in tool_history. Inject
        # the dependent calls alongside the original ones in the same batch —
        # the evaluator runs same-turn tool_calls in parallel.
        extra_effects: list[tuple[str, dict]] = []
        for tc in tool_calls:
            tc_name_f = tc["function"]["name"]
            try:
                tc_args_f = json.loads(tc["function"]["arguments"]) if isinstance(tc["function"]["arguments"], str) else (tc["function"]["arguments"] or {})
            except (json.JSONDecodeError, TypeError):
                tc_args_f = {}
            extra_effects.extend(self._compute_side_effects(ctx_id, tc_name_f, tc_args_f))

        if extra_effects:
            ctx_logger.info("Policy[F]: injecting side-effect calls", effects=[e[0] for e in extra_effects])
            new_tool_calls = list(tool_calls)
            for effect_name, effect_args in extra_effects:
                new_tool_calls.append({
                    "id": str(uuid4()),
                    "type": "function",
                    "function": {"name": effect_name, "arguments": json.dumps(effect_args)},
                })
            return {"type": "replace_with_calls", "tool_calls": new_tool_calls}

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
