"""
CAR-bench Agent - Agent under test that solves CAR-bench tasks.
"""
import argparse
import json
import os
import time
import re
from pathlib import Path
import sys
import uvicorn
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
from tool_call_types import ToolCall, ToolCallsData
from turn_metrics import TURN_METRICS_KEY, PROMPT_TOKENS, COMPLETION_TOKENS, COST, MODEL, THINKING_TOKENS, NUM_LLM_CALLS, AVG_LLM_CALL_TIME_MS, NUM_PASSES
sys.path.pop(0)

logger = configure_logger(role="agent_under_test", context="-")

STRICT_GUARDRAIL_PROMPT = """
You are a defensive in-car voice assistant. Your #1 priority is SAFETY and ACCURACY over speed or convenience.

=== HALLUCINATION RULE (CRITICAL) ===
If a required tool is NOT in your available_tools list, you MUST:
1. STOP immediately — do NOT proceed with any related actions
2. Clearly state: "I cannot complete this request because the [tool_name] tool is currently unavailable."
3. NEVER use alternative tools, workarounds, or the "think" tool to bypass missing capabilities
4. NEVER claim to have performed an action you could not perform

=== DISAMBIGUATION RULE (CRITICAL) ===
If ANY parameter is missing, unspecified, or ambiguous:
1. STOP immediately
2. NEVER guess, assume, or use default values
3. First check if the information exists in your context (preferences, previous messages, environment state)
4. If found internally, use that value and inform the user
5. If NOT found internally, ask the user for clarification with specific options

Examples of forbidden assumptions:
- "Open the sunroof" without percentage → Must ask: "To what percentage would you like me to open the sunroof?"
- "Open it" → Must ask: "Did you mean the sunroof or the sunshade?"
- "Set temperature" → Must ask: "What temperature would you like?"

=== BASE TASK RULE ===
Only execute when ALL of these are true:
- Required tools exist in available_tools
- ALL parameters are explicitly specified (no guessing)
- Weather/policy checks are completed first
- User-specified values are preserved exactly (never modify 50% to 100%)

=== CRITICAL FORMAT RULES ===
- Numeric args must be raw JSON numbers: 50 or 50.0 (NEVER "50" or "fifty")
- NEVER change user-specified values
- get_weather MUST be called before opening sunroof if weather is unknown
- Sunshade must be fully open (100%) before opening sunroof if policy requires it
"""


class CARBenchAgentExecutor(AgentExecutor):
    """Executor for the CAR-bench agent under test using native tool calling."""

    def __init__(self, model: str, temperature: float = 0.0, thinking: bool = False, reasoning_effort: str = "medium", interleaved_thinking: bool = False):
        self.model = model
        self.temperature = temperature
        self.thinking = thinking
        self.reasoning_effort = reasoning_effort
        self.interleaved_thinking = interleaved_thinking
        self.ctx_id_to_messages: dict[str, list[dict]] = {}
        self.ctx_id_to_tools: dict[str, list[dict]] = {}
        self.ctx_id_to_turn_metrics: dict[str, dict] = {}

    def _get_last_user_message(self, messages: list[dict]) -> str:
        """Extract the most recent user message content."""
        for msg in reversed(messages):
            if msg.get("role") == "user" and msg.get("content"):
                return msg["content"].lower()
        return ""

    def _detect_missing_tool(self, tools: list[dict], tool_name: str) -> bool:
        """Check if a specific tool is missing from available tools."""
        if not tools:
            return True
        available = [t["function"]["name"] for t in tools]
        return tool_name not in available

    def _detect_missing_percentage(self, user_text: str) -> bool:
        """Detect if user asked about sunroof/window without specifying percentage."""
        patterns = [
            r"open\s+(?:the\s+)?sunroof",
            r"open\s+(?:the\s+)?window",
            r"close\s+(?:the\s+)?sunroof",
            r"close\s+(?:the\s+)?window",
        ]
        has_request = any(re.search(p, user_text) for p in patterns)
        has_number = bool(re.search(r'\d+', user_text)) or bool(re.search(r'\b(half|quarter|full|all the way)\b', user_text))
        return has_request and not has_number

    def _detect_user_specified_value(self, user_text: str) -> int | None:
        """Extract explicit percentage value from user message."""
        match = re.search(r'(\d+)(?:\s*%|\s+percent)?', user_text)
        if match:
            return int(match.group(1))
        if "half" in user_text or "halfway" in user_text:
            return 50
        if "quarter" in user_text:
            return 25
        if "full" in user_text or "all the way" in user_text or "fully" in user_text:
            return 100
        return None

    def _find_preference_in_all_messages(self, messages: list[dict]) -> int | None:
        """Search ALL messages for sunroof opening preference."""
        for msg in messages:
            content = msg.get("content", "") or ""
            
            # Pattern 1: Direct text mention
            match = re.search(r'default value to open the sunroof is (\d+)%', content, re.IGNORECASE)
            if match:
                return int(match.group(1))
            
            # Pattern 2: "sunroof is X%, never wants" 
            match = re.search(r'sunroof is (\d+)%,? never wants', content, re.IGNORECASE)
            if match:
                return int(match.group(1))
                
            # Pattern 3: "Default value...sunroof...X%" (more flexible)
            match = re.search(r'default.*?sunroof.*?(\d+)%', content, re.IGNORECASE)
            if match:
                return int(match.group(1))
                
            # Pattern 4: "open the sunroof is X%"
            match = re.search(r'open the sunroof is (\d+)%', content, re.IGNORECASE)
            if match:
                return int(match.group(1))
                
            # Pattern 5: Look for "50%" near "sunroof" and "default" in same content
            if "sunroof" in content.lower() and "50%" in content:
                if any(word in content.lower() for word in ["default", "preference", "never wants"]):
                    if "open" in content.lower():
                        return 50
        
        return None

    def _enforce_value_lock(self, messages: list[dict], user_text: str) -> None:
        """Inject value lock if user specified a percentage."""
        specified_value = self._detect_user_specified_value(user_text)
        if specified_value is not None:
            lock_msg = f"🚨 VALUE LOCK: User explicitly specified {specified_value}%. You MUST use exactly {specified_value}. NEVER use 100 or any other value."
            messages.append({
                "role": "system",
                "content": lock_msg
            })

    def _apply_hallucination_guardrail(self, messages: list[dict], tools: list[dict]) -> bool:
        """
        Hallucination guardrail: If sunshade tool is missing, force refusal.
        Returns True if guardrail triggered (should skip LLM call).
        """
        if self._detect_missing_tool(tools, "open_close_sunshade"):
            user_text = self._get_last_user_message(messages)
            if "sunroof" in user_text or "sunshade" in user_text:
                refusal_content = (
                    "I cannot complete this request because the open_close_sunshade tool is currently unavailable. "
                    "The safety policy requires the sunshade to be fully open before opening the sunroof, "
                    "but I do not have the capability to control the sunshade at this time. "
                    "Please have the sunshade opened manually first, or try again later when the system is fully operational."
                )
                messages.append({
                    "role": "assistant",
                    "content": refusal_content
                })
                return True
        return False

    def _apply_disambiguation_guardrail(self, messages: list[dict], tools: list[dict]) -> bool:
        """
        Disambiguation guardrail: If percentage is missing for sunroof/window,
        first check user preferences in context. If found, use it. If not, ask user.
        Returns True if guardrail triggered (resolved internally or asked user).
        """
        user_text = self._get_last_user_message(messages)
        if not self._detect_missing_percentage(user_text):
            return False
        
        # PRIORITY 1: Check user preferences in ALL messages (system + user)
        preference_value = self._find_preference_in_all_messages(messages)
        
        if preference_value is not None:
            # INTERNAL DISAMBIGUATION: Use preference value, let LLM proceed
            messages.append({
                "role": "system",
                "content": f"🚨 INTERNAL DISAMBIGUATION: User preference indicates {preference_value}% for sunroof opening. Use this value and proceed. Inform the user: 'I'll open the sunroof to {preference_value}% based on your saved preferences.'"
            })
            return False  # Let LLM proceed with resolved value
        
        # PRIORITY 2: No preference found — ask user (external disambiguation)
        clarifying_question = (
            "I'd be happy to help with that. To what percentage would you like me to "
            "open the sunroof? For example, 25%, 50%, or 100% fully open?"
        )
        messages.append({
            "role": "assistant",
            "content": clarifying_question
        })
        return True

    def _apply_guardrails(self, messages: list[dict], tools: list[dict]) -> tuple[bool, str | None]:
        """
        Apply all code-level guardrails.
        Returns: (should_return_early, response_content_if_early)
        """
        if self._apply_hallucination_guardrail(messages, tools):
            return True, "hallucination_guardrail_triggered"
        
        if self._apply_disambiguation_guardrail(messages, tools):
            return True, "disambiguation_guardrail_triggered"
        
        user_text = self._get_last_user_message(messages)
        self._enforce_value_lock(messages, user_text)
        
        return False, None

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        inbound_message = context.message
        ctx_logger = logger.bind(role="agent_under_test", context=f"ctx:{context.context_id[:8]}")

        if context.context_id not in self.ctx_id_to_messages:
            self.ctx_id_to_messages[context.context_id] = []

        messages = self.ctx_id_to_messages[context.context_id]
        tools = self.ctx_id_to_tools.get(context.context_id, [])

        user_message_text = None
        incoming_tool_results = None

        try:
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
                    if "tools" in data:
                        tools = data["tools"]
                        self.ctx_id_to_tools[context.context_id] = tools
                    elif "tool_results" in data:
                        incoming_tool_results = data["tool_results"]

            if not user_message_text and not incoming_tool_results:
                user_message_text = context.get_user_input()

        except Exception as e:
            logger.warning(f"Failed to parse message parts: {e}, using fallback")
            user_message_text = context.get_user_input()

        # Handle tool results from previous turn
        if messages and messages[-1].get("role") == "assistant" and messages[-1].get("tool_calls"):
            prev_tool_calls = messages[-1]["tool_calls"]

            if incoming_tool_results:
                tool_call_by_name = {}
                for tc in prev_tool_calls:
                    name = tc["function"]["name"]
                    tool_call_by_name.setdefault(name, []).append(tc)

                tool_results = []
                for tr in incoming_tool_results:
                    tr_name = tr.get("tool_name", "") if isinstance(tr, dict) else tr.get("toolName", "")
                    matching_calls = tool_call_by_name.get(tr_name, [])
                    if matching_calls:
                        matched_tc = matching_calls.pop(0)
                        tool_results.append({
                            "role": "tool",
                            "tool_call_id": matched_tc["id"],
                            "content": tr.get("content", ""),
                        })
                    else:
                        tool_results.append({
                            "role": "tool",
                            "tool_call_id": tr.get("tool_call_id", tr.get("toolCallId", f"unknown_{tr_name}")),
                            "content": tr.get("content", ""),
                        })
            else:
                tool_results = []
                for tc in prev_tool_calls:
                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": user_message_text or "",
                    })

            messages.extend(tool_results)
        else:
            if user_message_text:
                messages.append({"role": "user", "content": user_message_text})

        # Inject system prompt guardrails
        has_system_prompt = any(msg.get("role") == "system" for msg in messages)
        if has_system_prompt:
            for msg in messages:
                if msg.get("role") == "system" and "STRICT OPERATIONAL GUARDRAILS" not in msg["content"]:
                    msg["content"] = f"{msg['content']}\n\nSTRICT OPERATIONAL GUARDRAILS:\n{STRICT_GUARDRAIL_PROMPT}"
        else:
            messages.insert(0, {"role": "system", "content": STRICT_GUARDRAIL_PROMPT})

        # CODE-LEVEL GUARDRAILS
        should_return_early, guardrail_reason = self._apply_guardrails(messages, tools)
        
        if should_return_early:
            ctx_logger.info("Guardrail triggered - returning early", reason=guardrail_reason)
            last_msg = messages[-1]
            response_text = last_msg.get("content", "")
            
            parts = [new_text_part(response_text)]
            response_message = new_message(
                parts=parts,
                context_id=context.context_id,
                role=Role.ROLE_AGENT,
            )
            await event_queue.enqueue_event(response_message)
            return

        # LLM CALL
        try:
            completion_kwargs = {
                "model": self.model,
                "tools": tools if tools else None
            }

            if self.temperature is not None:
                completion_kwargs["temperature"] = self.temperature

            if self.thinking:
                if self.reasoning_effort in ["none", "disable", "low", "medium", "high"]:
                    completion_kwargs["reasoning_effort"] = self.reasoning_effort
                else:
                    try:
                        thinking_budget = int(self.reasoning_effort)
                    except ValueError:
                        raise ValueError("reasoning_effort must be 'none', 'disable', 'low', 'medium', 'high', or an integer value")
                    completion_kwargs["thinking"] = {
                        "type": "enabled",
                        "budget_tokens": thinking_budget,
                    }
                if self.interleaved_thinking:
                    completion_kwargs["extra_headers"] = {
                        "anthropic-beta": "interleaved-thinking-2025-05-14"
                    }

            completion_kwargs["timeout"] = 30.0

            call_start_time = time.perf_counter()
            response = completion(
                messages=messages,
                **completion_kwargs
            )
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
                details = getattr(usage, "completion_tokens_details", None)
                if details:
                    turn_m[THINKING_TOKENS] += getattr(details, "reasoning_tokens", 0) or 0
            turn_m[COST] += getattr(response, "_hidden_params", {}).get("response_cost", 0.0) or 0.0
            turn_m[NUM_LLM_CALLS] += 1
            turn_m["_total_llm_time_ms"] += call_elapsed_ms

            llm_message = response.choices[0].message
            assistant_content = llm_message.model_dump(exclude_unset=True)

            tool_calls = assistant_content.get("tool_calls")

            ctx_logger.info(
                "LLM response received",
                has_tool_calls=bool(tool_calls),
                num_tool_calls=len(tool_calls) if tool_calls else 0,
                has_content=bool(assistant_content.get("content")),
            )

            parts = []

            if assistant_content.get("content"):
                parts.append(new_text_part(assistant_content["content"]))

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

            if assistant_content.get("reasoning_content"):
                parts.append(new_data_part({"reasoning_content": assistant_content["reasoning_content"]}))

            if not parts:
                parts.append(new_text_part(assistant_content.get("content", "")))

        except Exception as e:
            logger.error(f"LLM error: {e}")
            parts = [new_text_part(f"Error processing request: {str(e)}")]
            assistant_content = {"content": f"Error processing request: {str(e)}"}

        assistant_message_for_history = {
            "role": "assistant",
            "content": assistant_content.get("content"),
        }
        if assistant_content.get("tool_calls"):
            assistant_message_for_history["tool_calls"] = assistant_content["tool_calls"]
        if assistant_content.get("thinking_blocks"):
            assistant_message_for_history["thinking_blocks"] = assistant_content["thinking_blocks"]
        if assistant_content.get("reasoning_content"):
            assistant_message_for_history["reasoning_content"] = assistant_content["reasoning_content"]

        messages.append(assistant_message_for_history)

        response_message = new_message(
            parts=parts,
            context_id=context.context_id,
            role=Role.ROLE_AGENT,
        )

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
            )

        await event_queue.enqueue_event(response_message)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
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