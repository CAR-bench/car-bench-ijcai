"""Planner/executor variant of the Track 2 Cerebras CAR-bench agent."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))
from track_2_agent_under_test_cerebras.car_bench_agent import (
    AgentInferenceResult,
    CEREBRAS_DEVELOPER_INSTRUCTIONS,
    NEXT_ACTION_OUTPUT_SCHEMA,
    CARBenchAgentExecutor as CerebrasNextActionExecutor,
    build_next_action_prompt,
    compact_tools_for_prompt,
    logger as cerebras_agent_logger,
    parse_next_action,
)
from track_2_agent_under_test_cerebras.cerebras_client import (
    DEFAULT_CEREBRAS_API_BASE,
    DEFAULT_EXECUTOR_MODEL,
    DEFAULT_EXECUTOR_REASONING_EFFORT,
    CerebrasCompletionClient,
    CerebrasTemplateError,
    MalformedModelResponseError,
    TokenUsage,
    add_token_usage,
)
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
)
sys.path.pop(0)


DEFAULT_PLANNER_MAX_COMPLETION_TOKENS = 4096
DEFAULT_EXECUTOR_MAX_COMPLETION_TOKENS = 1024
DEFAULT_PLANNER_MODEL = DEFAULT_EXECUTOR_MODEL
DEFAULT_PLANNER_REASONING_EFFORT = "high"


class PlannerExecutorCARBenchAgentExecutor(CerebrasNextActionExecutor):
    """A2A executor with Cerebras private planning and Cerebras execution."""

    def __init__(
        self,
        *,
        planner_model: str = DEFAULT_PLANNER_MODEL,
        executor_model: str = DEFAULT_EXECUTOR_MODEL,
        planner_max_completion_tokens: int = DEFAULT_PLANNER_MAX_COMPLETION_TOKENS,
        executor_max_completion_tokens: int = DEFAULT_EXECUTOR_MAX_COMPLETION_TOKENS,
        api_base: str | None = DEFAULT_CEREBRAS_API_BASE,
        service_tier: str | None = None,
        temperature: float | None = None,
        planner_temperature: float | None = None,
        executor_temperature: float | None = None,
        planner_reasoning_effort: str | None = DEFAULT_PLANNER_REASONING_EFFORT,
        executor_reasoning_effort: str | None = DEFAULT_EXECUTOR_REASONING_EFFORT,
        malformed_retries: int = 1,
    ) -> None:
        if executor_temperature is None:
            executor_temperature = temperature
        super().__init__(
            model=executor_model,
            api_base=api_base,
            service_tier=service_tier,
            temperature=executor_temperature,
            reasoning_effort=executor_reasoning_effort,
            max_completion_tokens=executor_max_completion_tokens,
            malformed_retries=malformed_retries,
        )
        self.planner_model = planner_model
        self.executor_model = executor_model
        self.planner_max_completion_tokens = planner_max_completion_tokens
        self.executor_max_completion_tokens = executor_max_completion_tokens
        self.planner_temperature = planner_temperature
        self.executor_temperature = executor_temperature
        self.planner_reasoning_effort = planner_reasoning_effort
        self.executor_reasoning_effort = executor_reasoning_effort
        self.planner_client = CerebrasCompletionClient(
            api_base=api_base,
            service_tier=service_tier,
            logger=cerebras_agent_logger.bind(
                role="agent_under_test",
                context="planner",
            ),
        )
        self._last_internal_call_count = 0
        self._active_private_plans_by_context: dict[str, dict[str, Any]] = {}

    async def cancel(self, context, event_queue) -> None:
        self._active_private_plans_by_context.pop(context.context_id, None)
        await super().cancel(context, event_queue)

    def _on_policy_controlled_action(
        self,
        context_id: str,
        controlled_action,
    ) -> None:
        self._active_private_plans_by_context.pop(context_id, None)

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
        internal_call_count = 0
        planner_ms = 0.0
        plan_source = "new_user_turn"
        private_plan: dict[str, Any] | None = None
        last_model_stage = "planner"
        last_model_text = ""
        last_finish_reason: str | None = None
        last_token_usage: TokenUsage | None = None

        if _should_create_private_plan(messages):
            self._active_private_plans_by_context.pop(context_id, None)
        else:
            private_plan = self._active_private_plans_by_context.get(context_id)
            if private_plan is not None:
                plan_source = "active_plan"
                ctx_logger.debug(
                    "Reusing private Cerebras plan",
                    plan_summary=_summarize_private_plan(private_plan),
                    num_messages=len(messages),
                )
            else:
                plan_source = "fallback_no_active_plan"
                private_plan = _build_fallback_private_plan(messages)
                ctx_logger.warning(
                    "No active private plan for continuation; using executor fallback guidance",
                    num_messages=len(messages),
                    plan_summary=_summarize_private_plan(private_plan),
                )

        for attempt in range(self.malformed_retries + 1):
            try:
                if private_plan is None:
                    planner_prompt = build_planner_prompt(
                        messages=messages,
                        tools=tools,
                        correction=correction,
                    )
                    ctx_logger.debug(
                        "Calling private planner",
                        attempt=attempt + 1,
                        model=self.planner_model,
                        num_messages=len(messages),
                        num_tools=len(tools),
                        prompt_chars=len(planner_prompt),
                        max_completion_tokens=self.planner_max_completion_tokens,
                        reasoning_effort=self.planner_reasoning_effort,
                    )
                    plan_result = self.planner_client.generate(
                        model=self.planner_model,
                        messages=[
                            {
                                "role": "system",
                                "content": PLANNER_DEVELOPER_INSTRUCTIONS,
                            },
                            {"role": "user", "content": planner_prompt},
                        ],
                        response_schema=PRIVATE_PLAN_OUTPUT_SCHEMA,
                        response_schema_name="private_plan",
                        max_completion_tokens=self.planner_max_completion_tokens,
                        temperature=self.planner_temperature,
                        reasoning_effort=self.planner_reasoning_effort,
                    )
                    last_model_stage = "planner"
                    last_model_text = plan_result.text
                    last_finish_reason = plan_result.finish_reason
                    last_token_usage = plan_result.token_usage
                    internal_call_count += 1
                    total_duration_ms += plan_result.duration_ms
                    total_cost += plan_result.cost
                    total_quota_wait_ms += plan_result.quota_wait_ms
                    total_token_usage = add_token_usage(
                        total_token_usage,
                        plan_result.token_usage,
                    )
                    private_plan = parse_private_plan(plan_result.text)
                    planner_ms = plan_result.duration_ms
                    self._active_private_plans_by_context[context_id] = private_plan
                    ctx_logger.debug(
                        "Parsed private plan",
                        raw_preview=plan_result.text[:500],
                        plan_summary=_summarize_private_plan(private_plan),
                        planner_model=plan_result.model,
                        planner_reasoning_effort=self.planner_reasoning_effort,
                        planner_finish_reason=plan_result.finish_reason,
                        planner_ms=round(planner_ms, 1),
                        planner_estimated_request_tokens=(
                            plan_result.estimated_request_tokens
                        ),
                        planner_cerebras_rate_limit_headers=(
                            plan_result.rate_limit_headers.as_dict()
                            if plan_result.rate_limit_headers is not None
                            else None
                        ),
                    )

                executor_prompt = build_executor_prompt(
                    messages=messages,
                    tools=tools,
                    private_plan=private_plan,
                    correction=correction,
                )
                ctx_logger.debug(
                    "Calling Cerebras executor",
                    attempt=attempt + 1,
                    model=self.executor_model,
                    plan_source=plan_source,
                    planner_called=planner_ms > 0,
                    prompt_chars=len(executor_prompt),
                    max_completion_tokens=self.executor_max_completion_tokens,
                    reasoning_effort=self.executor_reasoning_effort,
                )
                executor_result = self.client.generate(
                    model=self.executor_model,
                    messages=[
                        {
                            "role": "system",
                            "content": EXECUTOR_DEVELOPER_INSTRUCTIONS,
                        },
                        {"role": "user", "content": executor_prompt},
                    ],
                    response_schema=NEXT_ACTION_OUTPUT_SCHEMA,
                    response_schema_name="next_action",
                    max_completion_tokens=self.executor_max_completion_tokens,
                    temperature=self.executor_temperature,
                    reasoning_effort=self.executor_reasoning_effort,
                )
                last_model_stage = "executor"
                last_model_text = executor_result.text
                last_finish_reason = executor_result.finish_reason
                last_token_usage = executor_result.token_usage
                internal_call_count += 1
                total_duration_ms += executor_result.duration_ms
                total_cost += executor_result.cost
                total_token_usage = add_token_usage(
                    total_token_usage,
                    executor_result.token_usage,
                )
                total_quota_wait_ms += executor_result.quota_wait_ms
                parsed = parse_next_action(executor_result.text)
                if parsed["action"] == "respond":
                    self._active_private_plans_by_context.pop(context_id, None)
                else:
                    self._active_private_plans_by_context[context_id] = private_plan
                self._last_internal_call_count = internal_call_count
                ctx_logger.info(
                    "Planner/executor response received",
                    action=parsed["action"],
                    num_tool_calls=len(parsed.get("tool_calls") or []),
                    plan_source=plan_source,
                    planner_called=planner_ms > 0,
                    planner_model=self.planner_model,
                    executor_model=executor_result.model,
                    planner_reasoning_effort=self.planner_reasoning_effort,
                    executor_reasoning_effort=self.executor_reasoning_effort,
                    executor_finish_reason=executor_result.finish_reason,
                    planner_ms=round(planner_ms, 1),
                    executor_ms=round(executor_result.duration_ms, 1),
                    total_inference_ms=round(total_duration_ms, 1),
                    executor_estimated_request_tokens=(
                        executor_result.estimated_request_tokens
                    ),
                    executor_cerebras_rate_limit_headers=(
                        executor_result.rate_limit_headers.as_dict()
                        if executor_result.rate_limit_headers is not None
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
                    quota_wait_ms=round(total_quota_wait_ms, 1),
                )
                return AgentInferenceResult(
                    next_action=parsed,
                    elapsed_ms=total_duration_ms,
                    token_usage=total_token_usage,
                    cost=total_cost,
                    internal_calls=max(internal_call_count, 1),
                    quota_wait_ms=total_quota_wait_ms,
                )
            except (MalformedModelResponseError, json.JSONDecodeError) as exc:
                last_error = exc
                self._last_internal_call_count = max(internal_call_count, 1)
                correction = (
                    "The previous planner/executor output was invalid. Return "
                    f"strict JSON matching the requested schema. Error: {exc}"
                )
                warning_message = (
                    "Malformed planner/executor response"
                    f" | stage={last_model_stage}"
                    f" | attempt={attempt + 1}"
                    f" | retrying={attempt < self.malformed_retries}"
                    f" | finish_reason={last_finish_reason or 'unknown'}"
                    f" | output_chars={len(last_model_text)}"
                    f" | error={exc}"
                )
                ctx_logger.warning(
                    warning_message,
                    attempt=attempt + 1,
                    retrying=attempt < self.malformed_retries,
                    plan_source=plan_source,
                    stage=last_model_stage,
                    finish_reason=last_finish_reason,
                    output_chars=len(last_model_text),
                    token_usage=(
                        last_token_usage.__dict__
                        if last_token_usage is not None
                        else None
                    ),
                    error=str(exc),
                )
                ctx_logger.debug(
                    "Malformed planner/executor raw output preview",
                    stage=last_model_stage,
                    finish_reason=last_finish_reason,
                    raw_preview=last_model_text[:1000],
                )
            except CerebrasTemplateError:
                raise

        raise MalformedModelResponseError(
            "Planner/executor did not produce a valid next-action JSON "
            f"object: {last_error}"
        )

    def _record_turn_metrics(
        self,
        context_id: str,
        elapsed_ms: float,
        *,
        token_usage: TokenUsage | None = None,
        cost: float = 0.0,
        internal_calls: int | None = None,
        quota_wait_ms: float = 0.0,
    ) -> None:
        internal_calls = max(
            internal_calls
            if internal_calls is not None
            else self._last_internal_call_count,
            0,
        )
        metrics = self.ctx_id_to_turn_metrics.setdefault(
            context_id,
            {
                PROMPT_TOKENS: 0,
                COMPLETION_TOKENS: 0,
                COST: 0.0,
                MODEL: f"{self.planner_model}->{self.executor_model}",
                THINKING_TOKENS: 0,
                NUM_LLM_CALLS: 0,
                NUM_PASSES: 1,
                QUOTA_WAIT_TIME_MS: 0.0,
                "_total_llm_time_ms": 0.0,
            },
        )
        metrics[NUM_LLM_CALLS] += internal_calls
        if token_usage is not None:
            metrics[PROMPT_TOKENS] += token_usage.input_tokens
            metrics[COMPLETION_TOKENS] += token_usage.output_tokens
            metrics[THINKING_TOKENS] += token_usage.reasoning_output_tokens
        metrics[COST] += cost
        metrics["_total_llm_time_ms"] += elapsed_ms
        metrics[QUOTA_WAIT_TIME_MS] += quota_wait_ms
        metrics[AVG_LLM_CALL_TIME_MS] = (
            round(metrics["_total_llm_time_ms"] / metrics[NUM_LLM_CALLS], 1)
            if metrics[NUM_LLM_CALLS]
            else 0.0
        )
        metrics[NUM_PASSES] = max(metrics.get(NUM_PASSES, 1), max(internal_calls, 1))


def build_planner_prompt(
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    correction: str | None = None,
) -> str:
    prompt_tools = compact_tools_for_prompt(tools)
    planning_tool = _find_tool(prompt_tools, "planning_tool")
    prompt = {
        "task": (
            "Create a private plan for the current user request. The executor "
            "will reuse this plan across subsequent tool-result turns until it "
            "can respond to the user."
        ),
        "available_tools": prompt_tools,
        "planning_tool_schema": planning_tool,
        "conversation_transcript": _messages_for_private_prompt(messages),
        "rules": [
            "Use the planning_tool-shaped JSON contract as internal reasoning.",
            "Do not ask the evaluator to execute planning_tool from this planner step.",
            "Do not invent observations; only actual tool results in the transcript are observations.",
            "Plan the full path from the latest user request through likely tool calls and final response.",
            "Return at least one planning_tool step. Never return an empty steps array.",
            "Include enough guidance for the executor to continue after tool observations.",
            "The final plan step should verify whether all user intents can be resolved before responding.",
            "Keep the plan compact so the executor can use it quickly.",
        ],
    }
    if correction:
        prompt["correction"] = correction
    return json.dumps(prompt, ensure_ascii=False, indent=2)


def build_executor_prompt(
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    private_plan: dict[str, Any],
    correction: str | None = None,
) -> str:
    payload = json.loads(
        build_next_action_prompt(
            messages=messages,
            tools=tools,
            correction=correction,
        )
    )
    payload["private_plan"] = private_plan
    payload["private_plan_rules"] = [
        "The private_plan is internal guidance, not a tool result.",
        "This plan was created after the latest user message and may be reused across tool-result turns.",
        "Do not mention the plan to the user.",
        "Do not wait for private replanning after tool results; continue executing from the transcript and private_plan.",
        "If the private_plan is insufficient and planning_tool is available, you may call planning_tool as a normal benchmark-visible tool call.",
        "Return exactly one final next-action JSON object.",
        "Use only available_tools for any returned tool call.",
    ]
    return json.dumps(payload, ensure_ascii=False, indent=2)


def parse_private_plan(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise MalformedModelResponseError(
                f"No private plan JSON found in: {text[:200]}"
            )
        payload = json.loads(text[start : end + 1])

    planning_tool = payload.get("planning_tool")
    if not isinstance(planning_tool, dict):
        raise MalformedModelResponseError("private plan requires planning_tool object")
    if planning_tool.get("command") != "create":
        raise MalformedModelResponseError(
            "private planning_tool command must be create"
        )
    steps = planning_tool.get("steps")
    if not isinstance(steps, list) or not steps:
        raise MalformedModelResponseError(
            "private planning_tool requires non-empty steps"
        )
    for step in steps:
        if not isinstance(step, dict):
            raise MalformedModelResponseError(
                "each private plan step must be an object"
            )
        if not isinstance(step.get("step_description"), str):
            raise MalformedModelResponseError(
                "private plan steps require step_description"
            )
        dependencies = step.get("step_dependent_on")
        if not isinstance(dependencies, list) or not all(
            isinstance(item, int) for item in dependencies
        ):
            raise MalformedModelResponseError(
                "private plan steps require integer step_dependent_on list"
            )

    return payload


def _find_tool(tools: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    for tool in tools:
        if tool.get("function", {}).get("name") == name:
            return tool
    return None


def _messages_for_private_prompt(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return json.loads(
        build_next_action_prompt(messages=messages, tools=[], correction=None)
    )["conversation_transcript"]


def _should_create_private_plan(messages: list[dict[str, Any]]) -> bool:
    return bool(messages) and messages[-1].get("role") == "user"


def _build_fallback_private_plan(messages: list[dict[str, Any]]) -> dict[str, Any]:
    latest_tool_names = [
        str(message.get("name"))
        for message in messages
        if message.get("role") == "tool" and message.get("name")
    ][-3:]
    observation_note = (
        f" Latest tool observations came from: {', '.join(latest_tool_names)}."
        if latest_tool_names
        else ""
    )
    return {
        "planning_tool": {
            "command": "create",
            "plan_id": "executor_continuation_without_cached_plan",
            "title": "Continue from transcript",
            "steps": [
                {
                    "step_description": (
                        "Review the benchmark-visible transcript, especially "
                        "the latest tool observations."
                    ),
                    "step_dependent_on": [],
                },
                {
                    "step_description": (
                        "If the user goal still needs environment action, call "
                        "only available CAR-bench tools; otherwise respond "
                        "briefly to the user."
                    ),
                    "step_dependent_on": [0],
                },
            ],
        },
        "notes": (
            "No cached private plan was available for this continuation turn. "
            "Continue from transcript evidence only." + observation_note
        ),
        "risk_flags": ["missing_cached_private_plan"],
    }


def _summarize_private_plan(private_plan: dict[str, Any]) -> dict[str, Any]:
    planning_tool = private_plan.get("planning_tool") or {}
    steps = planning_tool.get("steps") or []
    return {
        "title": planning_tool.get("title"),
        "num_steps": len(steps),
        "risk_flags": private_plan.get("risk_flags") or [],
    }


PRIVATE_PLAN_OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["planning_tool", "notes", "risk_flags"],
    "properties": {
        "planning_tool": {
            "type": "object",
            "required": ["command", "plan_id", "title", "steps"],
            "properties": {
                "command": {"type": "string", "enum": ["create"]},
                "plan_id": {"type": "string"},
                "title": {"type": "string"},
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["step_description", "step_dependent_on"],
                        "properties": {
                            "step_description": {"type": "string"},
                            "step_dependent_on": {
                                "type": "array",
                                "items": {"type": "integer"},
                            },
                        },
                        "additionalProperties": False,
                    },
                },
            },
            "additionalProperties": False,
        },
        "notes": {"type": "string"},
        "risk_flags": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "additionalProperties": False,
}


PLANNER_DEVELOPER_INSTRUCTIONS = """You are a private CAR-bench planning layer.
Use the planning_tool-shaped schema as internal reasoning only.
Do not execute tools. Do not answer the user.
Return only JSON matching the requested private plan schema.
Base the plan only on the transcript, supplied tool definitions, and actual tool
results already present in the transcript."""


EXECUTOR_DEVELOPER_INSTRUCTIONS = CEREBRAS_DEVELOPER_INSTRUCTIONS + """
You are the executor in a planner/executor harness.
You may use private_plan as guidance, but it is not a tool result and must not be
mentioned to the user. Return only the final benchmark next-action JSON."""
