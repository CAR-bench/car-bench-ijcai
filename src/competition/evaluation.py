"""Single-unit CAR-bench execution and normalized result records."""

from __future__ import annotations

import math
import random
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Callable


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def execute_orchestrator(orchestrator, env, task_index: int, max_steps: int):
    """Small explicit seam that keeps the scenario max-steps contract testable."""

    return orchestrator.execute(
        env=env,
        task_index=task_index,
        max_num_steps=max_steps,
    )


def run_single_task_trial(
    *,
    args,
    task_index: int,
    task_id: str,
    trial: int,
    trial_seed: int,
    agent_factory: Callable[..., Any],
):
    """Run exactly one official trial without using upstream checkpoint files."""

    from car_bench.envs import get_env
    from car_bench.envs.car_voice_assistant.mock_data import car_va_data_manager
    from car_bench.envs.policy_evaluator import policy_errors_during_runtime
    from car_bench.envs.tool_execution_error_evaluator import (
        tool_execution_errors_during_runtime,
    )
    from car_bench.envs.user.user_end_conversation import end_conversation_failure
    from car_bench.orchestrator import AgentOrchestrator
    from car_bench.types import EnvRunResult
    from run import _init_context_state, _reset_context_state

    random.seed(trial_seed)
    args.seed = int(trial_seed)
    car_va_data_manager.initialize()
    isolated_env = get_env(
        args.env,
        user_strategy=args.user_strategy,
        user_model=args.user_model,
        policy_evaluator_strategy=args.policy_evaluator_strategy,
        policy_evaluator_model=args.policy_evaluator_model,
        task_type=args.task_type,
        task_split=args.task_split,
        user_provider=args.user_model_provider,
        policy_evaluator_provider=args.policy_evaluator_model_provider,
        user_thinking=args.user_thinking,
        task_index=task_index,
        evaluate_policy=args.evaluate_policy,
        score_tool_execution_errors=args.score_tool_execution_errors,
        score_policy_errors=args.score_policy_errors,
        use_user_as_a_tool_tools=args.use_user_as_a_tool_tools,
    )
    actual_task_id = str(isolated_env.tasks[task_index].task_id)
    if actual_task_id != task_id:
        raise RuntimeError(
            f"Hidden task index mismatch: expected {task_id}, got {actual_task_id}"
        )

    local_agent = agent_factory(
        tools_info=isolated_env.tools_info,
        wiki=isolated_env.wiki,
        args=args,
    )
    context_tokens = None
    error_tokens = None
    try:
        context_tokens = _init_context_state(isolated_env, task_index)
        error_tokens = (
            policy_errors_during_runtime.set([]),
            tool_execution_errors_during_runtime.set([]),
            end_conversation_failure.set([]),
        )
        orchestrator = AgentOrchestrator(
            local_agent,
            remove_planning_tools=not args.planning_and_thinking_tool,
        )
        solved = execute_orchestrator(
            orchestrator,
            isolated_env,
            task_index,
            int(args.max_steps),
        )
        result = EnvRunResult(
            task_index=task_index,
            task_id=task_id,
            reward=solved.reward,
            info=solved.info,
            traj=solved.messages,
            trial=int(trial),
        )
    except Exception as exc:
        result = EnvRunResult(
            task_index=task_index,
            task_id=task_id,
            reward=0.0,
            info={"error": str(exc), "traceback": traceback.format_exc()},
            traj=[],
            trial=int(trial),
        )
    finally:
        if context_tokens is not None:
            _reset_context_state(*context_tokens)
        if error_tokens is not None:
            policy_errors_during_runtime.reset(error_tokens[0])
            tool_execution_errors_during_runtime.reset(error_tokens[1])
            end_conversation_failure.reset(error_tokens[2])
    return result


def _number(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) and number >= 0 else 0.0


def _sum_reported(messages: list[dict[str, Any]], field: str) -> int:
    return int(
        sum(
            _number(message.get("turn_metrics", {}).get(field, 0))
            for message in messages
            if isinstance(message.get("turn_metrics"), dict)
        )
    )


def _sum_reported_float(messages: list[dict[str, Any]], field: str) -> float:
    return sum(
        _number(message.get("turn_metrics", {}).get(field, 0))
        for message in messages
        if isinstance(message.get("turn_metrics"), dict)
    )


def _field_fully_reported(messages: list[dict[str, Any]], field: str) -> bool:
    if not messages:
        return False
    for message in messages:
        metrics = message.get("turn_metrics")
        if not isinstance(metrics, dict):
            return False
        reported_fields = metrics.get("_reported_fields")
        if isinstance(reported_fields, list):
            if field not in reported_fields:
                return False
        elif field not in metrics:
            return False
        value = metrics.get(field)
        if value is None or isinstance(value, bool):
            return False
        try:
            number = float(value)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(number) or number < 0:
            return False
    return True


def normalize_unit_record(
    *,
    result,
    dataset_fingerprint: str,
    category: str,
    task_id: str,
    trial: int,
    trial_seed: int,
    attempt_count: int,
    started_at: str,
    completed_at: str,
    duration_seconds: float,
) -> dict[str, Any]:
    """Convert one EnvRunResult into the private schema-v2 unit record."""

    payload = result.model_dump(mode="json")
    trajectory = [
        message
        for message in payload.get("traj", [])
        if isinstance(message, dict) and message.get("role") != "system"
    ]
    assistant_messages = [
        message for message in trajectory if message.get("role") == "assistant"
    ]
    telemetry_expected_messages = [
        message for message in assistant_messages if not message.get("tool_calls")
    ]
    reported_messages = [
        message
        for message in assistant_messages
        if isinstance(message.get("turn_metrics"), dict)
    ]

    raw_turns: list[float] = []
    effective_turns: list[float] = []
    for message in assistant_messages:
        evaluator_metrics = message.get("evaluator_metrics")
        if not isinstance(evaluator_metrics, dict):
            continue
        raw = evaluator_metrics.get("a2a_turn_time_ms")
        effective = evaluator_metrics.get("a2a_effective_turn_time_ms")
        if isinstance(raw, (int, float)):
            raw_turns.append(float(raw))
        if isinstance(effective, (int, float)):
            effective_turns.append(float(effective))

    telemetry_reported = bool(reported_messages)
    telemetry_complete = bool(telemetry_expected_messages) and (
        len(reported_messages) == len(telemetry_expected_messages)
    )
    token_presence = {
        field: telemetry_complete
        and _field_fully_reported(reported_messages, field)
        for field in ("prompt_tokens", "completion_tokens", "thinking_tokens")
    }
    tokens_fully_reported = all(token_presence.values())
    prompt_tokens = _sum_reported(reported_messages, "prompt_tokens")
    completion_tokens = _sum_reported(reported_messages, "completion_tokens")
    thinking_tokens = _sum_reported(reported_messages, "thinking_tokens")
    quota_wait_reported = telemetry_complete and _field_fully_reported(
        reported_messages, "quota_wait_time_ms"
    )
    quota_wait_ms = _sum_reported_float(reported_messages, "quota_wait_time_ms")
    calls_reported = telemetry_complete and _field_fully_reported(
        reported_messages, "num_llm_calls"
    )
    llm_calls = _sum_reported(reported_messages, "num_llm_calls")
    cost_reported = telemetry_complete and _field_fully_reported(
        reported_messages, "cost"
    )
    cost = _sum_reported_float(reported_messages, "cost")
    llm_latency_reported = calls_reported and _field_fully_reported(
        reported_messages, "avg_llm_call_time_ms"
    )
    models = sorted(
        {
            str(message["turn_metrics"].get("model"))
            for message in reported_messages
            if message["turn_metrics"].get("model")
        }
    )

    info = payload.get("info") or {}
    reward = float(payload.get("reward", 0.0))
    return {
        "schema_version": 2,
        "unit": {
            "dataset_fingerprint": dataset_fingerprint,
            "category": category,
            "task_id": task_id,
            "trial": int(trial),
        },
        "status": "completed",
        "attempt_count": int(attempt_count),
        "trial_seed": int(trial_seed),
        "started_at": started_at,
        "completed_at": completed_at,
        "duration_seconds": round(float(duration_seconds), 6),
        "reward": reward,
        "passed": math.isclose(reward, 1.0, rel_tol=0.0, abs_tol=1e-6),
        "reward_info": info.get("reward_info"),
        "task": info.get("task"),
        "trajectory": trajectory,
        "error": None,
        "timing": {
            "a2a_raw": {
                "source": "evaluator_measured",
                "turns_ms": raw_turns,
                "total_ms": sum(raw_turns),
            },
            "a2a_quota_adjusted": {
                "source": "derived_using_participant_self_reported_quota",
                "turns_ms": effective_turns,
                "total_ms": (
                    sum(effective_turns)
                    if effective_turns and quota_wait_reported
                    else None
                ),
            },
            "llm_self_reported_ms": (
                _number(info.get("total_llm_induced_latency_ms"))
                if llm_latency_reported
                else None
            ),
            "quota_wait_self_reported_ms": (
                quota_wait_ms if quota_wait_reported else None
            ),
        },
        "telemetry": {
            "source": "participant_self_reported",
            "reported": telemetry_reported,
            "complete": telemetry_complete,
            "reported_assistant_turns": len(reported_messages),
            "expected_reported_assistant_turns": len(telemetry_expected_messages),
            "assistant_turns": len(assistant_messages),
            "presence": {
                "llm_latency": llm_latency_reported,
                "quota_wait": quota_wait_reported,
                "num_llm_calls": calls_reported,
                "cost": cost_reported,
            },
            "models": models,
            "num_llm_calls": llm_calls if calls_reported else None,
            "cost": cost if cost_reported else None,
            "tokens": {
                "reported": tokens_fully_reported,
                "presence": token_presence,
                "prompt_tokens": (
                    prompt_tokens if token_presence["prompt_tokens"] else None
                ),
                "completion_tokens": (
                    completion_tokens
                    if token_presence["completion_tokens"]
                    else None
                ),
                "thinking_tokens": (
                    thinking_tokens if token_presence["thinking_tokens"] else None
                ),
                "total_tokens": (
                    prompt_tokens + completion_tokens + thinking_tokens
                    if tokens_fully_reported
                    else None
                ),
            },
        },
        "evaluator": {
            "user_cost": info.get("user_cost"),
        },
    }


def timed_single_task_trial(**kwargs):
    started_at = utc_now()
    started = time.perf_counter()
    result = run_single_task_trial(**kwargs)
    return result, started_at, utc_now(), time.perf_counter() - started
