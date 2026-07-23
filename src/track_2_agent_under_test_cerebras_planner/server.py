"""Server entry point for the Track 2 Cerebras planner/executor agent."""

import argparse
import os
import sys
from pathlib import Path

import uvicorn
from starlette.applications import Starlette

from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard

if __package__:
    from .planner_agent import (
        DEFAULT_EXECUTOR_MAX_COMPLETION_TOKENS,
        DEFAULT_EXECUTOR_REASONING_EFFORT,
        DEFAULT_PLANNER_MODEL,
        DEFAULT_PLANNER_MAX_COMPLETION_TOKENS,
        DEFAULT_PLANNER_REASONING_EFFORT,
        PlannerExecutorCARBenchAgentExecutor,
    )
else:
    from planner_agent import (
        DEFAULT_EXECUTOR_MAX_COMPLETION_TOKENS,
        DEFAULT_EXECUTOR_REASONING_EFFORT,
        DEFAULT_PLANNER_MODEL,
        DEFAULT_PLANNER_MAX_COMPLETION_TOKENS,
        DEFAULT_PLANNER_REASONING_EFFORT,
        PlannerExecutorCARBenchAgentExecutor,
    )

sys.path.insert(0, str(Path(__file__).parent.parent))
from logging_utils import configure_logger
from track_2_agent_under_test_cerebras.cerebras_client import (
    DEFAULT_CEREBRAS_API_BASE,
    DEFAULT_EXECUTOR_MODEL,
)
sys.path.pop(0)

logger = configure_logger(role="agent_under_test", context="server")


def _env_or_default(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value


def _env_float(name: str, default: float | None = None) -> float | None:
    value = _env_or_default(name)
    if value is None:
        return default
    return float(value)


def _env_int(name: str, default: int) -> int:
    value = _env_or_default(name)
    if value is None:
        return default
    return int(value)


def prepare_agent_card(url: str) -> AgentCard:
    card = AgentCard(
        name="car_bench_agent_cerebras_planner",
        description=(
            "In-car voice assistant using private planning and direct "
            "Cerebras SDK execution"
        ),
        version="1.0.0",
        default_input_modes=["text/plain", "application/json"],
        default_output_modes=["text/plain", "application/json"],
    )

    iface = card.supported_interfaces.add()
    iface.url = url
    iface.protocol_binding = "JSONRPC"
    iface.protocol_version = "1.0"

    card.capabilities.streaming = False
    card.capabilities.push_notifications = False
    card.capabilities.extended_agent_card = False

    skill = card.skills.add()
    skill.id = "car_assistant"
    skill.name = "In-Car Voice Assistant (Cerebras Planner/Executor)"
    skill.description = "Privately plans and returns CAR-bench A2A text or tool calls"
    skill.tags.extend(["benchmark", "car-bench", "voice-assistant", "cerebras"])

    return card


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the CAR-bench Track 2 Cerebras planner/executor agent."
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--card-url", type=str)
    parser.add_argument("--planner-model", type=str, default=None)
    parser.add_argument("--executor-model", type=str, default=None)
    parser.add_argument("--api-base", type=str, default=None)
    parser.add_argument("--service-tier", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--planner-temperature", type=float, default=None)
    parser.add_argument("--executor-temperature", type=float, default=None)
    parser.add_argument("--planner-reasoning-effort", type=str, default=None)
    parser.add_argument("--executor-reasoning-effort", type=str, default=None)
    parser.add_argument("--reasoning-effort", type=str, default=None)
    parser.add_argument("--planner-max-completion-tokens", type=int, default=None)
    parser.add_argument("--executor-max-completion-tokens", type=int, default=None)
    parser.add_argument("--malformed-retries", type=int, default=None)
    args = parser.parse_args()

    if not _env_or_default("CEREBRAS_API_KEY"):
        raise SystemExit("CEREBRAS_API_KEY must be set for Track 2 Cerebras runs.")

    planner_model = (
        args.planner_model
        if args.planner_model is not None
        else _env_or_default("TRACK2_PLANNER_MODEL", DEFAULT_PLANNER_MODEL)
    )
    planner_reasoning_effort = (
        args.planner_reasoning_effort
        if args.planner_reasoning_effort is not None
        else _env_or_default(
            "TRACK2_PLANNER_REASONING_EFFORT",
            DEFAULT_PLANNER_REASONING_EFFORT,
        )
    )

    executor_model = (
        args.executor_model
        if args.executor_model is not None
        else _env_or_default("TRACK2_EXECUTOR_MODEL", DEFAULT_EXECUTOR_MODEL)
    )
    api_base = (
        args.api_base
        if args.api_base is not None
        else _env_or_default("TRACK2_CEREBRAS_API_BASE", DEFAULT_CEREBRAS_API_BASE)
    )
    service_tier = (
        args.service_tier
        if args.service_tier is not None
        else _env_or_default("TRACK2_CEREBRAS_SERVICE_TIER")
    )
    planner_temperature = (
        args.planner_temperature
        if args.planner_temperature is not None
        else _env_float("TRACK2_PLANNER_TEMPERATURE")
    )
    executor_temperature = (
        args.executor_temperature
        if args.executor_temperature is not None
        else (
            args.temperature
            if args.temperature is not None
            else _env_float("TRACK2_TEMPERATURE")
        )
    )
    executor_reasoning_effort = (
        args.executor_reasoning_effort
        if args.executor_reasoning_effort is not None
        else (
            args.reasoning_effort
            if args.reasoning_effort is not None
            else _env_or_default(
                "TRACK2_EXECUTOR_REASONING_EFFORT",
                DEFAULT_EXECUTOR_REASONING_EFFORT,
            )
        )
    )
    planner_max_completion_tokens = (
        args.planner_max_completion_tokens
        if args.planner_max_completion_tokens is not None
        else _env_int(
            "TRACK2_PLANNER_MAX_COMPLETION_TOKENS",
            DEFAULT_PLANNER_MAX_COMPLETION_TOKENS,
        )
    )
    executor_max_completion_tokens = (
        args.executor_max_completion_tokens
        if args.executor_max_completion_tokens is not None
        else _env_int(
            "TRACK2_MAX_COMPLETION_TOKENS",
            DEFAULT_EXECUTOR_MAX_COMPLETION_TOKENS,
        )
    )
    malformed_retries = (
        args.malformed_retries
        if args.malformed_retries is not None
        else _env_int("TRACK2_LLM_MALFORMED_RETRIES", 1)
    )
    if not 0 <= malformed_retries <= 3:
        parser.error(
            "--malformed-retries / TRACK2_LLM_MALFORMED_RETRIES must be between "
            "0 and 3 so planner/executor baseline steps never exceed five "
            "sequential LLM calls."
        )

    logger.info(
        "Starting CAR-bench agent (Cerebras planner/executor)",
        planner_model=planner_model,
        executor_model=executor_model,
        api_base=api_base,
        service_tier=service_tier,
        planner_temperature=planner_temperature,
        executor_temperature=executor_temperature,
        planner_reasoning_effort=planner_reasoning_effort,
        executor_reasoning_effort=executor_reasoning_effort,
        planner_max_completion_tokens=planner_max_completion_tokens,
        executor_max_completion_tokens=executor_max_completion_tokens,
        malformed_retries=malformed_retries,
        host=args.host,
        port=args.port,
    )

    card = prepare_agent_card(args.card_url or f"http://{args.host}:{args.port}/")

    request_handler = DefaultRequestHandler(
        agent_executor=PlannerExecutorCARBenchAgentExecutor(
            planner_model=planner_model,
            executor_model=executor_model or DEFAULT_EXECUTOR_MODEL,
            planner_max_completion_tokens=planner_max_completion_tokens,
            executor_max_completion_tokens=executor_max_completion_tokens,
            api_base=api_base,
            service_tier=service_tier,
            planner_temperature=planner_temperature,
            executor_temperature=executor_temperature,
            planner_reasoning_effort=planner_reasoning_effort,
            executor_reasoning_effort=executor_reasoning_effort,
            malformed_retries=malformed_retries,
        ),
        task_store=InMemoryTaskStore(),
        agent_card=card,
    )

    routes = create_jsonrpc_routes(request_handler, "/", enable_v0_3_compat=True)
    card_routes = create_agent_card_routes(card)
    app = Starlette(routes=routes + card_routes)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        timeout_keep_alive=1000,
    )


if __name__ == "__main__":
    main()
