"""Server entry point for the Track 2 Cerebras CAR-bench agent."""

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
    from .car_bench_agent import CARBenchAgentExecutor
    from .cerebras_client import (
        DEFAULT_CEREBRAS_API_BASE,
        DEFAULT_EXECUTOR_MODEL,
    )
else:
    from car_bench_agent import CARBenchAgentExecutor
    from cerebras_client import (
        DEFAULT_CEREBRAS_API_BASE,
        DEFAULT_EXECUTOR_MODEL,
    )

sys.path.insert(0, str(Path(__file__).parent.parent))
from logging_utils import configure_logger
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
    """Create the agent card for the Cerebras agent under test."""

    card = AgentCard(
        name="car_bench_agent_cerebras",
        description=(
            "In-car voice assistant agent for CAR-bench using direct "
            "Cerebras SDK inference"
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
    skill.name = "In-Car Voice Assistant (Cerebras)"
    skill.description = "Returns CAR-bench text responses or tool calls through A2A"
    skill.tags.extend(["benchmark", "car-bench", "voice-assistant", "cerebras"])

    return card


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the CAR-bench Track 2 Cerebras agent under test."
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--card-url", type=str)
    parser.add_argument("--executor-model", type=str, default=None)
    parser.add_argument("--service-tier", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--reasoning-effort", type=str, default=None)
    parser.add_argument("--executor-reasoning-effort", type=str, default=None)
    parser.add_argument("--max-completion-tokens", type=int, default=None)
    parser.add_argument("--malformed-retries", type=int, default=None)
    args = parser.parse_args()

    if not _env_or_default("CEREBRAS_API_KEY"):
        raise SystemExit("CEREBRAS_API_KEY must be set for Track 2 Cerebras runs.")

    executor_model = (
        args.executor_model
        if args.executor_model is not None
        else _env_or_default("TRACK2_EXECUTOR_MODEL", DEFAULT_EXECUTOR_MODEL)
    )
    service_tier = (
        args.service_tier
        if args.service_tier is not None
        else _env_or_default("TRACK2_CEREBRAS_SERVICE_TIER")
    )
    temperature = (
        args.temperature
        if args.temperature is not None
        else _env_float("TRACK2_TEMPERATURE")
    )
    reasoning_effort = (
        args.executor_reasoning_effort
        if args.executor_reasoning_effort is not None
        else (
            args.reasoning_effort
            if args.reasoning_effort is not None
            else _env_or_default("TRACK2_EXECUTOR_REASONING_EFFORT")
        )
    )
    max_completion_tokens = (
        args.max_completion_tokens
        if args.max_completion_tokens is not None
        else _env_int("TRACK2_MAX_COMPLETION_TOKENS", 1024)
    )
    malformed_retries = (
        args.malformed_retries
        if args.malformed_retries is not None
        else _env_int("TRACK2_LLM_MALFORMED_RETRIES", 1)
    )

    logger.info(
        "Starting CAR-bench agent (Cerebras)",
        executor_model=executor_model,
        service_tier=service_tier,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        max_completion_tokens=max_completion_tokens,
        malformed_retries=malformed_retries,
        host=args.host,
        port=args.port,
    )

    card = prepare_agent_card(args.card_url or f"http://{args.host}:{args.port}/")

    request_handler = DefaultRequestHandler(
        agent_executor=CARBenchAgentExecutor(
            model=executor_model or DEFAULT_EXECUTOR_MODEL,
            api_base=DEFAULT_CEREBRAS_API_BASE,
            service_tier=service_tier,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            max_completion_tokens=max_completion_tokens,
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
