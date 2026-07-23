"""Server entry point for the PACT Track 2 Cerebras agent."""

from __future__ import annotations

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


# Local scenarios and the release image intentionally invoke this file as a
# script.  Import through the package even in that mode so sibling modules can
# retain normal relative imports and behave identically under ``python -m``.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from logging_utils import configure_logger
from track_2_agent_under_test_cerebras.pact_agent import PACTAgentExecutor
from track_2_agent_under_test_cerebras.plan_compiler_backend import (
    CerebrasCompilerSettings,
    create_cerebras_semantic_compiler,
)


logger = configure_logger(role="agent_under_test", context="server")


def _env_or_default(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value


def prepare_agent_card(url: str) -> AgentCard:
    """Create the public A2A description for the PACT runtime."""

    card = AgentCard(
        name="pact_track_2_cerebras",
        description=(
            "Policy-Aware Contract-guided Tool-use agent with locally verified "
            "obligation plans and direct Cerebras inference"
        ),
        version="2.0.0",
        default_input_modes=["text/plain", "application/json"],
        default_output_modes=["text/plain", "application/json"],
    )

    interface = card.supported_interfaces.add()
    interface.url = url
    interface.protocol_binding = "JSONRPC"
    interface.protocol_version = "1.0"

    card.capabilities.streaming = False
    card.capabilities.push_notifications = False
    card.capabilities.extended_agent_card = False

    skill = card.skills.add()
    skill.id = "contract_guided_assistance"
    skill.name = "Contract-Guided Tool Assistance"
    skill.description = (
        "Compiles requests into typed plans, verifies them against live tool "
        "contracts, and grounds final responses in execution evidence"
    )
    skill.tags.extend(["pact", "tool-use", "a2a", "cerebras"])

    return card


def main() -> None:
    """Start the non-streaming A2A server used by local and GHCR runs."""

    parser = argparse.ArgumentParser(
        description="Run the PACT Track 2 Cerebras agent under test."
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--card-url", type=str)
    args = parser.parse_args()

    if not _env_or_default("CEREBRAS_API_KEY"):
        raise SystemExit("CEREBRAS_API_KEY must be set for Track 2 Cerebras runs.")

    # Resolve settings once.  The same immutable snapshot configures both the
    # provider backend and the model label reported in A2A turn metrics.
    settings = CerebrasCompilerSettings.from_env()
    compiler = create_cerebras_semantic_compiler(
        settings=settings,
        logger=logger.bind(context="compiler"),
    )

    logger.info(
        "Starting PACT Track 2 agent",
        compiler_model=settings.model,
        service_tier=settings.service_tier,
        temperature=settings.temperature,
        reasoning_effort=settings.reasoning_effort,
        max_completion_tokens=settings.max_completion_tokens,
        max_repair_attempts=settings.max_repair_attempts,
        host=args.host,
        port=args.port,
    )

    card = prepare_agent_card(args.card_url or f"http://{args.host}:{args.port}/")
    request_handler = DefaultRequestHandler(
        agent_executor=PACTAgentExecutor(
            compiler=compiler,
            model=settings.model,
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
