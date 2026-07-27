"""Per-run Docker Compose generation for organizer evaluations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import tomli_w


CONTAINER_PORT = 9009
CONTAINER_DATASET_DIR = "/hidden-dataset"
CONTAINER_RUN_DIR = "/run-state"
CONTROL_TOKEN_REF = "${CAR_BENCH_ORGANIZER_TOKEN:?Organizer control token missing}"


def _yaml_string(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return json.dumps(str(value))


def _environment_lines(values: dict[str, Any], indent: int = 6) -> str:
    spaces = " " * indent
    if not values:
        return f"{spaces}{{}}"
    return "\n".join(f"{spaces}{key}: {_yaml_string(value)}" for key, value in values.items())


def _command(args: list[str]) -> str:
    return json.dumps(args)


def generate_internal_scenario(
    *,
    manifest: dict[str, Any],
    config: dict[str, Any],
) -> str:
    data = {
        "evaluator": {"endpoint": f"http://evaluator:{CONTAINER_PORT}"},
        "agent_under_test": {"endpoint": f"http://agent-under-test:{CONTAINER_PORT}"},
        "competition_run": {
            "schema_version": 2,
            "run_id": manifest["run_id"],
            "team_id": manifest["team_id"],
            "team_name": manifest["team_name"],
            "track": manifest["track"],
            "run_dir": CONTAINER_RUN_DIR,
            "dataset_fingerprint": manifest["dataset"]["fingerprint"],
            "base_seed": manifest["base_seed"],
        },
        "config": config,
    }
    return tomli_w.dumps(data)


def generate_competition_compose(
    *,
    manifest: dict[str, Any],
    scenario: dict[str, Any],
    run_dir: Path,
    dataset_dir: Path,
    project_root: Path,
) -> str:
    evaluator = scenario["evaluator"]
    agent = scenario["agent_under_test"]
    evaluator_env = {
        "PYTHONUNBUFFERED": "1",
        "HOME": "/tmp",
        "UV_CACHE_DIR": "/tmp/uv-cache",
        **evaluator.get("env", {}),
        "CAR_BENCH_HIDDEN_DATA_DIR": CONTAINER_DATASET_DIR,
        "CAR_BENCH_RUN_DIR": CONTAINER_RUN_DIR,
        "CAR_BENCH_ORGANIZER_TOKEN": CONTROL_TOKEN_REF,
    }
    agent_env = {"PYTHONUNBUFFERED": "1", **agent.get("env", {})}
    evaluator_image = manifest["images"]["evaluator"]["resolved"]
    agent_image = manifest["images"]["agent_under_test"]["resolved"]
    agent_command = [
        "--host", "0.0.0.0", "--port", str(CONTAINER_PORT),
        "--card-url", f"http://agent-under-test:{CONTAINER_PORT}",
        *agent.get("command_args", []),
    ]
    evaluator_command = [
        "--host", "0.0.0.0", "--port", str(CONTAINER_PORT),
        "--card-url", f"http://evaluator:{CONTAINER_PORT}",
    ]
    scenario_path = run_dir / "a2a-scenario.toml"
    return f"""# Generated organizer-only competition Compose file.
services:
  evaluator:
    image: {_yaml_string(evaluator_image)}
    platform: linux/amd64
    user: {_yaml_string(f"{manifest['runtime']['host_uid']}:{manifest['runtime']['host_gid']}")}
    command: {_command(evaluator_command)}
    environment:
{_environment_lines(evaluator_env)}
    volumes:
      - type: bind
        source: {_yaml_string(dataset_dir.resolve())}
        target: {CONTAINER_DATASET_DIR}
        read_only: true
      - type: bind
        source: {_yaml_string(run_dir.resolve())}
        target: {CONTAINER_RUN_DIR}
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:{CONTAINER_PORT}/.well-known/agent-card.json"]
      interval: 5s
      timeout: 3s
      retries: 18
      start_period: 30s
    depends_on:
      agent-under-test:
        condition: service_healthy
    networks: [competition-network]

  agent-under-test:
    image: {_yaml_string(agent_image)}
    platform: linux/amd64
    command: {_command(agent_command)}
    environment:
{_environment_lines(agent_env)}
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:{CONTAINER_PORT}/.well-known/agent-card.json"]
      interval: 5s
      timeout: 3s
      retries: 18
      start_period: 30s
    networks: [competition-network]

  a2a-client:
    build:
      context: {_yaml_string(project_root.resolve())}
      dockerfile: src/agentbeats/Dockerfile.a2a-client
    platform: linux/amd64
    environment:
      CAR_BENCH_ORGANIZER_TOKEN: {_yaml_string(CONTROL_TOKEN_REF)}
    volumes:
      - type: bind
        source: {_yaml_string(scenario_path.resolve())}
        target: /home/carbench/app/scenario.toml
        read_only: true
    command: ["scenario.toml"]
    depends_on:
      evaluator:
        condition: service_healthy
      agent-under-test:
        condition: service_healthy
    networks: [competition-network]

networks:
  competition-network:
    driver: bridge
"""
