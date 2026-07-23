#!/usr/bin/env python3
"""Generate Docker Compose configuration from a CAR-bench scenario TOML."""

import argparse
import os
import re
import stat
import sys
from pathlib import Path
from typing import Any

try:
    import tomli  # type: ignore[import-not-found]
except ImportError:
    try:
        import tomllib as tomli
    except ImportError:
        print("Error: tomli required. Install with: pip install tomli")
        sys.exit(1)
try:
    import tomli_w
except ImportError:
    print("Error: tomli-w required. Install with: pip install tomli-w")
    sys.exit(1)
COMPOSE_FILENAME = "docker-compose.yml"
A2A_SCENARIO_FILENAME = "a2a-scenario.toml"
ENV_PATH = ".env"
RESULTS_DIR = "output"
PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_PORT = 9009
DEFAULT_ENV_VARS = {"PYTHONUNBUFFERED": "1"}

COMPOSE_TEMPLATE = """# Auto-generated from scenario.toml
# Run from the repository root with:
# docker compose --env-file .env -f {compose_path} up --abort-on-container-exit --exit-code-from a2a-client

services:
  evaluator:{evaluator_build_or_image}
    platform: linux/amd64
    command: {evaluator_command}
    environment:{evaluator_env}{evaluator_volumes}
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:{port}/.well-known/agent-card.json"]
      interval: 5s
      timeout: 3s
      retries: 10
      start_period: 30s
    depends_on:
      agent-under-test:
        condition: service_healthy
    networks:
      - agent-network

  agent-under-test:{agent_under_test_build_or_image}
    platform: linux/amd64
    command: {agent_under_test_command}
    environment:{agent_under_test_env}{agent_under_test_volumes}
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:{port}/.well-known/agent-card.json"]
      interval: 5s
      timeout: 3s
      retries: 10
      start_period: 30s
    networks:
      - agent-network

  a2a-client:
    build:
      context: {client_build_context}
      dockerfile: src/agentbeats/Dockerfile.a2a-client
    platform: linux/amd64
    volumes:
      - ./{a2a_scenario_filename}:/home/carbench/app/scenario.toml:ro
      - {results_mount}:/home/carbench/app/output
    command: ["scenario.toml", "output"]
    depends_on:{client_depends}
    networks:
      - agent-network

networks:
  agent-network:
    driver: bridge
"""

A2A_SCENARIO_TEMPLATE = """[evaluator]
endpoint = "http://evaluator:{port}"

[agent_under_test]
endpoint = "http://agent-under-test:{port}"

{config}"""


def resolve_image(agent: dict[str, Any], name: str) -> None:
    """Validate docker image/build config for a service."""
    has_image = "image" in agent
    has_build = "build" in agent

    if has_image and has_build:
        print(
            f"Error: {name} has multiple deployment methods; use only one of 'image' or 'build'."
        )
        sys.exit(1)
    elif has_image:
        print(f"Using {name} image: {agent['image']}")
    elif has_build:
        build_info = agent["build"]
        if isinstance(build_info, dict):
            print(
                f"Using {name} build: {build_info.get('dockerfile', 'Dockerfile')} in {build_info.get('context', '.')}"
            )
        else:
            print(f"Using {name} build: {build_info}")
    else:
        print(f"Error: {name} must have either 'image' or 'build' field")
        sys.exit(1)


def parse_scenario(scenario_path: Path) -> dict[str, Any]:
    toml_data = scenario_path.read_text()
    data = tomli.loads(toml_data)

    if "green_agent" in data or "participants" in data:
        print(
            "Error: old scenario shape is unsupported; use [evaluator] and [agent_under_test]."
        )
        sys.exit(1)
    evaluator = data.get("evaluator")
    if not isinstance(evaluator, dict):
        print("Error: scenario requires an [evaluator] table.")
        sys.exit(1)
    agent_under_test = data.get("agent_under_test")
    if not isinstance(agent_under_test, dict):
        print("Error: scenario requires an [agent_under_test] table.")
        sys.exit(1)

    resolve_image(evaluator, "evaluator")
    resolve_image(agent_under_test, "agent_under_test")

    data["_scenario_path"] = str(scenario_path)
    return data


def _relative_to_output_dir(path: Path, output_dir: Path) -> str:
    return os.path.relpath(path.resolve(), output_dir.resolve())


def _relative_repo_path(path_value: str, output_dir: Path) -> str:
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return _relative_to_output_dir(PROJECT_ROOT / path, output_dir)


def format_build_or_image(agent: dict[str, Any], output_dir: Path) -> str:
    """Format either build or image field for docker-compose."""
    if "build" in agent:
        build_config = agent["build"]
        if isinstance(build_config, str):
            return f"\n    build: {_relative_repo_path(build_config, output_dir)}"
        else:
            # Handle build object with context and dockerfile
            context = str(build_config.get("context", "."))
            lines = ["\n    build:"]
            lines.append(f"      context: {_relative_repo_path(context, output_dir)}")
            if "dockerfile" in build_config:
                lines.append(f"      dockerfile: {build_config['dockerfile']}")
            return "\n".join(lines)
    elif "image" in agent:
        return f"\n    image: {agent['image']}"
    else:
        raise ValueError("Agent must have either 'image' or 'build' field")


def format_command(base_args: list[str], command_args: list[str] | None = None) -> str:
    """Format command with optional additional arguments."""
    all_args = base_args + (command_args or [])
    return "[" + ", ".join(f'"{arg}"' for arg in all_args) + "]"


def format_env_vars(env_dict: dict[str, Any]) -> str:
    env_vars = {**DEFAULT_ENV_VARS, **env_dict}
    lines = [f"      - {key}={value}" for key, value in env_vars.items()]
    return "\n" + "\n".join(lines)


def _format_volume(volume: str, output_dir: Path) -> str:
    """Rewrite relative host paths so compose files can live in scenario folders."""
    if ":" not in volume:
        return volume

    host, rest = volume.split(":", 1)
    if (
        not host
        or host.startswith("${")
        or host.startswith("/")
        or host.startswith("~")
    ):
        return volume

    if host.startswith(".") or "/" in host:
        host = _relative_repo_path(host, output_dir)
    return f"{host}:{rest}"


def format_volumes(volumes: list[str] | None, output_dir: Path) -> str:
    """Format optional Docker volumes from scenario TOML."""
    if not volumes:
        return ""
    lines = ["", "    volumes:"]
    lines.extend(f"      - {_format_volume(volume, output_dir)}" for volume in volumes)
    return "\n".join(lines)


def format_depends_on(services: list[str]) -> str:
    lines = []
    for service in services:
        lines.append(f"      {service}:")
        lines.append("        condition: service_healthy")
    return "\n" + "\n".join(lines)


def generate_docker_compose(
    scenario: dict[str, Any],
    *,
    output_dir: Path | None = None,
    results_dir: Path | None = None,
) -> str:
    evaluator = scenario["evaluator"]
    agent_under_test = scenario["agent_under_test"]
    output_dir = output_dir or Path.cwd()
    results_dir = results_dir or Path(RESULTS_DIR)

    return COMPOSE_TEMPLATE.format(
        compose_path=str(output_dir / COMPOSE_FILENAME),
        evaluator_build_or_image=format_build_or_image(evaluator, output_dir),
        agent_under_test_build_or_image=format_build_or_image(
            agent_under_test, output_dir
        ),
        port=DEFAULT_PORT,
        evaluator_command=format_command(
            [
                "--host",
                "0.0.0.0",
                "--port",
                str(DEFAULT_PORT),
                "--card-url",
                f"http://evaluator:{DEFAULT_PORT}",
            ],
            evaluator.get("command_args"),
        ),
        agent_under_test_command=format_command(
            [
                "--host",
                "0.0.0.0",
                "--port",
                str(DEFAULT_PORT),
                "--card-url",
                f"http://agent-under-test:{DEFAULT_PORT}",
            ],
            agent_under_test.get("command_args"),
        ),
        evaluator_env=format_env_vars(evaluator.get("env", {})),
        agent_under_test_env=format_env_vars(agent_under_test.get("env", {})),
        evaluator_volumes=format_volumes(evaluator.get("volumes"), output_dir),
        agent_under_test_volumes=format_volumes(
            agent_under_test.get("volumes"), output_dir
        ),
        client_build_context=_relative_repo_path(".", output_dir),
        a2a_scenario_filename=A2A_SCENARIO_FILENAME,
        results_mount=_relative_repo_path(str(results_dir), output_dir),
        client_depends=format_depends_on(["evaluator", "agent-under-test"]),
    )


def generate_a2a_scenario(scenario: dict[str, Any]) -> str:
    config_section = scenario.get("config", {})
    run_section = {
        "source_scenario": scenario.get("_scenario_path", ""),
        "scenario_name": _scenario_name(scenario),
        "agent_name": _agent_name(scenario),
        "agent_metadata": collect_agent_metadata(scenario.get("agent_under_test", {})),
    }
    config_lines = [
        tomli_w.dumps({"run": run_section}),
        tomli_w.dumps({"config": config_section}),
    ]

    return A2A_SCENARIO_TEMPLATE.format(
        port=DEFAULT_PORT, config="\n".join(config_lines)
    )


def _scenario_name(scenario: dict[str, Any]) -> str:
    path = Path(str(scenario.get("_scenario_path", "")))
    if path.parent.name:
        return f"{path.parent.name}/{path.stem}"
    return path.stem or "scenario"


def _agent_name(scenario: dict[str, Any]) -> str:
    agent = scenario.get("agent_under_test", {})
    if isinstance(agent, dict):
        for key in ("name", "result_label"):
            value = agent.get(key)
            if value:
                return str(value)
    path = Path(str(scenario.get("_scenario_path", "")))
    return path.parent.name or "agent_under_test"


def collect_agent_metadata(agent: dict[str, Any]) -> dict[str, Any]:
    """Keep lightweight model/runtime hints for result filenames and metadata."""
    metadata: dict[str, Any] = {}

    for key in ("name", "result_label", "result_model", "result_reasoning_effort"):
        if key in agent:
            metadata[key] = agent[key]

    for key, value in agent.get("env", {}).items():
        upper = key.upper()
        if upper in {
            "AGENT_LLM",
            "AGENT_REASONING_EFFORT",
            "CODEX_MODEL",
            "CODEX_REASONING_EFFORT",
            "CODEX_PLANNER_MODEL",
            "CODEX_EXECUTOR_MODEL",
            "CODEX_PLANNER_REASONING_EFFORT",
            "CODEX_EXECUTOR_REASONING_EFFORT",
            "PACT_COMPILER_MODEL",
            "PACT_COMPILER_REASONING_EFFORT",
            "TRACK2_PLANNER_MODEL",
            "TRACK2_EXECUTOR_MODEL",
            "TRACK2_PLANNER_REASONING_EFFORT",
            "TRACK2_EXECUTOR_REASONING_EFFORT",
        }:
            metadata[key] = value

    if "command_args" in agent:
        metadata["command_args"] = agent["command_args"]

    return metadata


def compose_up_command(compose_path: Path, env_path: str = ENV_PATH) -> str:
    """Return the recommended Compose command for generated scenario-local files."""
    return (
        f"docker compose --env-file {env_path} -f {compose_path} up "
        "--abort-on-container-exit --exit-code-from a2a-client"
    )


def prepare_results_mount(results_dir: Path, agent_name: str) -> Path:
    """Create a sticky bind-mount target writable by an arbitrary container UID."""

    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", agent_name).strip("-._")
    agent_dir = results_dir / (slug[:80].strip("-._") or "agent_under_test")
    for directory in (results_dir, agent_dir):
        directory.mkdir(parents=True, exist_ok=True)
        mode = stat.S_IMODE(directory.stat().st_mode)
        directory.chmod(mode | stat.S_IRWXO | stat.S_ISVTX)
    return agent_dir


def generate_env_file(scenario: dict[str, Any]) -> str:
    evaluator = scenario["evaluator"]
    agent_under_test = scenario["agent_under_test"]

    secrets = set()

    # Extract secrets from ${VAR} patterns in env values
    env_var_pattern = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)[^}]*\}")

    for service in (evaluator, agent_under_test):
        for value in service.get("env", {}).values():
            for match in env_var_pattern.findall(str(value)):
                secrets.add(match)
        for value in service.get("volumes", []):
            for match in env_var_pattern.findall(str(value)):
                secrets.add(match)

    if not secrets:
        return ""

    lines = []
    for secret in sorted(secrets):
        lines.append(f"{secret}=")

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="Generate Docker Compose from scenario.toml"
    )
    parser.add_argument(
        "--scenario", type=Path, required=True, help="Path to scenario.toml file"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated docker-compose.yml and a2a-scenario.toml (default: scenario folder).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(RESULTS_DIR),
        help="Host directory mounted for timestamped results (default: output).",
    )
    args = parser.parse_args()

    if not args.scenario.exists():
        print(f"Error: {args.scenario} not found")
        sys.exit(1)

    scenario = parse_scenario(args.scenario)
    output_dir = args.output_dir or args.scenario.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    prepare_results_mount(args.results_dir, _agent_name(scenario))
    compose_path = output_dir / COMPOSE_FILENAME
    a2a_scenario_path = output_dir / A2A_SCENARIO_FILENAME

    with open(compose_path, "w") as f:
        f.write(
            generate_docker_compose(
                scenario,
                output_dir=output_dir,
                results_dir=args.results_dir,
            )
        )

    with open(a2a_scenario_path, "w") as f:
        f.write(generate_a2a_scenario(scenario))

    env_content = generate_env_file(scenario)
    if env_content:
        env_path = Path(ENV_PATH)
        if env_path.exists():
            print(f"✓ {ENV_PATH} already exists; leaving it unchanged")
        else:
            with open(ENV_PATH, "w") as f:
                f.write(env_content)
            print(f"✓ Generated {ENV_PATH}")

    print(f"✓ Generated {compose_path} and {a2a_scenario_path}")
    print("\nNext steps:")
    print("  1. Review/edit .env file with your API keys")
    print(f"  2. {compose_up_command(compose_path)}")
    print(
        f"  3. Check {args.results_dir}/{_agent_name(scenario)}/ for timestamped results"
    )


if __name__ == "__main__":
    main()
