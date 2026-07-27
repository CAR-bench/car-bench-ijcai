"""Validation and immutable provenance for submitted competition scenarios."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import tomllib


OFFICIAL_EVALUATOR_REPOSITORY = "ghcr.io/car-bench/car-bench-evaluator"
DIGEST_IMAGE_RE = re.compile(r"^[a-z0-9][a-z0-9._/-]*@sha256:[0-9a-f]{64}$")
TEAM_ID_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$")

ALLOWED_EVALUATOR_KEYS = {"image", "env"}
ALLOWED_EVALUATOR_ENV = {
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "LOGURU_LEVEL",
}
ALLOWED_AGENT_KEYS = {
    "image",
    "env",
    "command_args",
    "name",
    "result_label",
    "result_model",
    "result_reasoning_effort",
}
OFFICIAL_CONFIG_KEYS = {
    "num_trials",
    "task_split",
    "tasks_base_num_tasks",
    "tasks_hallucination_num_tasks",
    "tasks_disambiguation_num_tasks",
    "max_steps",
    "user_model",
    "user_provider",
    "policy_evaluator_model",
    "policy_evaluator_provider",
}
OFFICIAL_EVALUATOR_MODEL_CONFIG = {
    "user_model": "gemini-3.5-flash",
    "user_provider": "gemini",
    "policy_evaluator_model": "gemini-3.5-flash",
    "policy_evaluator_provider": "gemini",
}


class ScenarioValidationError(ValueError):
    """Raised when a submitted scenario is unsafe or not an official hidden run."""


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _validate_env(env: Any, service: str) -> dict[str, Any]:
    if env is None:
        return {}
    if not isinstance(env, dict):
        raise ScenarioValidationError(f"{service}.env must be a table")
    for key, value in env.items():
        if not isinstance(key, str) or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
            raise ScenarioValidationError(f"Invalid environment variable name: {key!r}")
        if not isinstance(value, (str, int, float, bool)):
            raise ScenarioValidationError(f"{service}.env.{key} must be a scalar")
        upper = key.upper()
        secret_name = re.search(
            r"(?:^|_)(?:KEY|TOKEN|PASSWORD|SECRET)(?:_|$)",
            upper,
        )
        if secret_name:
            if not str(value).startswith("${"):
                raise ScenarioValidationError(
                    f"{service}.env.{key} must reference an environment variable, not store a secret"
                )
    return dict(env)


def validate_team_id(team_id: str) -> str:
    if team_id == "aggregate" or not TEAM_ID_RE.fullmatch(team_id):
        raise ScenarioValidationError(
            "team-id must be a non-reserved lowercase slug of 1-64 letters, "
            "digits, and hyphens"
        )
    return team_id


def load_and_validate_scenario(
    path: str | Path,
    *,
    development: bool = False,
) -> tuple[dict[str, Any], str]:
    scenario_path = Path(path).expanduser().resolve()
    if not scenario_path.is_file():
        raise ScenarioValidationError(f"Scenario file not found: {scenario_path}")
    text = scenario_path.read_text(encoding="utf-8")
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ScenarioValidationError(f"Invalid scenario TOML: {exc}") from exc

    unexpected_top = set(data) - {"evaluator", "agent_under_test", "config"}
    if unexpected_top:
        raise ScenarioValidationError(
            f"Unsupported top-level scenario tables: {sorted(unexpected_top)}"
        )
    evaluator = data.get("evaluator")
    agent = data.get("agent_under_test")
    config = data.get("config")
    if not isinstance(evaluator, dict) or not isinstance(agent, dict) or not isinstance(config, dict):
        raise ScenarioValidationError(
            "Scenario requires [evaluator], [agent_under_test], and [config] tables"
        )

    unexpected = set(evaluator) - ALLOWED_EVALUATOR_KEYS
    if unexpected:
        raise ScenarioValidationError(
            f"Evaluator settings are organizer-controlled; unsupported keys: {sorted(unexpected)}"
        )
    evaluator_image = evaluator.get("image")
    official_digest = re.fullmatch(
        re.escape(OFFICIAL_EVALUATOR_REPOSITORY) + r"@sha256:[0-9a-f]{64}",
        evaluator_image or "",
    )
    if not isinstance(evaluator_image, str) or not (
        evaluator_image == f"{OFFICIAL_EVALUATOR_REPOSITORY}:latest"
        or official_digest
    ):
        raise ScenarioValidationError(
            f"Evaluator image must be {OFFICIAL_EVALUATOR_REPOSITORY}:latest or an official digest"
        )
    evaluator["env"] = _validate_env(evaluator.get("env"), "evaluator")
    unexpected_evaluator_env = set(evaluator["env"]) - ALLOWED_EVALUATOR_ENV
    if unexpected_evaluator_env:
        raise ScenarioValidationError(
            "Unsupported evaluator environment variables: "
            f"{sorted(unexpected_evaluator_env)}"
        )

    unexpected = set(agent) - ALLOWED_AGENT_KEYS
    if unexpected:
        raise ScenarioValidationError(
            f"Unsupported agent-under-test keys: {sorted(unexpected)}"
        )
    agent_image = agent.get("image")
    if not isinstance(agent_image, str) or not DIGEST_IMAGE_RE.fullmatch(agent_image):
        raise ScenarioValidationError(
            "agent_under_test.image must be digest-pinned as repository@sha256:<64 hex>"
        )
    agent["env"] = _validate_env(agent.get("env"), "agent_under_test")
    command_args = agent.get("command_args", [])
    if not isinstance(command_args, list) or not all(isinstance(item, str) for item in command_args):
        raise ScenarioValidationError("agent_under_test.command_args must be an array of strings")

    if not development:
        unexpected_config = set(config) - OFFICIAL_CONFIG_KEYS
        if unexpected_config:
            raise ScenarioValidationError(
                f"Unsupported official evaluator config keys: {sorted(unexpected_config)}"
            )
        for key, value in OFFICIAL_EVALUATOR_MODEL_CONFIG.items():
            config.setdefault(key, value)
        expected = {
            "num_trials": 3,
            "task_split": "hidden",
            "tasks_base_num_tasks": -1,
            "tasks_hallucination_num_tasks": -1,
            "tasks_disambiguation_num_tasks": -1,
            "max_steps": 50,
            **OFFICIAL_EVALUATOR_MODEL_CONFIG,
        }
        differences = {
            key: {"expected": expected_value, "actual": config.get(key)}
            for key, expected_value in expected.items()
            if config.get(key) != expected_value
        }
        if differences:
            raise ScenarioValidationError(
                f"Official hidden config mismatch: {json.dumps(differences, sort_keys=True)}"
            )
    else:
        if config.get("task_split") != "hidden":
            raise ScenarioValidationError("Development competition runs still require task_split='hidden'")
        if int(config.get("num_trials", 0)) < 1:
            raise ScenarioValidationError("num_trials must be at least one")
        if int(config.get("max_steps", 0)) < 1:
            raise ScenarioValidationError("max_steps must be at least one")

    return data, text


def resolve_docker_image(image: str, *, pull: bool = True) -> dict[str, Any]:
    """Pull and resolve an image without persisting its container environment."""

    if pull:
        subprocess.run(
            ["docker", "pull", "--platform", "linux/amd64", image],
            check=True,
        )
    inspect = subprocess.run(
        [
            "docker",
            "image",
            "inspect",
            "--format={{json .Id}}|{{json .RepoDigests}}|"
            "{{json .Created}}|{{json .Config.Labels}}",
            image,
        ],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    image_id_json, repo_digests_json, created_json, labels_json = inspect.split("|", 3)
    image_id = json.loads(image_id_json)
    repo_digests = json.loads(repo_digests_json) or []
    labels = json.loads(labels_json) or {}
    repository = image.split("@", 1)[0].split(":", 1)[0]
    resolved_ref = next(
        (digest for digest in repo_digests if digest.startswith(f"{repository}@sha256:")),
        image if "@sha256:" in image else None,
    )
    if resolved_ref is None:
        raise ScenarioValidationError(f"Could not resolve a repository digest for {image}")
    labelled_version = labels.get("org.opencontainers.image.version")
    return {
        "requested": image,
        "resolved": resolved_ref,
        "image_id": image_id,
        "created": json.loads(created_json),
        "version": labelled_version or resolved_ref.split("@", 1)[1],
        "version_source": "oci_label" if labelled_version else "image_digest",
        "revision": labels.get("org.opencontainers.image.revision"),
    }
