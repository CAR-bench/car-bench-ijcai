#!/usr/bin/env python3
"""Render and validate the immutable PACT Track 2 scenario.toml."""

from __future__ import annotations

import argparse
import re
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = {
    "direct": ROOT / "submission/track_2_direct/scenario.toml.template",
}
DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}$")
PLACEHOLDER = "sha256:REPLACE_WITH_PUBLIC_IMAGE_DIGEST"
OFFICIAL_EVALUATOR_IMAGE = "ghcr.io/car-bench/car-bench-evaluator:latest"
ENV_REFERENCE_PATTERN = re.compile(r"\$\{[A-Z_][A-Z0-9_]*(?::[-?][^}]*)?\}")
TASK_COUNT_KEYS = (
    "tasks_base_num_tasks",
    "tasks_hallucination_num_tasks",
    "tasks_disambiguation_num_tasks",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the immutable PACT Track 2 submission scenario TOML."
    )
    parser.add_argument("--variant", choices=sorted(TEMPLATES), required=True)
    parser.add_argument(
        "--image-digest",
        required=True,
        help="Published GHCR digest in sha256:<64 lowercase hex chars> form.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination scenario.toml path. Existing files require --force.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing output file.",
    )
    return parser.parse_args()


def validate_submission(data: dict, *, digest: str) -> None:
    if data["evaluator"].get("image") != OFFICIAL_EVALUATOR_IMAGE:
        raise ValueError("Final submission must use the official evaluator image.")

    image = data["agent_under_test"]["image"]
    if not image.startswith("ghcr.io/"):
        raise ValueError("Agent image must be hosted on GHCR.")
    if not image.endswith(f"@{digest}"):
        raise ValueError("Agent image is not pinned to the requested digest.")

    config = data["config"]
    if config.get("task_split") != "hidden":
        raise ValueError('Final submission must set task_split = "hidden".')
    if config.get("num_trials") != 3:
        raise ValueError("Final submission must set num_trials = 3.")
    if config.get("max_steps") != 50:
        raise ValueError("Final submission must set max_steps = 50.")
    for key in TASK_COUNT_KEYS:
        if config.get(key) != -1:
            raise ValueError(f"Final submission must set {key} = -1.")

    for section in ("evaluator", "agent_under_test"):
        for key, value in data[section].get("env", {}).items():
            if not isinstance(value, str) or not ENV_REFERENCE_PATTERN.fullmatch(value):
                raise ValueError(
                    f"{section}.env.{key} must be an environment-variable reference, "
                    "not a literal value."
                )

    agent_env = data["agent_under_test"].get("env", {})
    required_agent_env = {
        "CEREBRAS_API_KEY",
        "PACT_COMPILER_MODEL",
        "PACT_COMPILER_CEREBRAS_API_BASE",
        "PACT_COMPILER_REASONING_EFFORT",
        "PACT_COMPILER_MAX_COMPLETION_TOKENS",
        "PACT_COMPILER_SEMANTIC_REVIEW",
    }
    missing = sorted(required_agent_env.difference(agent_env))
    if missing:
        raise ValueError(f"Final PACT scenario is missing agent env vars: {missing}")
    legacy_main = sorted(
        key
        for key in agent_env
        if key.startswith("TRACK2_EXECUTOR_") or key.startswith("TRACK2_PLANNER_")
    )
    if legacy_main:
        raise ValueError(
            f"Final PACT scenario contains legacy main env vars: {legacy_main}"
        )


def main() -> None:
    args = parse_args()
    if not DIGEST_PATTERN.fullmatch(args.image_digest):
        raise SystemExit(
            "--image-digest must be sha256: followed by 64 lowercase hex chars."
        )
    if args.output.exists() and not args.force:
        raise SystemExit(
            f"Refusing to overwrite existing file: {args.output}. Use --force."
        )

    template = TEMPLATES[args.variant]
    content = template.read_text()
    if PLACEHOLDER not in content:
        raise SystemExit(f"Template placeholder missing from {template}.")
    rendered = content.replace(PLACEHOLDER, args.image_digest)
    data = tomllib.loads(rendered)
    validate_submission(data, digest=args.image_digest)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)
    print(f"Wrote validated PACT Track 2 scenario: {args.output}")
    print(f"Pinned image: {data['agent_under_test']['image']}")


if __name__ == "__main__":
    main()
