"""Environment-configured Cerebras backend for the PACT semantic compiler."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping

from carbench_agent_core.semantic_compiler import (
    CompilerInputError,
    CompilerTokenUsage,
    ModelCandidate,
    SemanticCompiler,
    SemanticCompilerLimits,
)

from .cerebras_client import (
    DEFAULT_CEREBRAS_API_BASE,
    DEFAULT_EXECUTOR_MODEL,
    CerebrasCompletionClient,
)


class CompilerConfigurationError(CompilerInputError):
    """An environment-provided compiler setting is invalid."""


@dataclass(frozen=True)
class CerebrasCompilerSettings:
    """Complete inference and verification configuration for plan compilation."""

    model: str
    api_base: str
    service_tier: str | None
    max_completion_tokens: int
    temperature: float | None
    reasoning_effort: str | None
    semantic_review: bool
    max_repair_attempts: int
    max_nodes: int
    max_dependency_depth: int
    max_policy_chars: int
    max_goal_chars: int
    max_user_event_chars: int
    max_conversation_messages: int
    max_conversation_chars: int
    max_context_chars: int
    max_candidate_chars: int
    max_evidence_records: int

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> "CerebrasCompilerSettings":
        """Read every provider and compiler control from named environment vars."""

        env = os.environ if environ is None else environ
        settings = cls(
            model=_env_text(
                env,
                "PACT_COMPILER_MODEL",
                default=DEFAULT_EXECUTOR_MODEL,
            ),
            api_base=_env_text(
                env,
                "PACT_COMPILER_CEREBRAS_API_BASE",
                default=DEFAULT_CEREBRAS_API_BASE,
            ),
            service_tier=_env_optional_text(
                env,
                "PACT_COMPILER_SERVICE_TIER",
            ),
            max_completion_tokens=_env_int(
                env,
                "PACT_COMPILER_MAX_COMPLETION_TOKENS",
                default=8192,
            ),
            temperature=_env_optional_float(
                env,
                "PACT_COMPILER_TEMPERATURE",
            ),
            reasoning_effort=_env_optional_text(
                env,
                "PACT_COMPILER_REASONING_EFFORT",
                default="medium",
            ),
            semantic_review=_env_bool(
                env,
                "PACT_COMPILER_SEMANTIC_REVIEW",
                default=True,
            ),
            max_repair_attempts=_env_int(
                env,
                "PACT_COMPILER_MAX_REPAIR_ATTEMPTS",
                default=1,
            ),
            max_nodes=_env_int(
                env,
                "PACT_COMPILER_MAX_NODES",
                default=20,
            ),
            max_dependency_depth=_env_int(
                env,
                "PACT_COMPILER_MAX_DEPENDENCY_DEPTH",
                default=12,
            ),
            max_policy_chars=_env_int(
                env,
                "PACT_COMPILER_MAX_POLICY_CHARS",
                default=64_000,
            ),
            max_goal_chars=_env_int(
                env,
                "PACT_COMPILER_MAX_GOAL_CHARS",
                default=12_000,
            ),
            max_user_event_chars=_env_int(
                env,
                "PACT_COMPILER_MAX_USER_EVENT_CHARS",
                default=12_000,
            ),
            max_conversation_messages=_env_int(
                env,
                "PACT_COMPILER_MAX_CONVERSATION_MESSAGES",
                default=32,
            ),
            max_conversation_chars=_env_int(
                env,
                "PACT_COMPILER_MAX_CONVERSATION_CHARS",
                default=64_000,
            ),
            max_context_chars=_env_int(
                env,
                "PACT_COMPILER_MAX_CONTEXT_CHARS",
                default=96_000,
            ),
            max_candidate_chars=_env_int(
                env,
                "PACT_COMPILER_MAX_CANDIDATE_CHARS",
                default=64_000,
            ),
            max_evidence_records=_env_int(
                env,
                "PACT_COMPILER_MAX_EVIDENCE_RECORDS",
                default=128,
            ),
        )
        settings._validate()
        return settings

    def limits(self) -> SemanticCompilerLimits:
        """Translate environment settings into trusted local resource bounds."""

        return SemanticCompilerLimits(
            max_repair_attempts=self.max_repair_attempts,
            max_nodes=self.max_nodes,
            max_dependency_depth=self.max_dependency_depth,
            max_policy_chars=self.max_policy_chars,
            max_goal_chars=self.max_goal_chars,
            max_user_event_chars=self.max_user_event_chars,
            max_conversation_messages=self.max_conversation_messages,
            max_conversation_chars=self.max_conversation_chars,
            max_context_chars=self.max_context_chars,
            max_candidate_chars=self.max_candidate_chars,
            max_evidence_records=self.max_evidence_records,
        )

    def _validate(self) -> None:
        if not self.model:
            raise CompilerConfigurationError("PACT_COMPILER_MODEL cannot be empty")
        if not self.api_base:
            raise CompilerConfigurationError(
                "PACT_COMPILER_CEREBRAS_API_BASE cannot be empty"
            )
        if self.max_completion_tokens <= 0:
            raise CompilerConfigurationError(
                "PACT_COMPILER_MAX_COMPLETION_TOKENS must be positive"
            )
        if self.temperature is not None and not 0 <= self.temperature <= 2:
            raise CompilerConfigurationError(
                "PACT_COMPILER_TEMPERATURE must be between 0 and 2"
            )
        if self.reasoning_effort not in {None, "low", "medium", "high"}:
            raise CompilerConfigurationError(
                "PACT_COMPILER_REASONING_EFFORT must be low, medium, high, or empty"
            )
        try:
            self.limits()
        except ValueError as exc:
            raise CompilerConfigurationError(str(exc)) from exc


class CerebrasStructuredPlanBackend:
    """Adapt the shared Cerebras SDK wrapper to the compiler backend protocol."""

    def __init__(
        self,
        *,
        settings: CerebrasCompilerSettings | None = None,
        client: CerebrasCompletionClient | Any | None = None,
        logger: Any | None = None,
    ) -> None:
        self.settings = settings or CerebrasCompilerSettings.from_env()
        self.client = client or CerebrasCompletionClient(
            api_base=self.settings.api_base,
            service_tier=self.settings.service_tier,
            logger=logger,
        )

    def generate(
        self,
        *,
        messages: list[dict[str, str]],
        response_schema: dict[str, Any],
        response_schema_name: str,
    ) -> ModelCandidate:
        """Request one strict JSON candidate without interpreting its semantics."""

        result = self.client.generate(
            model=self.settings.model,
            messages=messages,
            response_schema=response_schema,
            response_schema_name=response_schema_name,
            max_completion_tokens=self.settings.max_completion_tokens,
            temperature=self.settings.temperature,
            reasoning_effort=self.settings.reasoning_effort,
        )
        provider_usage = result.token_usage
        usage = CompilerTokenUsage(
            prompt_tokens=(provider_usage.input_tokens if provider_usage else 0),
            # Cerebras includes reasoning tokens in completion_tokens.  A2A
            # reports mutually exclusive visible-completion and thinking
            # fields so the evaluator's sum does not double count reasoning.
            completion_tokens=(
                max(
                    0,
                    provider_usage.output_tokens
                    - provider_usage.reasoning_output_tokens,
                )
                if provider_usage
                else 0
            ),
            thinking_tokens=(
                provider_usage.reasoning_output_tokens if provider_usage else 0
            ),
        )
        return ModelCandidate(
            text=result.text,
            model=result.model,
            finish_reason=result.finish_reason,
            duration_ms=result.duration_ms,
            cost=result.cost,
            quota_wait_ms=result.quota_wait_ms,
            usage=usage,
        )


def create_cerebras_semantic_compiler(
    *,
    settings: CerebrasCompilerSettings | None = None,
    client: CerebrasCompletionClient | Any | None = None,
    logger: Any | None = None,
) -> SemanticCompiler:
    """Construct the compiler and its trusted limits from one env snapshot."""

    resolved = settings or CerebrasCompilerSettings.from_env()
    backend = CerebrasStructuredPlanBackend(
        settings=resolved,
        client=client,
        logger=logger,
    )
    return SemanticCompiler(
        backend,
        limits=resolved.limits(),
        semantic_review=resolved.semantic_review,
    )


def _env_text(
    environ: Mapping[str, str],
    name: str,
    *,
    default: str,
) -> str:
    raw = environ.get(name)
    return raw.strip() if raw is not None else default


def _env_optional_text(
    environ: Mapping[str, str],
    name: str,
    *,
    default: str | None = None,
) -> str | None:
    raw = environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip()
    return normalized or None


def _env_int(
    environ: Mapping[str, str],
    name: str,
    *,
    default: int,
) -> int:
    raw = environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        raise CompilerConfigurationError(f"{name} must be an integer") from None


def _env_optional_float(
    environ: Mapping[str, str],
    name: str,
) -> float | None:
    raw = environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        return float(raw)
    except ValueError:
        raise CompilerConfigurationError(f"{name} must be a number") from None


def _env_bool(
    environ: Mapping[str, str],
    name: str,
    *,
    default: bool,
) -> bool:
    raw = environ.get(name)
    if raw is None or not raw.strip():
        return default
    normalized = raw.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise CompilerConfigurationError(f"{name} must be a boolean")


__all__ = [
    "CerebrasCompilerSettings",
    "CerebrasStructuredPlanBackend",
    "CompilerConfigurationError",
    "create_cerebras_semantic_compiler",
]
