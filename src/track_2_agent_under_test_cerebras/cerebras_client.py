"""Cerebras SDK client with proactive pacing and reactive rate-limit handling."""

from __future__ import annotations

import json
import math
import os
import random
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, overload
from uuid import uuid4


DEFAULT_CEREBRAS_API_BASE = "https://api.cerebras.ai"
DEFAULT_EXECUTOR_MODEL = "gpt-oss-120b"
DEFAULT_EXECUTOR_REASONING_EFFORT = "medium"
DEFAULT_TOKEN_ESTIMATE_CHARS_PER_TOKEN = 4.0
DEFAULT_TOKEN_SAFETY_FACTOR = 1.1
DEFAULT_CEREBRAS_QUEUE_BACKOFF_SECONDS = 60.0
DEFAULT_CEREBRAS_QUEUE_BACKOFF_INITIAL_JITTER_RATIO = 0.1
DEFAULT_CEREBRAS_QUEUE_BACKOFF_SECOND_MIN_SECONDS = 90.0
DEFAULT_CEREBRAS_QUEUE_BACKOFF_SECOND_MAX_SECONDS = 120.0
DEFAULT_CEREBRAS_QUEUE_BACKOFF_CAP_MIN_SECONDS = 180.0
DEFAULT_CEREBRAS_QUEUE_BACKOFF_CAP_MAX_SECONDS = 300.0
DEFAULT_CEREBRAS_RATE_LIMIT_RETRY_BUFFER_SECONDS = 1.0
DEFAULT_CEREBRAS_TOKEN_QUOTA_WINDOW_SECONDS = 60.0
DEFAULT_CEREBRAS_MAX_RATE_LIMIT_RETRIES = 3
DEFAULT_CEREBRAS_MAX_RATE_LIMIT_WAIT_SECONDS = 600.0
CEREBRAS_RATE_LIMIT_HEADER_NAMES = (
    "x-ratelimit-limit-requests-day",
    "x-ratelimit-limit-tokens-minute",
    "x-ratelimit-remaining-requests-day",
    "x-ratelimit-remaining-tokens-minute",
    "x-ratelimit-reset-requests-day",
    "x-ratelimit-reset-tokens-minute",
)


class CerebrasTemplateError(RuntimeError):
    """Raised when a Cerebras-backed template call fails."""


class MalformedModelResponseError(CerebrasTemplateError):
    """Raised when the model output cannot be parsed as the expected JSON."""


class CerebrasRateLimitBudgetExceededError(CerebrasTemplateError):
    """Raised when a bounded rate-limit retry cannot be scheduled safely.

    The exception intentionally contains only local counters and configuration.
    Provider exception text, response bodies, headers, prompts, and credentials are
    excluded so callers may surface or log it without leaking provider data.
    """

    def __init__(
        self,
        *,
        reason: str,
        retries_completed: int,
        max_retries: int,
        cumulative_wait_seconds: float,
        max_wait_seconds: float,
        next_wait_seconds: float,
    ) -> None:
        self.reason = reason
        self.retries_completed = retries_completed
        self.max_retries = max_retries
        self.cumulative_wait_seconds = cumulative_wait_seconds
        self.max_wait_seconds = max_wait_seconds
        self.next_wait_seconds = next_wait_seconds
        super().__init__(
            "Cerebras rate-limit retry budget exhausted "
            f"(reason={reason}, retries_completed={retries_completed}, "
            f"max_retries={max_retries}, "
            f"cumulative_wait_seconds={cumulative_wait_seconds:.3f}, "
            f"max_wait_seconds={max_wait_seconds:.3f}, "
            f"next_wait_seconds={next_wait_seconds:.3f})"
        )


@dataclass
class TokenUsage:
    """Token usage reported by one model call."""

    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_output_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_provider_usage(cls, usage: Any) -> "TokenUsage | None":
        if usage is None:
            return None
        completion_details = _get_field(usage, "completion_tokens_details")
        prompt_details = _get_field(usage, "prompt_tokens_details")
        return cls(
            input_tokens=_safe_int(_get_field(usage, "prompt_tokens")),
            cached_input_tokens=_safe_int(_get_field(prompt_details, "cached_tokens")),
            output_tokens=_safe_int(_get_field(usage, "completion_tokens")),
            reasoning_output_tokens=_safe_int(
                _get_field(completion_details, "reasoning_tokens")
            ),
            total_tokens=_safe_int(_get_field(usage, "total_tokens")),
        )

    def __bool__(self) -> bool:
        return any(
            (
                self.input_tokens,
                self.cached_input_tokens,
                self.output_tokens,
                self.reasoning_output_tokens,
                self.total_tokens,
            )
        )


@dataclass(frozen=True)
class CerebrasRateLimitHeaders:
    """Cerebras rate-limit headers from a provider response."""

    limit_requests_day: float | None = None
    limit_tokens_minute: float | None = None
    remaining_requests_day: float | None = None
    remaining_tokens_minute: float | None = None
    reset_requests_day_seconds: float | None = None
    reset_tokens_minute_seconds: float | None = None
    raw_headers: dict[str, str] | None = None

    @classmethod
    def from_headers(cls, headers: Any) -> "CerebrasRateLimitHeaders | None":
        headers_dict = _headers_dict(headers)
        if not headers_dict:
            return None
        relevant_headers = {
            name: _header_value(headers_dict, name)
            for name in CEREBRAS_RATE_LIMIT_HEADER_NAMES
        }
        if not any(value is not None for value in relevant_headers.values()):
            return None
        return cls(
            limit_requests_day=_safe_float_or_none(
                relevant_headers["x-ratelimit-limit-requests-day"]
            ),
            limit_tokens_minute=_safe_float_or_none(
                relevant_headers["x-ratelimit-limit-tokens-minute"]
            ),
            remaining_requests_day=_safe_float_or_none(
                relevant_headers["x-ratelimit-remaining-requests-day"]
            ),
            remaining_tokens_minute=_safe_float_or_none(
                relevant_headers["x-ratelimit-remaining-tokens-minute"]
            ),
            reset_requests_day_seconds=_safe_float_or_none(
                relevant_headers["x-ratelimit-reset-requests-day"]
            ),
            reset_tokens_minute_seconds=_safe_float_or_none(
                relevant_headers["x-ratelimit-reset-tokens-minute"]
            ),
            raw_headers={
                key: value
                for key, value in relevant_headers.items()
                if value is not None
            },
        )

    def as_dict(self) -> dict[str, Any]:
        """Return parsed numeric quota data without raw provider header text."""

        return {
            "limit_requests_day": self.limit_requests_day,
            "limit_tokens_minute": self.limit_tokens_minute,
            "remaining_requests_day": self.remaining_requests_day,
            "remaining_tokens_minute": self.remaining_tokens_minute,
            "reset_requests_day_seconds": self.reset_requests_day_seconds,
            "reset_tokens_minute_seconds": self.reset_tokens_minute_seconds,
            # Raw header values are provider-controlled and may contain
            # diagnostic text. Parsed finite numbers above are sufficient for
            # pacing and safe persistence.
            "raw_headers": None,
        }


@dataclass
class CompletionCallResult:
    """Final model text, successful call duration, token usage, and provider data."""

    text: str
    duration_ms: float
    model: str
    finish_reason: str | None = None
    token_usage: TokenUsage | None = None
    cost: float = 0.0
    estimated_request_tokens: int = 0
    rate_limit_headers: CerebrasRateLimitHeaders | None = None
    quota_wait_ms: float = 0.0


@dataclass(frozen=True)
class CerebrasRateLimitSignal:
    """Provider-visible rate-limit signal used for reports and reactive retry."""

    code: str | None
    type: str | None
    param: str | None
    message: str | None
    source: str
    retry_after_seconds: float | None = None
    reset_tokens_minute_seconds: float | None = None
    reset_requests_day_seconds: float | None = None
    schedule_wait_seconds: float | None = None
    schedule_reason: str | None = None
    wait_source_header: str | None = None
    missing_expected_headers: tuple[str, ...] = ()
    x_should_retry: str | None = None
    quota_wait_eligible: bool = False
    queue_retry_attempt: int | None = None
    queue_backoff_min_seconds: float | None = None
    queue_backoff_max_seconds: float | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a log-safe projection of a provider-controlled error.

        Provider error messages and arbitrary token-like fields are not audit
        metadata: they can echo request content or credentials.  Scheduling
        uses the original in-memory fields, while persistent reports receive
        only allow-listed diagnostic categories and numeric timing data.
        """

        return {
            "code": _allowlisted_diagnostic(
                self.code,
                {
                    "queue_exceeded",
                    "request_quota_exceeded",
                    "token_quota_exceeded",
                },
            ),
            "type": _allowlisted_diagnostic(
                self.type,
                {
                    "rate_limit_error",
                    "too_many_requests_error",
                    "too_many_tokens_error",
                },
            ),
            "param": _allowlisted_diagnostic(self.param, {"queue", "quota"}),
            "message": None,
            "source": self.source,
            "retry_after_seconds": self.retry_after_seconds,
            "reset_tokens_minute_seconds": self.reset_tokens_minute_seconds,
            "reset_requests_day_seconds": self.reset_requests_day_seconds,
            "schedule_wait_seconds": self.schedule_wait_seconds,
            "schedule_reason": self.schedule_reason,
            "wait_source_header": self.wait_source_header,
            "missing_expected_headers": list(self.missing_expected_headers),
            "x_should_retry": _allowlisted_diagnostic(
                self.x_should_retry,
                {"0", "1", "false", "true"},
            ),
            "quota_wait_eligible": self.quota_wait_eligible,
            "queue_retry_attempt": self.queue_retry_attempt,
            "queue_backoff_min_seconds": self.queue_backoff_min_seconds,
            "queue_backoff_max_seconds": self.queue_backoff_max_seconds,
        }


def add_token_usage(
    left: TokenUsage | None,
    right: TokenUsage | None,
) -> TokenUsage | None:
    """Return the sum of two optional token usage records."""

    if left is None:
        return right
    if right is None:
        return left
    return TokenUsage(
        input_tokens=left.input_tokens + right.input_tokens,
        cached_input_tokens=left.cached_input_tokens + right.cached_input_tokens,
        output_tokens=left.output_tokens + right.output_tokens,
        reasoning_output_tokens=(
            left.reasoning_output_tokens + right.reasoning_output_tokens
        ),
        total_tokens=left.total_tokens + right.total_tokens,
    )


class CerebrasCompletionClient:
    """Synchronous Cerebras SDK completion wrapper with reactive 429 retry."""

    def __init__(
        self,
        *,
        api_base: str | None = None,
        service_tier: str | None = None,
        logger: Any | None = None,
        sdk_client: Any | None = None,
    ) -> None:
        self.api_base = api_base or DEFAULT_CEREBRAS_API_BASE
        self.service_tier = service_tier.strip() if service_tier else None
        self.logger = logger
        self.queue_backoff_seconds = _env_float(
            ("PACT_CEREBRAS_QUEUE_BACKOFF_SECONDS",),
            DEFAULT_CEREBRAS_QUEUE_BACKOFF_SECONDS,
        )
        self.queue_backoff_initial_jitter_ratio = _env_float(
            ("PACT_CEREBRAS_QUEUE_BACKOFF_INITIAL_JITTER_RATIO",),
            DEFAULT_CEREBRAS_QUEUE_BACKOFF_INITIAL_JITTER_RATIO,
        )
        self.queue_backoff_second_min_seconds = _env_float(
            ("PACT_CEREBRAS_QUEUE_BACKOFF_SECOND_MIN_SECONDS",),
            DEFAULT_CEREBRAS_QUEUE_BACKOFF_SECOND_MIN_SECONDS,
        )
        self.queue_backoff_second_max_seconds = _env_float(
            ("PACT_CEREBRAS_QUEUE_BACKOFF_SECOND_MAX_SECONDS",),
            DEFAULT_CEREBRAS_QUEUE_BACKOFF_SECOND_MAX_SECONDS,
        )
        self.queue_backoff_cap_min_seconds = _env_float(
            ("PACT_CEREBRAS_QUEUE_BACKOFF_CAP_MIN_SECONDS",),
            DEFAULT_CEREBRAS_QUEUE_BACKOFF_CAP_MIN_SECONDS,
        )
        self.queue_backoff_cap_max_seconds = _env_float(
            ("PACT_CEREBRAS_QUEUE_BACKOFF_CAP_MAX_SECONDS",),
            DEFAULT_CEREBRAS_QUEUE_BACKOFF_CAP_MAX_SECONDS,
        )
        self.rate_limit_retry_buffer_seconds = _env_float(
            ("PACT_CEREBRAS_RATE_LIMIT_RETRY_BUFFER_SECONDS",),
            DEFAULT_CEREBRAS_RATE_LIMIT_RETRY_BUFFER_SECONDS,
        )
        self.proactive_token_pacing = _env_bool(
            ("PACT_CEREBRAS_PROACTIVE_TOKEN_PACING",),
            default=True,
        )
        self.token_quota_window_seconds = _env_float(
            ("PACT_CEREBRAS_TOKEN_QUOTA_WINDOW_SECONDS",),
            DEFAULT_CEREBRAS_TOKEN_QUOTA_WINDOW_SECONDS,
        )
        self.max_rate_limit_retries = _env_nonnegative_int(
            ("PACT_CEREBRAS_MAX_RATE_LIMIT_RETRIES",),
            DEFAULT_CEREBRAS_MAX_RATE_LIMIT_RETRIES,
        )
        self.max_rate_limit_wait_seconds = _env_nonnegative_float(
            ("PACT_CEREBRAS_MAX_RATE_LIMIT_WAIT_SECONDS",),
            DEFAULT_CEREBRAS_MAX_RATE_LIMIT_WAIT_SECONDS,
        )
        self.rate_limit_report_dir = Path(
            os.getenv(
                "CAR_BENCH_CEREBRAS_RATE_LIMIT_REPORT_DIR",
                os.getenv(
                    "CAR_BENCH_RATE_LIMIT_REPORT_DIR",
                    "/tmp/car-bench-rate-limit-reports",
                ),
            )
        )
        self._sdk_client = sdk_client
        self._session_started_at = datetime.now().astimezone()
        self._session_started_monotonic = time.perf_counter()
        self._request_lock = threading.Lock()
        self._metrics_lock = threading.Lock()
        self._successful_calls = 0
        self._successful_calls_by_model: dict[str, int] = {}
        self._attempted_calls = 0
        self._attempted_calls_by_model: dict[str, int] = {}
        self._total_token_usage = TokenUsage()
        self._token_usage_by_model: dict[str, TokenUsage] = {}
        self._estimated_request_tokens = 0
        self._estimated_request_tokens_by_model: dict[str, int] = {}
        self._last_estimated_request_tokens_by_model: dict[str, int] = {}
        self._last_successful_token_usage_by_model: dict[str, TokenUsage] = {}
        self._last_rate_limit_headers_by_model: dict[
            str,
            tuple[CerebrasRateLimitHeaders, float],
        ] = {}
        self._previous_rate_limit_at: datetime | None = None
        self._previous_rate_limit_retry_at: datetime | None = None
        self._last_rate_limit_total_token_usage = TokenUsage()
        self._last_rate_limit_estimated_request_tokens = 0
        self._last_rate_limit_successful_calls = 0

    def generate(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        response_schema: dict[str, Any] | None,
        response_schema_name: str | None,
        max_completion_tokens: int,
        temperature: float | None,
        reasoning_effort: str | None = None,
    ) -> CompletionCallResult:
        with self._request_lock:
            return self._generate_locked(
                model=model,
                messages=messages,
                response_schema=response_schema,
                response_schema_name=response_schema_name,
                max_completion_tokens=max_completion_tokens,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
            )

    def _generate_locked(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        response_schema: dict[str, Any] | None,
        response_schema_name: str | None,
        max_completion_tokens: int,
        temperature: float | None,
        reasoning_effort: str | None = None,
    ) -> CompletionCallResult:
        normalized_model = normalize_cerebras_model(model)
        normalized_reasoning_effort = _optional_text(reasoning_effort)
        quota_wait_ms = 0.0
        rate_limit_retries = 0
        queue_retries = 0
        cumulative_rate_limit_wait_seconds = 0.0

        for _request_attempt in range(self.max_rate_limit_retries + 1):
            estimated_tokens = estimate_request_tokens(
                messages=messages,
                max_completion_tokens=max_completion_tokens,
            )
            proactive_wait_seconds = self._proactive_quota_wait_seconds(
                model=normalized_model,
                estimated_tokens=estimated_tokens,
            )
            if proactive_wait_seconds > 0:
                self._check_rate_limit_wait_budget(
                    reason="proactive_wait_limit",
                    retries_completed=rate_limit_retries,
                    next_wait_seconds=proactive_wait_seconds,
                    cumulative_wait_seconds=cumulative_rate_limit_wait_seconds,
                    check_retry_limit=False,
                )
                quota_wait_ms += proactive_wait_seconds * 1000.0
                cumulative_rate_limit_wait_seconds += proactive_wait_seconds
                if self.logger:
                    self.logger.info(
                        "Waiting for Cerebras token quota before request",
                        model=normalized_model,
                        estimated_request_tokens=estimated_tokens,
                        wait_seconds=round(proactive_wait_seconds, 3),
                        wait_source="previous_successful_rate_limit_headers",
                    )
                time.sleep(proactive_wait_seconds)
                self._clear_rate_limit_headers(normalized_model)
            previous_request_state = self._record_attempt(
                model=normalized_model,
                estimated_tokens=estimated_tokens,
            )
            kwargs = self._completion_kwargs(
                model=normalized_model,
                messages=messages,
                response_schema=response_schema,
                response_schema_name=response_schema_name,
                max_completion_tokens=max_completion_tokens,
                temperature=temperature,
                reasoning_effort=normalized_reasoning_effort,
            )
            if self.logger:
                self.logger.info(
                    "Sending Cerebras request",
                    model=normalized_model,
                    service_tier=self.service_tier,
                    reasoning_effort=normalized_reasoning_effort,
                    estimated_request_tokens=estimated_tokens,
                    previous_estimated_request_tokens=previous_request_state[
                        "previous_estimated_request_tokens"
                    ],
                    estimated_request_token_delta_since_previous=(
                        previous_request_state[
                            "estimated_request_token_delta_since_previous"
                        ]
                    ),
                    previous_successful_token_usage=previous_request_state[
                        "previous_successful_token_usage"
                    ],
                    previous_rate_limit_headers=previous_request_state[
                        "previous_rate_limit_headers"
                    ],
                    max_completion_tokens=max_completion_tokens,
                    has_output_schema=response_schema is not None,
                )
            start = time.perf_counter()
            try:
                raw_response = self._client.chat.completions.with_raw_response.create(
                    **kwargs
                )
                completion = raw_response.parse()
            except Exception as exc:
                duration_ms = (time.perf_counter() - start) * 1000.0
                details, rate_limit_signal, report_path = self._handle_completion_error(
                    exc=exc,
                    model=normalized_model,
                    messages=messages,
                    response_schema=response_schema,
                    response_schema_name=response_schema_name,
                    max_completion_tokens=max_completion_tokens,
                    reasoning_effort=normalized_reasoning_effort,
                    estimated_tokens=estimated_tokens,
                    duration_ms=duration_ms,
                    queue_retry_attempt=queue_retries + 1,
                )
                self._log_cerebras_error(
                    details=details,
                    rate_limit_signal=rate_limit_signal,
                    report_path=report_path,
                )
                if (
                    rate_limit_signal is not None
                    and rate_limit_signal.schedule_wait_seconds is not None
                    and rate_limit_signal.schedule_wait_seconds > 0
                ):
                    wait_seconds = rate_limit_signal.schedule_wait_seconds
                    self._check_rate_limit_wait_budget(
                        reason="reactive_retry_limit",
                        retries_completed=rate_limit_retries,
                        next_wait_seconds=wait_seconds,
                        cumulative_wait_seconds=cumulative_rate_limit_wait_seconds,
                        check_retry_limit=True,
                    )
                    rate_limit_retries += 1
                    if _is_queue_rate_limit_signal(rate_limit_signal):
                        queue_retries += 1
                    if rate_limit_signal.quota_wait_eligible:
                        quota_wait_ms += wait_seconds * 1000.0
                    cumulative_rate_limit_wait_seconds += wait_seconds
                    if self.logger:
                        resume_at = _format_future_time(wait_seconds)
                        self.logger.warning(
                            _format_rate_limit_wait_message(
                                signal=rate_limit_signal,
                                wait_seconds=wait_seconds,
                                resume_at=resume_at,
                                report_path=report_path,
                            ),
                            model=normalized_model,
                            wait_seconds=round(wait_seconds, 3),
                            resume_at=resume_at,
                            wait_reason=rate_limit_signal.schedule_reason,
                            wait_source_header=(rate_limit_signal.wait_source_header),
                            retry_after_seconds=(rate_limit_signal.retry_after_seconds),
                            reset_tokens_minute_seconds=(
                                rate_limit_signal.reset_tokens_minute_seconds
                            ),
                            reset_requests_day_seconds=(
                                rate_limit_signal.reset_requests_day_seconds
                            ),
                            queue_retry_attempt=(rate_limit_signal.queue_retry_attempt),
                            queue_backoff_min_seconds=(
                                rate_limit_signal.queue_backoff_min_seconds
                            ),
                            queue_backoff_max_seconds=(
                                rate_limit_signal.queue_backoff_max_seconds
                            ),
                            missing_expected_headers=list(
                                rate_limit_signal.missing_expected_headers
                            ),
                            quota_wait_eligible=rate_limit_signal.quota_wait_eligible,
                            retry_count=rate_limit_retries,
                            report_path=str(report_path) if report_path else None,
                        )
                    time.sleep(wait_seconds)
                    self._clear_rate_limit_headers(normalized_model)
                    continue
                # SDK exception strings may contain response bodies, request
                # fragments, or credentials.  The detailed cause is consumed
                # above for in-memory classification only and must not cross
                # this public exception boundary.
                raise CerebrasTemplateError(
                    f"Cerebras completion failed for {normalized_model}"
                ) from None

            duration_ms = (time.perf_counter() - start) * 1000.0
            choice = completion.choices[0]
            message = choice.message
            finish_reason = getattr(choice, "finish_reason", None)
            usage = TokenUsage.from_provider_usage(getattr(completion, "usage", None))
            rate_limit_headers = CerebrasRateLimitHeaders.from_headers(
                getattr(raw_response, "headers", None)
            )
            self._record_successful_call(
                model=normalized_model,
                token_usage=usage,
                rate_limit_headers=rate_limit_headers,
            )
            if self.logger:
                response_text = _message_content(message)
                self.logger.info(
                    "Cerebras response received",
                    model=getattr(completion, "model", None) or normalized_model,
                    reasoning_effort=normalized_reasoning_effort,
                    finish_reason=finish_reason,
                    duration_ms=round(duration_ms, 1),
                    estimated_request_tokens=estimated_tokens,
                    token_usage=_token_usage_to_dict(usage),
                    rate_limit_headers=(
                        rate_limit_headers.as_dict()
                        if rate_limit_headers is not None
                        else None
                    ),
                    quota_wait_ms=round(quota_wait_ms, 1),
                    content_type=type(_get_field(message, "content")).__name__,
                    content_chars=len(response_text),
                )
            else:
                response_text = _message_content(message)
            return CompletionCallResult(
                text=response_text,
                duration_ms=duration_ms,
                model=getattr(completion, "model", None) or normalized_model,
                finish_reason=finish_reason,
                token_usage=usage,
                cost=0.0,
                estimated_request_tokens=estimated_tokens,
                rate_limit_headers=rate_limit_headers,
                quota_wait_ms=quota_wait_ms,
            )

        # Every normal path returns, raises, or consumes a bounded retry above.
        # Keep a typed fail-closed guard in case future edits add another continue.
        raise CerebrasRateLimitBudgetExceededError(
            reason="retry_loop_exhausted",
            retries_completed=rate_limit_retries,
            max_retries=self.max_rate_limit_retries,
            cumulative_wait_seconds=cumulative_rate_limit_wait_seconds,
            max_wait_seconds=self.max_rate_limit_wait_seconds,
            next_wait_seconds=0.0,
        ) from None

    def _check_rate_limit_wait_budget(
        self,
        *,
        reason: str,
        retries_completed: int,
        next_wait_seconds: float,
        cumulative_wait_seconds: float,
        check_retry_limit: bool,
    ) -> None:
        """Fail before sleeping when either per-call rate-limit budget is spent."""

        retry_limit_exceeded = (
            check_retry_limit and retries_completed >= self.max_rate_limit_retries
        )
        wait_limit_exceeded = (
            not math.isfinite(next_wait_seconds)
            or cumulative_wait_seconds + next_wait_seconds
            > self.max_rate_limit_wait_seconds
        )
        if not retry_limit_exceeded and not wait_limit_exceeded:
            return

        exhausted_reason = reason if retry_limit_exceeded else "cumulative_wait_limit"
        raise CerebrasRateLimitBudgetExceededError(
            reason=exhausted_reason,
            retries_completed=retries_completed,
            max_retries=self.max_rate_limit_retries,
            cumulative_wait_seconds=cumulative_wait_seconds,
            max_wait_seconds=self.max_rate_limit_wait_seconds,
            next_wait_seconds=next_wait_seconds,
        ) from None

    @property
    def _client(self) -> Any:
        if self._sdk_client is None:
            self._sdk_client = _create_sdk_client(self.api_base)
        return self._sdk_client

    def _completion_kwargs(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        response_schema: dict[str, Any] | None,
        response_schema_name: str | None,
        max_completion_tokens: int,
        temperature: float | None,
        reasoning_effort: str | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": normalize_cerebras_model(model),
            "messages": messages,
            "max_completion_tokens": max_completion_tokens,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        normalized_reasoning_effort = _optional_text(reasoning_effort)
        if normalized_reasoning_effort:
            kwargs["reasoning_effort"] = normalized_reasoning_effort
        if self.service_tier:
            kwargs["service_tier"] = self.service_tier
        if response_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_schema_name or "structured_response",
                    "strict": True,
                    "schema": response_schema,
                },
            }
        return kwargs

    def _handle_completion_error(
        self,
        *,
        exc: BaseException,
        model: str,
        messages: list[dict[str, Any]],
        response_schema: dict[str, Any] | None,
        response_schema_name: str | None,
        max_completion_tokens: int,
        estimated_tokens: int,
        duration_ms: float,
        queue_retry_attempt: int,
        reasoning_effort: str | None = None,
    ) -> tuple[dict[str, Any], CerebrasRateLimitSignal | None, Path | None]:
        details = _exception_details(exc)
        signal = _extract_cerebras_rate_limit_signal(
            details,
            queue_backoff_seconds=self.queue_backoff_seconds,
            queue_backoff_initial_jitter_ratio=(
                self.queue_backoff_initial_jitter_ratio
            ),
            queue_backoff_second_min_seconds=(self.queue_backoff_second_min_seconds),
            queue_backoff_second_max_seconds=(self.queue_backoff_second_max_seconds),
            queue_backoff_cap_min_seconds=self.queue_backoff_cap_min_seconds,
            queue_backoff_cap_max_seconds=self.queue_backoff_cap_max_seconds,
            queue_retry_attempt=queue_retry_attempt,
            retry_buffer_seconds=self.rate_limit_retry_buffer_seconds,
        )
        report_path = None
        if signal is not None:
            report_path = self._write_rate_limit_report(
                model=model,
                messages=messages,
                response_schema=response_schema,
                response_schema_name=response_schema_name,
                max_completion_tokens=max_completion_tokens,
                reasoning_effort=reasoning_effort,
                estimated_tokens=estimated_tokens,
                duration_ms=duration_ms,
                error_details=_safe_exception_diagnostics(details),
                rate_limit_signal=signal,
            )
        return _safe_exception_diagnostics(details), signal, report_path

    def _record_attempt(
        self,
        *,
        model: str,
        estimated_tokens: int,
    ) -> dict[str, Any]:
        with self._metrics_lock:
            previous_estimated_tokens = (
                self._last_estimated_request_tokens_by_model.get(model)
            )
            previous_successful_usage = self._last_successful_token_usage_by_model.get(
                model
            )
            previous_headers_snapshot = self._last_rate_limit_headers_by_model.get(
                model
            )
            self._attempted_calls += 1
            self._attempted_calls_by_model[model] = (
                self._attempted_calls_by_model.get(model, 0) + 1
            )
            self._estimated_request_tokens += estimated_tokens
            self._estimated_request_tokens_by_model[model] = (
                self._estimated_request_tokens_by_model.get(model, 0) + estimated_tokens
            )
            self._last_estimated_request_tokens_by_model[model] = estimated_tokens

        previous_headers = (
            previous_headers_snapshot[0] if previous_headers_snapshot else None
        )
        return {
            "previous_estimated_request_tokens": previous_estimated_tokens,
            "estimated_request_token_delta_since_previous": (
                estimated_tokens - previous_estimated_tokens
                if previous_estimated_tokens is not None
                else None
            ),
            "previous_successful_token_usage": _token_usage_to_dict(
                previous_successful_usage
            )
            if previous_successful_usage is not None
            else None,
            "previous_rate_limit_headers": (
                previous_headers.as_dict() if previous_headers is not None else None
            ),
        }

    def _proactive_quota_wait_seconds(
        self,
        *,
        model: str,
        estimated_tokens: int,
    ) -> float:
        """Avoid a known token-quota rejection when the last response exposes it."""

        if not self.proactive_token_pacing:
            return 0.0
        with self._metrics_lock:
            snapshot = self._last_rate_limit_headers_by_model.get(model)
        if snapshot is None:
            return 0.0

        headers, observed_at = snapshot
        remaining = headers.remaining_tokens_minute
        if remaining is None or estimated_tokens <= remaining:
            return 0.0

        if headers.reset_tokens_minute_seconds is not None:
            reset_seconds = headers.reset_tokens_minute_seconds
        else:
            elapsed = max(0.0, time.perf_counter() - observed_at)
            reset_seconds = max(0.0, self.token_quota_window_seconds - elapsed)
        return max(0.0, reset_seconds + self.rate_limit_retry_buffer_seconds)

    def _clear_rate_limit_headers(self, model: str) -> None:
        with self._metrics_lock:
            self._last_rate_limit_headers_by_model.pop(model, None)

    def _record_successful_call(
        self,
        *,
        model: str,
        token_usage: TokenUsage | None,
        rate_limit_headers: CerebrasRateLimitHeaders | None = None,
    ) -> None:
        with self._metrics_lock:
            self._successful_calls += 1
            self._successful_calls_by_model[model] = (
                self._successful_calls_by_model.get(model, 0) + 1
            )
            if token_usage is not None:
                self._total_token_usage = (
                    add_token_usage(
                        self._total_token_usage,
                        token_usage,
                    )
                    or TokenUsage()
                )
                self._token_usage_by_model[model] = (
                    add_token_usage(
                        self._token_usage_by_model.get(model),
                        token_usage,
                    )
                    or TokenUsage()
                )
                self._last_successful_token_usage_by_model[model] = token_usage
            if rate_limit_headers is not None:
                self._last_rate_limit_headers_by_model[model] = (
                    rate_limit_headers,
                    time.perf_counter(),
                )

    def _write_rate_limit_report(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        response_schema: dict[str, Any] | None,
        response_schema_name: str | None,
        max_completion_tokens: int,
        estimated_tokens: int,
        duration_ms: float,
        error_details: dict[str, Any],
        rate_limit_signal: CerebrasRateLimitSignal,
        reasoning_effort: str | None = None,
    ) -> Path | None:
        created_at = datetime.now().astimezone()
        retry_at = (
            created_at + timedelta(seconds=rate_limit_signal.schedule_wait_seconds)
            if rate_limit_signal.schedule_wait_seconds is not None
            else None
        )
        previous_rate_limit_at = self._previous_rate_limit_at
        previous_retry_at = self._previous_rate_limit_retry_at
        wall_time_since_previous_rate_limit = (
            max(0.0, (created_at - previous_rate_limit_at).total_seconds())
            if previous_rate_limit_at is not None
            else None
        )
        wall_time_since_previous_retry_at = (
            max(0.0, (created_at - previous_retry_at).total_seconds())
            if previous_retry_at is not None
            else None
        )
        with self._metrics_lock:
            successful_calls = self._successful_calls
            successful_calls_by_model = dict(self._successful_calls_by_model)
            attempted_calls = self._attempted_calls
            attempted_calls_by_model = dict(self._attempted_calls_by_model)
            total_token_usage = self._total_token_usage
            token_usage_by_model = dict(self._token_usage_by_model)
            estimated_request_tokens = self._estimated_request_tokens
            estimated_request_tokens_by_model = dict(
                self._estimated_request_tokens_by_model
            )
            last_estimated_request_tokens_by_model = dict(
                self._last_estimated_request_tokens_by_model
            )
            last_rate_limit_headers_by_model = {
                key: value[0]
                for key, value in self._last_rate_limit_headers_by_model.items()
            }
            tokens_since_previous_rate_limit = _subtract_token_usage(
                total_token_usage,
                self._last_rate_limit_total_token_usage,
            )
            estimated_tokens_since_previous_rate_limit = (
                estimated_request_tokens
                - self._last_rate_limit_estimated_request_tokens
            )
            successful_calls_since_previous_rate_limit = (
                successful_calls - self._last_rate_limit_successful_calls
            )
        payload = {
            "schema_version": 1,
            "event": "cerebras_rate_limit",
            "created_at": created_at.isoformat(),
            "session_started_at": self._session_started_at.isoformat(),
            "wall_time_until_rate_limit_seconds": round(
                time.perf_counter() - self._session_started_monotonic,
                3,
            ),
            "previous_rate_limit_at": (
                previous_rate_limit_at.isoformat()
                if previous_rate_limit_at is not None
                else None
            ),
            "wall_time_since_previous_rate_limit_seconds": (
                round(wall_time_since_previous_rate_limit, 3)
                if wall_time_since_previous_rate_limit is not None
                else None
            ),
            "previous_retry_at": (
                previous_retry_at.isoformat() if previous_retry_at is not None else None
            ),
            "wall_time_since_previous_retry_at_seconds": (
                round(wall_time_since_previous_retry_at, 3)
                if wall_time_since_previous_retry_at is not None
                else None
            ),
            "retry_at": retry_at.isoformat() if retry_at is not None else None,
            "wait_seconds": (
                round(rate_limit_signal.schedule_wait_seconds, 3)
                if rate_limit_signal.schedule_wait_seconds is not None
                else None
            ),
            "rate_limit_signal": rate_limit_signal.as_dict(),
            "model": model,
            "service_tier": self.service_tier,
            "successful_cerebras_calls": successful_calls,
            "successful_cerebras_calls_by_model": successful_calls_by_model,
            "attempted_cerebras_calls": attempted_calls,
            "attempted_cerebras_calls_by_model": attempted_calls_by_model,
            "tokens_consumed": _token_usage_to_dict(total_token_usage),
            "tokens_consumed_since_previous_rate_limit": _token_usage_to_dict(
                tokens_since_previous_rate_limit
            ),
            "tokens_consumed_by_model": {
                key: _token_usage_to_dict(value)
                for key, value in sorted(token_usage_by_model.items())
            },
            "estimated_request_tokens_attempted": estimated_request_tokens,
            "estimated_request_tokens_attempted_since_previous_rate_limit": (
                estimated_tokens_since_previous_rate_limit
            ),
            "successful_cerebras_calls_since_previous_rate_limit": (
                successful_calls_since_previous_rate_limit
            ),
            "estimated_request_tokens_attempted_by_model": dict(
                sorted(estimated_request_tokens_by_model.items())
            ),
            "last_estimated_request_tokens_by_model": dict(
                sorted(last_estimated_request_tokens_by_model.items())
            ),
            "last_rate_limit_headers_by_model": {
                key: value.as_dict()
                for key, value in sorted(last_rate_limit_headers_by_model.items())
            },
            "current_call": {
                "num_messages": len(messages),
                "prompt_chars": len(
                    json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
                ),
                "estimated_request_tokens": estimated_tokens,
                "max_completion_tokens": max_completion_tokens,
                "reasoning_effort": reasoning_effort,
                "has_output_schema": response_schema is not None,
                "output_schema_name": response_schema_name,
                "duration_ms_until_error": round(duration_ms, 1),
            },
            "error_details": _json_safe(error_details),
        }

        timestamp = created_at.strftime("%Y%m%d-%H%M%S")
        filename = f"cerebras-rate-limit-{timestamp}-{uuid4().hex[:8]}.json"
        try:
            self.rate_limit_report_dir.mkdir(parents=True, exist_ok=True)
            path = self.rate_limit_report_dir / filename
            path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            self._previous_rate_limit_at = created_at
            self._previous_rate_limit_retry_at = retry_at
            self._record_rate_limit_report_snapshot(
                total_token_usage=total_token_usage,
                estimated_request_tokens=estimated_request_tokens,
                successful_calls=successful_calls,
            )
            return path
        except OSError as write_exc:
            if self.logger:
                self.logger.warning(
                    "Failed to write Cerebras rate-limit report",
                    report_dir=str(self.rate_limit_report_dir),
                    error=str(write_exc),
                )
            self._previous_rate_limit_at = created_at
            self._previous_rate_limit_retry_at = retry_at
            self._record_rate_limit_report_snapshot(
                total_token_usage=total_token_usage,
                estimated_request_tokens=estimated_request_tokens,
                successful_calls=successful_calls,
            )
            return None

    def _record_rate_limit_report_snapshot(
        self,
        *,
        total_token_usage: TokenUsage,
        estimated_request_tokens: int,
        successful_calls: int,
    ) -> None:
        with self._metrics_lock:
            self._last_rate_limit_total_token_usage = TokenUsage(
                input_tokens=total_token_usage.input_tokens,
                cached_input_tokens=total_token_usage.cached_input_tokens,
                output_tokens=total_token_usage.output_tokens,
                reasoning_output_tokens=total_token_usage.reasoning_output_tokens,
                total_tokens=total_token_usage.total_tokens,
            )
            self._last_rate_limit_estimated_request_tokens = estimated_request_tokens
            self._last_rate_limit_successful_calls = successful_calls

    def _log_cerebras_error(
        self,
        *,
        details: dict[str, Any],
        rate_limit_signal: CerebrasRateLimitSignal | None,
        report_path: Path | None,
    ) -> None:
        if self.logger is None:
            return
        signal_dict = (
            rate_limit_signal.as_dict() if rate_limit_signal is not None else None
        )
        self.logger.warning(
            "Cerebras SDK error observed",
            exception_type=details.get("exception_type"),
            status_code=details.get("status_code"),
            rate_limit_signal=signal_dict,
            rate_limit_report_path=str(report_path)
            if report_path is not None
            else None,
        )
        self.logger.debug(
            "Cerebras SDK error diagnostics",
            diagnostics=_json_safe(details),
        )


def estimate_request_tokens(
    *,
    messages: list[dict[str, Any]],
    max_completion_tokens: int,
    chars_per_token: float = DEFAULT_TOKEN_ESTIMATE_CHARS_PER_TOKEN,
    safety_factor: float = DEFAULT_TOKEN_SAFETY_FACTOR,
) -> int:
    """Estimate request size for logs and rate-limit reports."""

    chars_per_token = max(chars_per_token, 1.0)
    safety_factor = max(safety_factor, 1.0)
    prompt_chars = len(json.dumps(messages, ensure_ascii=False, separators=(",", ":")))
    estimated_prompt_tokens = max(1, math.ceil(prompt_chars / chars_per_token))
    completion_budget = max(0, max_completion_tokens)
    return math.ceil((estimated_prompt_tokens + completion_budget) * safety_factor)


def normalize_cerebras_model(model: str) -> str:
    """Accept legacy `cerebras/...` model names while calling SDK natively."""

    if model.startswith("cerebras/"):
        return model.split("/", 1)[1]
    return model


def _create_sdk_client(api_base: str | None) -> Any:
    try:
        from cerebras.cloud.sdk import Cerebras
    except ImportError as exc:
        raise CerebrasTemplateError(
            "cerebras-cloud-sdk is required for Track 2 Cerebras templates. "
            "Install the track-2-agent extra or run `uv sync --extra track-2-agent`."
        ) from exc

    kwargs: dict[str, Any] = {"max_retries": 0}
    if os.getenv("CEREBRAS_API_KEY"):
        kwargs["api_key"] = os.getenv("CEREBRAS_API_KEY")
    normalized_api_base = normalize_cerebras_api_base(api_base)
    if normalized_api_base:
        kwargs["base_url"] = normalized_api_base
    try:
        return Cerebras(**kwargs, warm_tcp_connection=False)
    except TypeError:
        return Cerebras(**kwargs)


def normalize_cerebras_api_base(api_base: str | None) -> str | None:
    """Normalize SDK base URLs, accepting legacy `/v1` values."""

    if api_base is None:
        return None
    normalized = api_base.strip().rstrip("/")
    if not normalized:
        return None
    if normalized.endswith("/v1"):
        return normalized[: -len("/v1")]
    return normalized


def _extract_cerebras_rate_limit_signal(
    details: dict[str, Any],
    *,
    queue_backoff_seconds: float,
    queue_backoff_initial_jitter_ratio: float = (
        DEFAULT_CEREBRAS_QUEUE_BACKOFF_INITIAL_JITTER_RATIO
    ),
    queue_backoff_second_min_seconds: float = (
        DEFAULT_CEREBRAS_QUEUE_BACKOFF_SECOND_MIN_SECONDS
    ),
    queue_backoff_second_max_seconds: float = (
        DEFAULT_CEREBRAS_QUEUE_BACKOFF_SECOND_MAX_SECONDS
    ),
    queue_backoff_cap_min_seconds: float = (
        DEFAULT_CEREBRAS_QUEUE_BACKOFF_CAP_MIN_SECONDS
    ),
    queue_backoff_cap_max_seconds: float = (
        DEFAULT_CEREBRAS_QUEUE_BACKOFF_CAP_MAX_SECONDS
    ),
    queue_retry_attempt: int = 1,
    retry_buffer_seconds: float = DEFAULT_CEREBRAS_RATE_LIMIT_RETRY_BUFFER_SECONDS,
) -> CerebrasRateLimitSignal | None:
    return _rate_limit_candidate(
        body=details.get("provider_error_body") or details.get("response_json"),
        headers=details.get("provider_error_headers")
        or details.get("response_headers"),
        status_code=details.get("status_code"),
        source="sdk",
        queue_backoff_seconds=queue_backoff_seconds,
        queue_backoff_initial_jitter_ratio=queue_backoff_initial_jitter_ratio,
        queue_backoff_second_min_seconds=queue_backoff_second_min_seconds,
        queue_backoff_second_max_seconds=queue_backoff_second_max_seconds,
        queue_backoff_cap_min_seconds=queue_backoff_cap_min_seconds,
        queue_backoff_cap_max_seconds=queue_backoff_cap_max_seconds,
        queue_retry_attempt=queue_retry_attempt,
        retry_buffer_seconds=retry_buffer_seconds,
    )


def _rate_limit_candidate(
    *,
    body: Any,
    headers: Any,
    status_code: Any,
    source: str,
    queue_backoff_seconds: float,
    queue_backoff_initial_jitter_ratio: float,
    queue_backoff_second_min_seconds: float,
    queue_backoff_second_max_seconds: float,
    queue_backoff_cap_min_seconds: float,
    queue_backoff_cap_max_seconds: float,
    queue_retry_attempt: int,
    retry_buffer_seconds: float = DEFAULT_CEREBRAS_RATE_LIMIT_RETRY_BUFFER_SECONDS,
) -> CerebrasRateLimitSignal | None:
    headers_dict = _headers_dict(headers) or {}
    payload = _error_payload(body)
    code = _payload_str(payload, "code")
    error_type = _payload_str(payload, "type")
    param = _payload_str(payload, "param")
    message = _payload_str(payload, "message")
    retry_after_seconds = _retry_after_seconds(headers_dict)
    reset_tokens_minute_seconds = _safe_float_or_none(
        _header_value(headers_dict, "x-ratelimit-reset-tokens-minute")
    )
    reset_requests_day_seconds = _safe_float_or_none(
        _header_value(headers_dict, "x-ratelimit-reset-requests-day")
    )
    is_429 = _safe_int(status_code) == 429
    is_rate_limit_payload = code in {
        "queue_exceeded",
        "token_quota_exceeded",
        "request_quota_exceeded",
    } or error_type in {
        "too_many_requests_error",
        "too_many_tokens_error",
        "rate_limit_error",
    }
    if not is_429 and not is_rate_limit_payload:
        return None

    schedule_wait_seconds = None
    schedule_reason = None
    wait_source_header = None
    missing_expected_headers: tuple[str, ...] = ()
    quota_wait_eligible = False
    queue_backoff_min_seconds = None
    queue_backoff_max_seconds = None
    queue_signal_attempt = None
    expects_token_reset = code == "token_quota_exceeded" or (
        error_type == "too_many_tokens_error"
    )
    if code == "queue_exceeded" or param == "queue":
        queue_signal_attempt = max(1, _safe_int(queue_retry_attempt))
        if retry_after_seconds is not None:
            schedule_wait_seconds = max(
                0.0,
                retry_after_seconds + max(0.0, retry_buffer_seconds),
            )
            schedule_reason = f"{source}_queue_retry_after"
            wait_source_header = "retry-after"
        else:
            (
                schedule_wait_seconds,
                queue_backoff_min_seconds,
                queue_backoff_max_seconds,
            ) = _queue_backoff_wait_seconds(
                attempt=queue_signal_attempt,
                initial_seconds=queue_backoff_seconds,
                initial_jitter_ratio=queue_backoff_initial_jitter_ratio,
                second_min_seconds=queue_backoff_second_min_seconds,
                second_max_seconds=queue_backoff_second_max_seconds,
                cap_min_seconds=queue_backoff_cap_min_seconds,
                cap_max_seconds=queue_backoff_cap_max_seconds,
            )
            schedule_reason = f"{source}_queue_exceeded_jitter_backoff"
    elif reset_tokens_minute_seconds is not None:
        schedule_wait_seconds = max(
            0.0,
            reset_tokens_minute_seconds + max(0.0, retry_buffer_seconds),
        )
        schedule_reason = f"{source}_headers_tokens_minute_reset"
        wait_source_header = "x-ratelimit-reset-tokens-minute"
        quota_wait_eligible = True
    elif reset_requests_day_seconds is not None:
        schedule_wait_seconds = max(
            0.0,
            reset_requests_day_seconds + max(0.0, retry_buffer_seconds),
        )
        schedule_reason = f"{source}_headers_requests_day_reset"
        wait_source_header = "x-ratelimit-reset-requests-day"
        if expects_token_reset:
            missing_expected_headers = ("x-ratelimit-reset-tokens-minute",)
        quota_wait_eligible = True
    elif retry_after_seconds is not None:
        schedule_wait_seconds = max(
            0.0,
            retry_after_seconds + max(0.0, retry_buffer_seconds),
        )
        schedule_reason = f"{source}_retry_after"
        wait_source_header = "retry-after"
        if expects_token_reset:
            missing_expected_headers = ("x-ratelimit-reset-tokens-minute",)
        quota_wait_eligible = True

    return CerebrasRateLimitSignal(
        code=code,
        type=error_type,
        param=param,
        message=message,
        source=source,
        retry_after_seconds=retry_after_seconds,
        reset_tokens_minute_seconds=reset_tokens_minute_seconds,
        reset_requests_day_seconds=reset_requests_day_seconds,
        schedule_wait_seconds=schedule_wait_seconds,
        schedule_reason=schedule_reason,
        wait_source_header=wait_source_header,
        missing_expected_headers=missing_expected_headers,
        x_should_retry=_header_value(headers_dict, "x-should-retry"),
        quota_wait_eligible=quota_wait_eligible,
        queue_retry_attempt=queue_signal_attempt,
        queue_backoff_min_seconds=queue_backoff_min_seconds,
        queue_backoff_max_seconds=queue_backoff_max_seconds,
    )


def _token_usage_to_dict(usage: TokenUsage | None) -> dict[str, int]:
    if usage is None:
        usage = TokenUsage()
    return {
        "input_tokens": usage.input_tokens,
        "cached_input_tokens": usage.cached_input_tokens,
        "output_tokens": usage.output_tokens,
        "reasoning_output_tokens": usage.reasoning_output_tokens,
        "total_tokens": usage.total_tokens,
    }


def _queue_backoff_wait_seconds(
    *,
    attempt: int,
    initial_seconds: float,
    initial_jitter_ratio: float,
    second_min_seconds: float,
    second_max_seconds: float,
    cap_min_seconds: float,
    cap_max_seconds: float,
) -> tuple[float, float, float]:
    attempt = max(1, _safe_int(attempt))
    initial_seconds = max(0.0, initial_seconds)
    initial_jitter_ratio = max(0.0, initial_jitter_ratio)
    if attempt == 1:
        minimum = max(0.0, initial_seconds * (1.0 - initial_jitter_ratio))
        maximum = max(minimum, initial_seconds * (1.0 + initial_jitter_ratio))
    elif attempt == 2:
        minimum, maximum = _ordered_nonnegative_range(
            second_min_seconds,
            second_max_seconds,
        )
    else:
        minimum, maximum = _ordered_nonnegative_range(
            cap_min_seconds,
            cap_max_seconds,
        )
    return random.uniform(minimum, maximum), minimum, maximum


def _ordered_nonnegative_range(left: float, right: float) -> tuple[float, float]:
    minimum = max(0.0, left)
    maximum = max(0.0, right)
    if maximum < minimum:
        minimum, maximum = maximum, minimum
    return minimum, maximum


def _is_queue_rate_limit_signal(signal: CerebrasRateLimitSignal) -> bool:
    return signal.code == "queue_exceeded" or signal.param == "queue"


def _format_rate_limit_wait_message(
    *,
    signal: CerebrasRateLimitSignal,
    wait_seconds: float,
    resume_at: str,
    report_path: Path | None,
) -> str:
    parts = [
        f"Cerebras rate limit reached; waiting {wait_seconds:.3f}s before retry",
        f"resume_at={resume_at}",
        f"reason={signal.schedule_reason or 'unknown'}",
        f"source_header={signal.wait_source_header or 'none'}",
    ]
    if signal.queue_retry_attempt is not None:
        parts.append(f"queue_attempt={signal.queue_retry_attempt}")
    if (
        signal.queue_backoff_min_seconds is not None
        and signal.queue_backoff_max_seconds is not None
    ):
        parts.append(
            "queue_wait_range="
            f"{signal.queue_backoff_min_seconds:.3f}-"
            f"{signal.queue_backoff_max_seconds:.3f}s"
        )
    if signal.reset_tokens_minute_seconds is not None:
        parts.append(f"reset_tokens_minute={signal.reset_tokens_minute_seconds:.3f}s")
    elif "x-ratelimit-reset-tokens-minute" in signal.missing_expected_headers:
        parts.append("reset_tokens_minute=missing")
    if signal.reset_requests_day_seconds is not None:
        parts.append(f"reset_requests_day={signal.reset_requests_day_seconds:.3f}s")
    if signal.retry_after_seconds is not None:
        parts.append(f"retry_after={signal.retry_after_seconds:.3f}s")
    if signal.missing_expected_headers:
        parts.append(
            "missing_expected_headers=" + ",".join(signal.missing_expected_headers)
        )
    parts.append(f"quota_wait_eligible={signal.quota_wait_eligible}")
    if report_path is not None:
        parts.append(f"report={report_path}")
    return " | ".join(parts)


def _subtract_token_usage(left: TokenUsage, right: TokenUsage) -> TokenUsage:
    return TokenUsage(
        input_tokens=max(0, left.input_tokens - right.input_tokens),
        cached_input_tokens=max(
            0,
            left.cached_input_tokens - right.cached_input_tokens,
        ),
        output_tokens=max(0, left.output_tokens - right.output_tokens),
        reasoning_output_tokens=max(
            0,
            left.reasoning_output_tokens - right.reasoning_output_tokens,
        ),
        total_tokens=max(0, left.total_tokens - right.total_tokens),
    )


def _exception_details(exc: BaseException) -> dict[str, Any]:
    response = getattr(exc, "response", None)
    status_code = getattr(exc, "status_code", None) or getattr(
        response,
        "status_code",
        None,
    )
    response_text = getattr(response, "text", None)
    response_json = None
    if response is not None and hasattr(response, "json"):
        try:
            response_json = response.json()
        except Exception:
            response_json = None
    provider_body = _provider_error_body(exc, response_json)
    return {
        "exception_type": type(exc).__name__,
        "exception_module": type(exc).__module__,
        "message": str(exc),
        "status_code": status_code,
        "code": getattr(exc, "code", None),
        "type": getattr(exc, "type", None),
        "response_headers": _headers_dict(getattr(response, "headers", None)),
        "response_text": response_text,
        "response_json": _json_safe(response_json),
        "exception_attrs": _exception_attrs(exc),
        "cause": _exception_summary(getattr(exc, "__cause__", None)),
        "context": _exception_summary(getattr(exc, "__context__", None)),
        "provider_error_body": provider_body,
        "provider_error_headers": _provider_error_headers(exc),
    }


def _safe_exception_diagnostics(details: dict[str, Any]) -> dict[str, Any]:
    """Project raw SDK error data into non-secret persistent diagnostics.

    The raw structure is needed briefly to classify 429 responses, but it may
    contain provider bodies, arbitrary headers, exception attributes, and
    chained exception text.  None of those values are safe to write or log.
    """

    status_code = details.get("status_code")
    return {
        "exception_type": _safe_class_name(details.get("exception_type")),
        "exception_module": _safe_module_name(details.get("exception_module")),
        "status_code": _safe_http_status(status_code),
    }


def _safe_class_name(value: Any) -> str:
    text = str(value) if value is not None else "Exception"
    return text if text.replace("_", "").isalnum() else "Exception"


def _safe_module_name(value: Any) -> str:
    text = str(value) if value is not None else "unknown"
    return text if text.replace("_", "").replace(".", "").isalnum() else "unknown"


def _safe_http_status(value: Any) -> int | None:
    try:
        status = int(value)
    except (TypeError, ValueError):
        return None
    return status if 100 <= status <= 599 else None


def _allowlisted_diagnostic(
    value: str | None,
    allowed: set[str],
) -> str | None:
    if value is None:
        return None
    normalized = value.casefold()
    return normalized if normalized in allowed else "other"


def _provider_error_body(exc: BaseException, response_json: Any = None) -> Any:
    for candidate in (
        getattr(exc, "body", None),
        getattr(exc, "error", None),
        response_json,
    ):
        payload = _error_payload(candidate)
        if payload:
            return payload
    context = getattr(exc, "__context__", None)
    if context is not None:
        return _provider_error_body(context, None)
    return None


def _provider_error_headers(exc: BaseException) -> dict[str, str] | None:
    response = getattr(exc, "response", None)
    headers = _headers_dict(getattr(response, "headers", None))
    if headers:
        return headers
    headers = _headers_dict(getattr(exc, "headers", None))
    if headers:
        return headers
    context = getattr(exc, "__context__", None)
    if context is not None:
        return _provider_error_headers(context)
    return None


def _message_content(message: Any) -> str:
    content = _get_field(message, "content")
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        # Some OpenAI-compatible SDK versions deserialize structured-output
        # content into a mapping instead of preserving the JSON string.  A
        # Python ``str(dict)`` is not JSON (single quotes, non-JSON literals),
        # so serialize it canonically before it crosses the compiler boundary.
        return json.dumps(
            content,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                text_parts.append(part["text"])
            elif hasattr(part, "text"):
                text_parts.append(str(part.text))
        return "".join(text_parts)
    return "" if content is None else str(content)


def _error_payload(body: Any) -> dict[str, Any]:
    if body is None:
        return {}
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            return error
        return body
    if isinstance(body, str):
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            return {"message": body}
        return _error_payload(parsed)
    return _object_public_attrs(body)


def _payload_str(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    return str(value) if value is not None else None


def _retry_after_seconds(headers: dict[str, str]) -> float | None:
    value = _header_value(headers, "retry-after")
    if value is None:
        return None
    seconds = _safe_float_or_none(value)
    if seconds is not None:
        return max(0.0, seconds)
    try:
        retry_at = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    now = datetime.now(tz=retry_at.tzinfo)
    return max(0.0, (retry_at - now).total_seconds())


def _headers_dict(headers: Any) -> dict[str, str] | None:
    if not headers:
        return None
    if hasattr(headers, "items"):
        try:
            return {str(key).lower(): str(value) for key, value in headers.items()}
        except Exception:
            return None
    if isinstance(headers, list):
        result = {}
        for item in headers:
            if isinstance(item, (tuple, list)) and len(item) == 2:
                result[str(item[0]).lower()] = str(item[1])
        return result or None
    return None


def _header_value(headers: dict[str, str] | None, name: str) -> str | None:
    if not headers:
        return None
    return headers.get(name.lower())


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _get_field(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _safe_int(value: Any) -> int:
    try:
        if value is None:
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0


def _safe_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@overload
def _env_float(name: str | tuple[str, ...], default: float) -> float: ...


@overload
def _env_float(name: str | tuple[str, ...], default: None = None) -> float | None: ...


def _env_float(
    name: str | tuple[str, ...], default: float | None = None
) -> float | None:
    names = (name,) if isinstance(name, str) else name
    configured_name, value = _first_configured_env(names)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError:
        raise ValueError(
            f"{configured_name} must be a finite non-negative number"
        ) from None
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(
            f"{configured_name} must be a finite non-negative number"
        ) from None
    return parsed


def _env_nonnegative_int(names: tuple[str, ...], default: int) -> int:
    name, raw_value = _first_configured_env(names)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _env_nonnegative_float(names: tuple[str, ...], default: float) -> float:
    name, raw_value = _first_configured_env(names)
    if raw_value is None:
        return default
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a finite non-negative number") from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return value


def _first_configured_env(names: tuple[str, ...]) -> tuple[str, str | None]:
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip() != "":
            return name, value.strip()
    return names[0], None


def _env_bool(name: str | tuple[str, ...], *, default: bool) -> bool:
    names = (name,) if isinstance(name, str) else name
    configured_name, value = _first_configured_env(names)
    if value is None:
        return default
    normalized = value.lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{configured_name} must be a boolean") from None


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(key): _json_safe(val) for key, val in value.items()}
        if isinstance(value, (list, tuple)):
            return [_json_safe(item) for item in value]
        return str(value)


def _exception_attrs(exc: BaseException) -> dict[str, Any]:
    attrs = {}
    for key in dir(exc):
        if key.startswith("_") or key in {"args", "with_traceback"}:
            continue
        try:
            value = getattr(exc, key)
        except Exception:
            continue
        if callable(value):
            continue
        if key in {"response", "__cause__", "__context__"}:
            continue
        if isinstance(value, (str, int, float, bool, type(None), dict, list, tuple)):
            attrs[key] = _json_safe(value)
    return attrs


def _exception_summary(exc: BaseException | None) -> dict[str, Any] | None:
    if exc is None:
        return None
    response = getattr(exc, "response", None)
    return {
        "exception_type": type(exc).__name__,
        "exception_module": type(exc).__module__,
        "message": str(exc),
        "status_code": getattr(exc, "status_code", None)
        or getattr(response, "status_code", None),
        "response_headers": _headers_dict(getattr(response, "headers", None)),
        "response_text": getattr(response, "text", None),
        "body": _json_safe(getattr(exc, "body", None)),
    }


def _object_public_attrs(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        try:
            value = getattr(obj, key)
        except Exception:
            continue
        if callable(value):
            continue
        if isinstance(value, (str, int, float, bool, type(None), dict, list, tuple)):
            result[key] = _json_safe(value)
    return result


def _format_future_time(wait_seconds: float) -> str:
    return (datetime.now().astimezone() + timedelta(seconds=wait_seconds)).isoformat()
