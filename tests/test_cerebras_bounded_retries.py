from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from track_2_agent_under_test_cerebras import cerebras_client as client_module
from track_2_agent_under_test_cerebras.cerebras_client import (
    CerebrasCompletionClient,
    CerebrasRateLimitBudgetExceededError,
    CerebrasRateLimitHeaders,
    CerebrasTemplateError,
)


@dataclass
class FakeClock:
    now: float = 0.0
    sleeps: list[float] = field(default_factory=list)

    def perf_counter(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class FakeProviderResponse:
    def __init__(
        self,
        *,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> None:
        self.status_code = 429
        self.headers = headers
        self.text = json.dumps(payload)
        self._payload = payload

    def json(self) -> dict[str, Any]:
        return self._payload


class FakeRateLimitError(Exception):
    def __init__(self, response: FakeProviderResponse, *, secret: str) -> None:
        super().__init__(f"provider rejected secret={secret}")
        self.response = response
        self.status_code = response.status_code


class FakeRawResponse:
    headers: dict[str, str] = {}

    def parse(self) -> Any:
        return SimpleNamespace(
            model="gpt-oss-120b",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="done"),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=2,
                total_tokens=12,
                prompt_tokens_details=SimpleNamespace(cached_tokens=0),
                completion_tokens_details=SimpleNamespace(reasoning_tokens=1),
            ),
        )


class FakeCreateEndpoint:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> object:
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class FakeSDKClient:
    def __init__(self, outcomes: list[object]) -> None:
        self.endpoint = FakeCreateEndpoint(outcomes)
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(
                with_raw_response=SimpleNamespace(create=self.endpoint.create)
            )
        )


class RecordingLogger:
    def __init__(self) -> None:
        self.records: list[tuple[str, str, dict[str, Any]]] = []

    def info(self, message: str, **fields: Any) -> None:
        self.records.append(("info", message, fields))

    def warning(self, message: str, **fields: Any) -> None:
        self.records.append(("warning", message, fields))

    def debug(self, message: str, **fields: Any) -> None:
        self.records.append(("debug", message, fields))


def queue_error(secret: str = "SENSITIVE_PROVIDER_VALUE") -> FakeRateLimitError:
    return FakeRateLimitError(
        FakeProviderResponse(
            headers={"retry-after": "2"},
            payload={
                "error": {
                    "code": "queue_exceeded",
                    "type": "too_many_requests_error",
                    "param": "queue",
                    "message": f"busy; secret={secret}",
                }
            },
        ),
        secret=secret,
    )


def token_quota_error() -> FakeRateLimitError:
    return FakeRateLimitError(
        FakeProviderResponse(
            headers={"x-ratelimit-reset-tokens-minute": "2"},
            payload={
                "error": {
                    "code": "token_quota_exceeded",
                    "type": "too_many_tokens_error",
                    "param": "quota",
                    "message": "token window exhausted",
                }
            },
        ),
        secret="not-present-in-budget-error",
    )


def configure_budgets(
    monkeypatch: pytest.MonkeyPatch,
    *,
    retries: int,
    wait_seconds: float,
) -> None:
    monkeypatch.setenv("PACT_CEREBRAS_MAX_RATE_LIMIT_RETRIES", str(retries))
    monkeypatch.setenv("PACT_CEREBRAS_MAX_RATE_LIMIT_WAIT_SECONDS", str(wait_seconds))
    monkeypatch.setenv("PACT_CEREBRAS_RATE_LIMIT_RETRY_BUFFER_SECONDS", "0")
    monkeypatch.setenv("PACT_CEREBRAS_PROACTIVE_TOKEN_PACING", "false")


def install_clock(monkeypatch: pytest.MonkeyPatch, clock: FakeClock) -> None:
    monkeypatch.setattr(client_module.time, "perf_counter", clock.perf_counter)
    monkeypatch.setattr(client_module.time, "sleep", clock.sleep)


def generate(client: CerebrasCompletionClient) -> Any:
    return client.generate(
        model="gpt-oss-120b",
        messages=[{"role": "user", "content": "hello"}],
        response_schema=None,
        response_schema_name=None,
        max_completion_tokens=16,
        temperature=None,
    )


def test_rate_limit_retry_count_is_bounded_and_error_is_secret_safe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configure_budgets(monkeypatch, retries=2, wait_seconds=100)
    clock = FakeClock()
    install_clock(monkeypatch, clock)
    sdk = FakeSDKClient([queue_error(), queue_error(), queue_error()])
    test_logger = RecordingLogger()
    client = CerebrasCompletionClient(sdk_client=sdk, logger=test_logger)
    client.rate_limit_report_dir = tmp_path

    with pytest.raises(CerebrasRateLimitBudgetExceededError) as caught:
        generate(client)

    error = caught.value
    assert isinstance(error, CerebrasTemplateError)
    assert error.reason == "reactive_retry_limit"
    assert error.retries_completed == 2
    assert error.max_retries == 2
    assert error.cumulative_wait_seconds == 4.0
    assert len(sdk.endpoint.calls) == 3
    assert clock.sleeps == [2.0, 2.0]
    assert "SENSITIVE_PROVIDER_VALUE" not in str(error)
    assert "busy" not in str(error)
    persisted = "\n".join(
        path.read_text(encoding="utf-8") for path in tmp_path.glob("*.json")
    )
    logged = json.dumps(test_logger.records, sort_keys=True)
    assert "SENSITIVE_PROVIDER_VALUE" not in persisted
    assert "busy" not in persisted
    assert "SENSITIVE_PROVIDER_VALUE" not in logged
    assert "busy" not in logged


def test_non_rate_limit_provider_exception_is_redacted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_budgets(monkeypatch, retries=0, wait_seconds=0)
    secret = "SENSITIVE_PROVIDER_EXCEPTION_VALUE"
    sdk = FakeSDKClient([RuntimeError(f"request failed with {secret}")])
    test_logger = RecordingLogger()
    client = CerebrasCompletionClient(sdk_client=sdk, logger=test_logger)

    with pytest.raises(CerebrasTemplateError) as caught:
        generate(client)

    assert secret not in str(caught.value)
    assert caught.value.__cause__ is None
    assert secret not in json.dumps(test_logger.records, sort_keys=True)


def test_rate_limit_header_projection_does_not_persist_raw_values() -> None:
    secret = "SENSITIVE_PROVIDER_HEADER_VALUE"

    headers = CerebrasRateLimitHeaders.from_headers(
        {"x-ratelimit-remaining-tokens-minute": secret}
    )

    assert headers is not None
    assert secret not in json.dumps(headers.as_dict(), sort_keys=True)


def test_cumulative_wait_budget_stops_before_oversized_sleep(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configure_budgets(monkeypatch, retries=5, wait_seconds=3)
    clock = FakeClock()
    install_clock(monkeypatch, clock)
    sdk = FakeSDKClient([queue_error(), queue_error()])
    client = CerebrasCompletionClient(sdk_client=sdk)
    client.rate_limit_report_dir = tmp_path

    with pytest.raises(CerebrasRateLimitBudgetExceededError) as caught:
        generate(client)

    error = caught.value
    assert error.reason == "cumulative_wait_limit"
    assert error.retries_completed == 1
    assert error.cumulative_wait_seconds == 2.0
    assert error.next_wait_seconds == 2.0
    assert len(sdk.endpoint.calls) == 2
    assert clock.sleeps == [2.0]


def test_quota_retry_preserves_wait_metrics_and_existing_backoff(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configure_budgets(monkeypatch, retries=2, wait_seconds=10)
    clock = FakeClock()
    install_clock(monkeypatch, clock)
    sdk = FakeSDKClient([token_quota_error(), FakeRawResponse()])
    client = CerebrasCompletionClient(sdk_client=sdk)
    client.rate_limit_report_dir = tmp_path

    result = generate(client)

    assert result.text == "done"
    assert result.quota_wait_ms == 2000.0
    assert result.token_usage.input_tokens == 10
    assert len(sdk.endpoint.calls) == 2
    assert clock.sleeps == [2.0]


def test_pact_budget_environment_names_take_precedence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRACK2_CEREBRAS_MAX_RATE_LIMIT_RETRIES", "9")
    monkeypatch.setenv("TRACK2_CEREBRAS_MAX_RATE_LIMIT_WAIT_SECONDS", "99")
    monkeypatch.setenv("PACT_CEREBRAS_MAX_RATE_LIMIT_RETRIES", "1")
    monkeypatch.setenv("PACT_CEREBRAS_MAX_RATE_LIMIT_WAIT_SECONDS", "7.5")

    client = CerebrasCompletionClient(sdk_client=FakeSDKClient([]))

    assert client.max_rate_limit_retries == 1
    assert client.max_rate_limit_wait_seconds == 7.5


def test_archived_rate_limit_environment_names_are_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PACT_CEREBRAS_MAX_RATE_LIMIT_RETRIES", raising=False)
    monkeypatch.delenv("PACT_CEREBRAS_MAX_RATE_LIMIT_WAIT_SECONDS", raising=False)
    monkeypatch.setenv("TRACK2_CEREBRAS_MAX_RATE_LIMIT_RETRIES", "9")
    monkeypatch.setenv("TRACK2_CEREBRAS_MAX_RATE_LIMIT_WAIT_SECONDS", "99")

    client = CerebrasCompletionClient(sdk_client=FakeSDKClient([]))

    assert (
        client.max_rate_limit_retries
        == client_module.DEFAULT_CEREBRAS_MAX_RATE_LIMIT_RETRIES
    )
    assert (
        client.max_rate_limit_wait_seconds
        == client_module.DEFAULT_CEREBRAS_MAX_RATE_LIMIT_WAIT_SECONDS
    )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("PACT_CEREBRAS_QUEUE_BACKOFF_SECONDS", "SENSITIVE_INVALID_NUMBER"),
        ("PACT_CEREBRAS_TOKEN_QUOTA_WINDOW_SECONDS", "nan"),
        ("PACT_CEREBRAS_PROACTIVE_TOKEN_PACING", "SENSITIVE_INVALID_BOOLEAN"),
    ],
)
def test_invalid_pacing_configuration_fails_without_echoing_value(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
) -> None:
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError) as caught:
        CerebrasCompletionClient(sdk_client=FakeSDKClient([]))

    assert name in str(caught.value)
    assert value not in str(caught.value)


def test_proactive_wait_also_respects_cumulative_wait_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PACT_CEREBRAS_MAX_RATE_LIMIT_RETRIES", "3")
    monkeypatch.setenv("PACT_CEREBRAS_MAX_RATE_LIMIT_WAIT_SECONDS", "1")
    monkeypatch.setenv("PACT_CEREBRAS_RATE_LIMIT_RETRY_BUFFER_SECONDS", "0")
    monkeypatch.setenv("PACT_CEREBRAS_PROACTIVE_TOKEN_PACING", "true")
    clock = FakeClock()
    install_clock(monkeypatch, clock)
    sdk = FakeSDKClient([])
    client = CerebrasCompletionClient(sdk_client=sdk)
    client._last_rate_limit_headers_by_model["gpt-oss-120b"] = (
        CerebrasRateLimitHeaders(
            remaining_tokens_minute=0,
            reset_tokens_minute_seconds=2,
        ),
        clock.perf_counter(),
    )

    with pytest.raises(CerebrasRateLimitBudgetExceededError) as caught:
        generate(client)

    assert caught.value.reason == "cumulative_wait_limit"
    assert caught.value.retries_completed == 0
    assert sdk.endpoint.calls == []
    assert clock.sleeps == []
