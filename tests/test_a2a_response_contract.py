import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from google.protobuf.json_format import MessageToDict
from litellm.completion_extras.litellm_responses_transformation.transformation import (
    LiteLLMResponsesTransformationHandler,
)
from litellm.main import responses_api_bridge_check

from a2a.helpers.proto_helpers import new_data_part, new_text_part
from agentbeats.sync_client import (
    build_send_message_jsonrpc_request,
    create_message_with_parts,
)
from evaluator.car_bench_evaluator import (
    _normalize_tool_arguments,
    _parse_tool_calls_data,
    _sum_successful_llm_time_seconds,
    _tool_parameter_schemas,
)
from car_bench.envs.base import Env
from car_bench.envs.tool_execution_error_evaluator import (
    tool_execution_errors_during_runtime,
)
from car_bench.types import Task, TaskType
from track_1_agent_under_test.car_bench_agent import (
    _assistant_content as track_1_assistant_content,
    _completion_kwargs as track_1_completion_kwargs,
    _is_missing_response_item_error,
    _remove_unencrypted_reasoning_items,
)
from track_2_agent_under_test_cerebras.car_bench_agent import (
    CARBenchAgentExecutor as CerebrasCARBenchAgentExecutor,
)
from track_2_agent_under_test_cerebras import cerebras_client as cerebras_client_module
from track_2_agent_under_test_cerebras.cerebras_client import (
    CerebrasCompletionClient,
    CerebrasRateLimitHeaders,
    TokenUsage,
    _exception_details,
    _extract_cerebras_rate_limit_signal,
    estimate_request_tokens,
    normalize_cerebras_api_base,
    normalize_cerebras_model,
)
from track_2_agent_under_test_cerebras_planner.planner_agent import (
    PlannerExecutorCARBenchAgentExecutor as CerebrasPlannerExecutor,
)
from turn_metrics import (
    AVG_LLM_CALL_TIME_MS,
    COMPLETION_TOKENS,
    COST,
    MODEL,
    NUM_LLM_CALLS,
    NUM_PASSES,
    PROMPT_TOKENS,
    QUOTA_WAIT_TIME_MS,
    THINKING_TOKENS,
)


class FakeRawResponse:
    def __init__(self, completion, headers: dict[str, str] | None = None) -> None:
        self._completion = completion
        self.headers = headers or {}

    def parse(self):
        return self._completion


class FakeCerebrasCreateEndpoint:
    def __init__(self, outcomes) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class FakeCerebrasSDKClient:
    def __init__(self, outcomes) -> None:
        self.create_endpoint = FakeCerebrasCreateEndpoint(outcomes)
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(
                with_raw_response=SimpleNamespace(
                    create=self.create_endpoint.create
                )
            )
        )


class FakeProviderResponse:
    def __init__(
        self,
        *,
        status_code: int,
        headers: dict[str, str],
        json_payload: dict,
        text: str | None = None,
    ) -> None:
        self.status_code = status_code
        self.headers = headers
        self.text = text if text is not None else json.dumps(json_payload)
        self._json_payload = json_payload

    def json(self):
        return self._json_payload


class FakeRateLimitError(Exception):
    def __init__(self, response: FakeProviderResponse) -> None:
        super().__init__("Cerebras rate limit")
        self.response = response
        self.status_code = response.status_code


class FakeLogger:
    def __init__(self) -> None:
        self.entries: list[tuple[str, str, dict]] = []

    def bind(self, **kwargs):
        return self

    def info(self, event: str, **kwargs) -> None:
        self.entries.append(("info", event, kwargs))

    def warning(self, event: str, **kwargs) -> None:
        self.entries.append(("warning", event, kwargs))

    def debug(self, event: str, **kwargs) -> None:
        self.entries.append(("debug", event, kwargs))


def fake_completion(
    *,
    content: str = '{"action":"respond","content":"Done.","tool_calls":[]}',
    model: str = "gpt-oss-120b",
    finish_reason: str = "stop",
):
    return SimpleNamespace(
        model=model,
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason=finish_reason,
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            prompt_tokens_details=SimpleNamespace(cached_tokens=30),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=7),
        ),
    )


def fake_task() -> Task:
    return Task(
        task_id="test_task",
        calendar_id="calendar",
        actions=[],
        persona="",
        instruction="",
        context_init_config={},
        task_type=TaskType.BASE,
    )


class A2AResponseContractTest(unittest.TestCase):
    def test_track_1_gpt_5_6_sol_chat_tools_disable_reasoning(self) -> None:
        kwargs = track_1_completion_kwargs(
            model="gpt-5.6-sol",
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "demo",
                        "description": "demo",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            temperature=None,
            thinking=False,
            reasoning_effort=None,
            interleaved_thinking=False,
        )

        self.assertEqual(kwargs["reasoning_effort"], "none")
        self.assertNotIn("temperature", kwargs)

    def test_track_1_other_requests_keep_reasoning_unspecified(self) -> None:
        for model, tools in (
            ("gpt-5.6-sol", []),
            (
                "another-provider/model",
                [
                    {
                        "type": "function",
                        "function": {
                            "name": "demo",
                            "description": "demo",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            ),
        ):
            with self.subTest(model=model, has_tools=bool(tools)):
                kwargs = track_1_completion_kwargs(
                    model=model,
                    tools=tools,
                    temperature=None,
                    thinking=False,
                    reasoning_effort=None,
                    interleaved_thinking=False,
                )
                self.assertNotIn("reasoning_effort", kwargs)

    def test_track_1_azure_responses_uses_stateless_encrypted_reasoning(self) -> None:
        kwargs = track_1_completion_kwargs(
            model="azure/gpt-5.6-sol",
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "demo",
                        "description": "demo",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            temperature=None,
            thinking=False,
            reasoning_effort=None,
            interleaved_thinking=False,
            api_mode="responses",
        )

        self.assertEqual(kwargs["model"], "azure/responses/gpt-5.6-sol")
        self.assertIs(kwargs["store"], False)
        self.assertEqual(kwargs["include"], ["reasoning.encrypted_content"])
        self.assertNotIn("temperature", kwargs)
        self.assertNotIn("reasoning_effort", kwargs)
        bridge, routed_model = responses_api_bridge_check(
            model="responses/gpt-5.6-sol",
            custom_llm_provider="azure",
        )
        self.assertEqual(bridge["mode"], "responses")
        self.assertEqual(routed_model, "gpt-5.6-sol")

    def test_track_1_merges_responses_bridge_output_items(self) -> None:
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        model_dump=lambda **_: {
                            "content": "Working.",
                            "reasoning_items": [
                                {
                                    "id": "rs_1",
                                    "type": "reasoning",
                                    "encrypted_content": "encrypted-trace",
                                }
                            ],
                        }
                    )
                ),
                SimpleNamespace(
                    message=SimpleNamespace(
                        model_dump=lambda **_: {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "demo",
                                        "arguments": "{}",
                                    },
                                }
                            ],
                        }
                    )
                ),
            ]
        )

        content = track_1_assistant_content(response)

        self.assertEqual(content["content"], "Working.")
        self.assertEqual(content["tool_calls"][0]["id"], "call_1")
        self.assertEqual(content["reasoning_items"][0]["id"], "rs_1")
        self.assertEqual(
            content["reasoning_items"][0]["encrypted_content"],
            "encrypted-trace",
        )

    def test_track_1_provider_default_responses_state_is_runtime_selectable(
        self,
    ) -> None:
        kwargs = track_1_completion_kwargs(
            model="azure/gpt-5.6-sol",
            tools=[],
            temperature=None,
            thinking=False,
            reasoning_effort=None,
            interleaved_thinking=False,
            api_mode="responses",
            responses_state_mode="provider-default",
        )

        self.assertNotIn("store", kwargs)
        self.assertNotIn("include", kwargs)

    def test_track_1_litellm_bridge_replays_encrypted_reasoning(self) -> None:
        request = LiteLLMResponsesTransformationHandler().transform_request(
            model="gpt-5.6-sol",
            messages=[
                {"role": "system", "content": "system"},
                {"role": "user", "content": "user"},
                {
                    "role": "assistant",
                    "content": None,
                    "reasoning_items": [
                        {
                            "id": "rs_portable",
                            "type": "reasoning",
                            "summary": [],
                            "encrypted_content": "encrypted-trace",
                        }
                    ],
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "demo",
                                "arguments": "{}",
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": "ok",
                },
            ],
            optional_params={
                "store": False,
                "include": ["reasoning.encrypted_content"],
            },
            litellm_params={},
            headers={},
            litellm_logging_obj=SimpleNamespace(),
        )

        self.assertIs(request["store"], False)
        self.assertEqual(
            request["include"],
            ["reasoning.encrypted_content"],
        )
        reasoning_item = next(
            item for item in request["input"] if item["type"] == "reasoning"
        )
        self.assertEqual(
            reasoning_item["encrypted_content"],
            "encrypted-trace",
        )

    def test_track_1_removes_only_unencrypted_reasoning_items(self) -> None:
        messages = [
            {
                "role": "assistant",
                "content": None,
                "reasoning_items": [
                    {"id": "rs_stale", "type": "reasoning"},
                    {
                        "id": "rs_portable",
                        "type": "reasoning",
                        "encrypted_content": "encrypted-trace",
                    },
                ],
            },
            {
                "role": "assistant",
                "content": None,
                "reasoning_items": [
                    {
                        "id": "rs_none",
                        "type": "reasoning",
                        "encrypted_content": None,
                    }
                ],
            },
        ]

        removed = _remove_unencrypted_reasoning_items(messages)

        self.assertEqual(removed, 2)
        self.assertEqual(
            messages[0]["reasoning_items"],
            [
                {
                    "id": "rs_portable",
                    "type": "reasoning",
                    "encrypted_content": "encrypted-trace",
                }
            ],
        )
        self.assertNotIn("reasoning_items", messages[1])

    def test_track_1_recognizes_missing_responses_item_error(self) -> None:
        self.assertTrue(
            _is_missing_response_item_error(
                RuntimeError(
                    "invalid_request_error: Item with id 'rs_stale' not found"
                )
            )
        )
        self.assertFalse(
            _is_missing_response_item_error(
                RuntimeError("invalid_request_error: malformed tool arguments")
            )
        )

    def test_sync_client_serializes_a2a_1_json_field_names(self) -> None:
        message = create_message_with_parts(
            parts=[new_text_part("hello")],
            context_id="ctx-1",
        )

        payload = build_send_message_jsonrpc_request(message)
        serialized_message = payload["params"]["message"]

        self.assertEqual(payload["method"], "SendMessage")
        self.assertIn("messageId", serialized_message)
        self.assertIn("contextId", serialized_message)
        self.assertNotIn("message_id", serialized_message)
        self.assertNotIn("context_id", serialized_message)
        self.assertEqual(serialized_message["parts"], [{"text": "hello"}])

    def test_cerebras_usage_parses_reasoning_tokens(self) -> None:
        usage = TokenUsage.from_provider_usage(fake_completion().usage)

        self.assertIsNotNone(usage)
        self.assertEqual(usage.input_tokens, 100)
        self.assertEqual(usage.cached_input_tokens, 30)
        self.assertEqual(usage.output_tokens, 20)
        self.assertEqual(usage.reasoning_output_tokens, 7)
        self.assertEqual(usage.total_tokens, 120)

    def test_cerebras_sdk_request_uses_strict_schema_and_normalized_model(self) -> None:
        fake_sdk = FakeCerebrasSDKClient(
            [
                FakeRawResponse(
                    fake_completion(),
                    headers={
                        "x-ratelimit-limit-requests-day": "10",
                        "x-ratelimit-limit-tokens-minute": "30000",
                        "x-ratelimit-remaining-requests-day": "9",
                        "x-ratelimit-remaining-tokens-minute": "1200",
                        "x-ratelimit-reset-requests-day": "3600",
                        "x-ratelimit-reset-tokens-minute": "12.5",
                    },
                )
            ]
        )
        client = CerebrasCompletionClient(
            service_tier="priority",
            sdk_client=fake_sdk,
        )

        result = client.generate(
            model="cerebras/gpt-oss-120b",
            messages=[{"role": "user", "content": "hello"}],
            response_schema={"type": "object", "additionalProperties": False},
            response_schema_name="next_action",
            max_completion_tokens=1024,
            temperature=0.0,
            reasoning_effort="medium",
        )

        kwargs = fake_sdk.create_endpoint.calls[0]
        self.assertEqual(kwargs["model"], "gpt-oss-120b")
        self.assertEqual(kwargs["service_tier"], "priority")
        self.assertEqual(kwargs["max_completion_tokens"], 1024)
        self.assertEqual(kwargs["reasoning_effort"], "medium")
        self.assertTrue(kwargs["response_format"]["json_schema"]["strict"])
        self.assertEqual(result.model, "gpt-oss-120b")
        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(result.token_usage.input_tokens, 100)
        self.assertEqual(result.rate_limit_headers.remaining_tokens_minute, 1200.0)

    def test_cerebras_executor_temperature_is_omitted_by_default(self) -> None:
        client = CerebrasCompletionClient(
            sdk_client=FakeCerebrasSDKClient([]),
        )

        kwargs = client._completion_kwargs(
            model="gpt-oss-120b",
            messages=[{"role": "user", "content": "hello"}],
            response_schema={"type": "object"},
            response_schema_name="next_action",
            max_completion_tokens=1024,
            temperature=None,
        )

        self.assertNotIn("temperature", kwargs)
        self.assertNotIn("reasoning_effort", kwargs)

    def test_cerebras_executor_temperature_can_be_set_explicitly(self) -> None:
        client = CerebrasCompletionClient(
            sdk_client=FakeCerebrasSDKClient([]),
        )

        kwargs = client._completion_kwargs(
            model="gpt-oss-120b",
            messages=[{"role": "user", "content": "hello"}],
            response_schema={"type": "object"},
            response_schema_name="next_action",
            max_completion_tokens=1024,
            temperature=0.0,
        )

        self.assertEqual(kwargs["temperature"], 0.0)

    def test_cerebras_reasoning_effort_can_be_set_explicitly(self) -> None:
        client = CerebrasCompletionClient(
            sdk_client=FakeCerebrasSDKClient([]),
        )

        kwargs = client._completion_kwargs(
            model="gpt-oss-120b",
            messages=[{"role": "user", "content": "hello"}],
            response_schema={"type": "object"},
            response_schema_name="next_action",
            max_completion_tokens=1024,
            temperature=None,
            reasoning_effort="high",
        )

        self.assertEqual(kwargs["reasoning_effort"], "high")

    def test_cerebras_model_prefix_is_normalized(self) -> None:
        self.assertEqual(
            normalize_cerebras_model("cerebras/gpt-oss-120b"),
            "gpt-oss-120b",
        )
        self.assertEqual(normalize_cerebras_model("gpt-oss-120b"), "gpt-oss-120b")

    def test_cerebras_sdk_base_url_strips_legacy_v1_suffix(self) -> None:
        self.assertEqual(
            normalize_cerebras_api_base("https://api.cerebras.ai/v1"),
            "https://api.cerebras.ai",
        )
        self.assertEqual(
            normalize_cerebras_api_base("https://api.cerebras.ai/v1/"),
            "https://api.cerebras.ai",
        )
        self.assertEqual(
            normalize_cerebras_api_base("https://api.cerebras.ai"),
            "https://api.cerebras.ai",
        )

    def test_cerebras_rate_limit_headers_parse_from_sdk_response(self) -> None:
        headers = CerebrasRateLimitHeaders.from_headers(
            {
                "x-ratelimit-limit-requests-day": "10",
                "x-ratelimit-limit-tokens-minute": "30000",
                "x-ratelimit-remaining-requests-day": "9",
                "x-ratelimit-remaining-tokens-minute": "1200",
                "x-ratelimit-reset-requests-day": "3600",
                "x-ratelimit-reset-tokens-minute": "12.5",
            }
        )

        self.assertIsNotNone(headers)
        self.assertEqual(headers.limit_requests_day, 10.0)
        self.assertEqual(headers.limit_tokens_minute, 30000.0)
        self.assertEqual(headers.remaining_requests_day, 9.0)
        self.assertEqual(headers.remaining_tokens_minute, 1200.0)
        self.assertEqual(headers.reset_requests_day_seconds, 3600.0)
        self.assertEqual(headers.reset_tokens_minute_seconds, 12.5)

    def test_cerebras_estimates_prompt_plus_completion_budget(self) -> None:
        estimated = estimate_request_tokens(
            messages=[{"role": "user", "content": "abcd"}],
            max_completion_tokens=10,
            chars_per_token=1000,
            safety_factor=1.0,
        )

        self.assertEqual(estimated, 11)

    def test_cerebras_record_attempt_reports_previous_estimate_delta(self) -> None:
        client = CerebrasCompletionClient(sdk_client=FakeCerebrasSDKClient([]))
        client._record_attempt(model="gpt-oss-120b", estimated_tokens=100)

        state = client._record_attempt(
            model="gpt-oss-120b",
            estimated_tokens=140,
        )

        self.assertEqual(state["previous_estimated_request_tokens"], 100)
        self.assertEqual(
            state["estimated_request_token_delta_since_previous"],
            40,
        )

    def test_cerebras_queue_exceeded_signal_uses_jittered_backoff(self) -> None:
        details = {
            "status_code": 429,
            "provider_error_body": {
                "message": "We're experiencing high traffic right now!",
                "type": "too_many_requests_error",
                "param": "queue",
                "code": "queue_exceeded",
            },
            "provider_error_headers": {"x-should-retry": "false"},
        }
        original_uniform = cerebras_client_module.random.uniform
        cerebras_client_module.random.uniform = lambda left, right: (
            left + right
        ) / 2.0
        try:
            first_signal = _extract_cerebras_rate_limit_signal(
                details,
                queue_backoff_seconds=60.0,
                queue_retry_attempt=1,
                retry_buffer_seconds=1.0,
            )
            second_signal = _extract_cerebras_rate_limit_signal(
                details,
                queue_backoff_seconds=60.0,
                queue_retry_attempt=2,
                retry_buffer_seconds=1.0,
            )
            third_signal = _extract_cerebras_rate_limit_signal(
                details,
                queue_backoff_seconds=60.0,
                queue_retry_attempt=3,
                retry_buffer_seconds=1.0,
            )
        finally:
            cerebras_client_module.random.uniform = original_uniform

        self.assertIsNotNone(first_signal)
        self.assertEqual(first_signal.code, "queue_exceeded")
        self.assertEqual(first_signal.source, "sdk")
        self.assertEqual(first_signal.schedule_wait_seconds, 60.0)
        self.assertEqual(
            first_signal.schedule_reason,
            "sdk_queue_exceeded_jitter_backoff",
        )
        self.assertEqual(first_signal.queue_retry_attempt, 1)
        self.assertEqual(first_signal.queue_backoff_min_seconds, 54.0)
        self.assertEqual(first_signal.queue_backoff_max_seconds, 66.0)
        self.assertFalse(first_signal.quota_wait_eligible)
        self.assertEqual(first_signal.x_should_retry, "false")

        self.assertIsNotNone(second_signal)
        self.assertEqual(second_signal.schedule_wait_seconds, 105.0)
        self.assertEqual(second_signal.queue_retry_attempt, 2)
        self.assertEqual(second_signal.queue_backoff_min_seconds, 90.0)
        self.assertEqual(second_signal.queue_backoff_max_seconds, 120.0)

        self.assertIsNotNone(third_signal)
        self.assertEqual(third_signal.schedule_wait_seconds, 240.0)
        self.assertEqual(third_signal.queue_retry_attempt, 3)
        self.assertEqual(third_signal.queue_backoff_min_seconds, 180.0)
        self.assertEqual(third_signal.queue_backoff_max_seconds, 300.0)

    def test_cerebras_queue_exceeded_respects_retry_after(self) -> None:
        signal = _extract_cerebras_rate_limit_signal(
            {
                "status_code": 429,
                "provider_error_body": {
                    "message": "We're experiencing high traffic right now!",
                    "type": "too_many_requests_error",
                    "param": "queue",
                    "code": "queue_exceeded",
                },
                "provider_error_headers": {
                    "retry-after": "75",
                    "x-should-retry": "false",
                },
            },
            queue_backoff_seconds=60.0,
            queue_retry_attempt=3,
            retry_buffer_seconds=1.0,
        )

        self.assertIsNotNone(signal)
        self.assertEqual(signal.schedule_wait_seconds, 76.0)
        self.assertEqual(signal.schedule_reason, "sdk_queue_retry_after")
        self.assertEqual(signal.wait_source_header, "retry-after")
        self.assertEqual(signal.queue_retry_attempt, 3)
        self.assertIsNone(signal.queue_backoff_min_seconds)
        self.assertIsNone(signal.queue_backoff_max_seconds)
        self.assertFalse(signal.quota_wait_eligible)

    def test_cerebras_quota_signal_uses_reset_header(self) -> None:
        signal = _extract_cerebras_rate_limit_signal(
            {
                "status_code": 429,
                "provider_error_body": {
                    "message": "Tokens per minute limit exceeded",
                    "type": "too_many_tokens_error",
                    "param": "quota",
                    "code": "token_quota_exceeded",
                },
                "provider_error_headers": {
                    "x-ratelimit-reset-tokens-minute": "60",
                },
            },
            queue_backoff_seconds=0.0,
            retry_buffer_seconds=1.0,
        )

        self.assertIsNotNone(signal)
        self.assertEqual(signal.code, "token_quota_exceeded")
        self.assertEqual(signal.schedule_wait_seconds, 61.0)
        self.assertEqual(signal.schedule_reason, "sdk_headers_tokens_minute_reset")
        self.assertEqual(signal.wait_source_header, "x-ratelimit-reset-tokens-minute")
        self.assertEqual(signal.reset_tokens_minute_seconds, 60.0)
        self.assertEqual(signal.retry_after_seconds, None)
        self.assertEqual(signal.missing_expected_headers, ())
        self.assertTrue(signal.quota_wait_eligible)

    def test_cerebras_quota_signal_prefers_reset_header_over_retry_after(self) -> None:
        signal = _extract_cerebras_rate_limit_signal(
            {
                "status_code": 429,
                "provider_error_body": {
                    "type": "too_many_tokens_error",
                    "param": "quota",
                    "code": "token_quota_exceeded",
                },
                "provider_error_headers": {
                    "retry-after": "59",
                    "x-ratelimit-reset-tokens-minute": "12",
                },
            },
            queue_backoff_seconds=0.0,
            retry_buffer_seconds=1.0,
        )

        self.assertIsNotNone(signal)
        self.assertEqual(signal.schedule_wait_seconds, 13.0)
        self.assertEqual(signal.schedule_reason, "sdk_headers_tokens_minute_reset")
        self.assertEqual(signal.wait_source_header, "x-ratelimit-reset-tokens-minute")
        self.assertEqual(signal.retry_after_seconds, 59.0)

    def test_cerebras_quota_signal_falls_back_to_retry_after(self) -> None:
        signal = _extract_cerebras_rate_limit_signal(
            {
                "status_code": 429,
                "provider_error_body": {
                    "type": "too_many_tokens_error",
                    "param": "quota",
                    "code": "token_quota_exceeded",
                },
                "provider_error_headers": {
                    "retry-after": "59",
                },
            },
            queue_backoff_seconds=0.0,
            retry_buffer_seconds=1.0,
        )

        self.assertIsNotNone(signal)
        self.assertEqual(signal.schedule_wait_seconds, 60.0)
        self.assertEqual(signal.schedule_reason, "sdk_retry_after")
        self.assertEqual(signal.wait_source_header, "retry-after")
        self.assertEqual(
            signal.missing_expected_headers,
            ("x-ratelimit-reset-tokens-minute",),
        )
        self.assertTrue(signal.quota_wait_eligible)

    def test_cerebras_generate_logs_reactive_wait_from_token_reset_header(self) -> None:
        logger = FakeLogger()
        rate_limit_error = FakeRateLimitError(
            FakeProviderResponse(
                status_code=429,
                headers={
                    "retry-after": "59",
                    "x-ratelimit-reset-tokens-minute": "12",
                },
                json_payload={
                    "message": "Tokens per minute limit exceeded",
                    "type": "too_many_tokens_error",
                    "param": "quota",
                    "code": "token_quota_exceeded",
                },
            )
        )
        fake_sdk = FakeCerebrasSDKClient(
            [
                rate_limit_error,
                FakeRawResponse(fake_completion()),
            ]
        )
        client = CerebrasCompletionClient(
            sdk_client=fake_sdk,
            logger=logger,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            client.rate_limit_report_dir = Path(tmpdir)
            original_sleep = cerebras_client_module.time.sleep
            cerebras_client_module.time.sleep = lambda _: None
            try:
                result = client.generate(
                    model="gpt-oss-120b",
                    messages=[{"role": "user", "content": "hello"}],
                    response_schema={"type": "object"},
                    response_schema_name="next_action",
                    max_completion_tokens=1024,
                    temperature=0.0,
                )
            finally:
                cerebras_client_module.time.sleep = original_sleep

        wait_logs = [
            (message, kwargs)
            for level, message, kwargs in logger.entries
            if level == "warning"
            and message.startswith("Cerebras rate limit reached; waiting")
        ]
        self.assertEqual(len(wait_logs), 1)
        wait_message, wait_fields = wait_logs[0]
        self.assertIn("waiting 13.000s before retry", wait_message)
        self.assertIn("reason=sdk_headers_tokens_minute_reset", wait_message)
        self.assertIn(
            "source_header=x-ratelimit-reset-tokens-minute",
            wait_message,
        )
        self.assertIn("reset_tokens_minute=12.000s", wait_message)
        self.assertIn("retry_after=59.000s", wait_message)
        self.assertEqual(wait_fields["wait_seconds"], 13.0)
        self.assertEqual(wait_fields["wait_reason"], "sdk_headers_tokens_minute_reset")
        self.assertEqual(
            wait_fields["wait_source_header"],
            "x-ratelimit-reset-tokens-minute",
        )
        self.assertEqual(wait_fields["retry_after_seconds"], 59.0)
        self.assertEqual(wait_fields["reset_tokens_minute_seconds"], 12.0)
        self.assertEqual(wait_fields["missing_expected_headers"], [])
        self.assertEqual(len(fake_sdk.create_endpoint.calls), 2)
        self.assertEqual(result.quota_wait_ms, 13000.0)

    def test_cerebras_exception_details_include_sdk_response_payload(self) -> None:
        exc = FakeRateLimitError(
            FakeProviderResponse(
                status_code=429,
                headers={"x-ratelimit-reset-tokens-minute": "60"},
                json_payload={
                    "error": {
                        "message": "Tokens per minute limit exceeded",
                        "type": "too_many_tokens_error",
                        "param": "quota",
                        "code": "token_quota_exceeded",
                    }
                },
            )
        )

        details = _exception_details(exc)

        self.assertEqual(details["status_code"], 429)
        self.assertEqual(
            details["provider_error_body"]["code"],
            "token_quota_exceeded",
        )
        self.assertEqual(
            details["provider_error_headers"]["x-ratelimit-reset-tokens-minute"],
            "60",
        )

    def test_cerebras_rate_limit_report_writes_current_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            client = CerebrasCompletionClient(sdk_client=FakeCerebrasSDKClient([]))
            client.rate_limit_report_dir = Path(tmpdir)
            client._record_attempt(model="gpt-oss-120b", estimated_tokens=1234)
            client._record_successful_call(
                model="gpt-oss-120b",
                token_usage=TokenUsage(
                    input_tokens=1000,
                    cached_input_tokens=250,
                    output_tokens=100,
                    reasoning_output_tokens=40,
                    total_tokens=1140,
                ),
                rate_limit_headers=CerebrasRateLimitHeaders(
                    remaining_tokens_minute=5000,
                    reset_tokens_minute_seconds=12,
                ),
            )

            signal = _extract_cerebras_rate_limit_signal(
                {
                    "status_code": 429,
                    "provider_error_body": {
                        "code": "token_quota_exceeded",
                        "param": "quota",
                    },
                    "provider_error_headers": {
                        "x-ratelimit-reset-tokens-minute": "60",
                    },
                },
                queue_backoff_seconds=60.0,
                retry_buffer_seconds=1.0,
            )
            path = client._write_rate_limit_report(
                model="gpt-oss-120b",
                messages=[{"role": "user", "content": "hello"}],
                response_schema={"type": "object"},
                response_schema_name="next_action",
                max_completion_tokens=1024,
                estimated_tokens=1234,
                duration_ms=61115.6,
                error_details={
                    "status_code": 429,
                    "provider_error_body": {
                        "code": "token_quota_exceeded",
                        "param": "quota",
                    },
                    "provider_error_headers": {
                        "x-ratelimit-reset-tokens-minute": "60",
                    },
                },
                rate_limit_signal=signal,
            )

            self.assertIsNotNone(path)
            payload = json.loads(path.read_text())
            self.assertEqual(payload["event"], "cerebras_rate_limit")
            self.assertEqual(payload["model"], "gpt-oss-120b")
            self.assertEqual(payload["successful_cerebras_calls"], 1)
            self.assertEqual(payload["attempted_cerebras_calls"], 1)
            self.assertEqual(payload["tokens_consumed"]["input_tokens"], 1000)
            self.assertEqual(payload["tokens_consumed"]["output_tokens"], 100)
            self.assertEqual(
                payload["tokens_consumed_since_previous_rate_limit"][
                    "input_tokens"
                ],
                1000,
            )
            self.assertEqual(payload["estimated_request_tokens_attempted"], 1234)
            self.assertEqual(
                payload[
                    "estimated_request_tokens_attempted_since_previous_rate_limit"
                ],
                1234,
            )
            self.assertEqual(
                payload["current_call"]["duration_ms_until_error"],
                61115.6,
            )
            self.assertEqual(
                payload["rate_limit_signal"]["code"],
                "token_quota_exceeded",
            )
            self.assertEqual(payload["wait_seconds"], 61.0)
            self.assertTrue(payload["rate_limit_signal"]["quota_wait_eligible"])
            self.assertNotIn("scheduler_config", payload)
            self.assertNotIn("scheduler_state", payload)
            self.assertIn("last_rate_limit_headers_by_model", payload)

    def test_evaluator_successful_llm_time_uses_agent_state_latency(self) -> None:
        result = SimpleNamespace(
            info={"total_llm_induced_latency_ms": 1234.5},
        )

        value = _sum_successful_llm_time_seconds({"base": [result]})

        self.assertEqual(value, 1.2345)

    def test_a2a_data_part_decodes_integer_numbers_as_floats(self) -> None:
        part = new_data_part(
            {
                "tool_calls": [
                    {
                        "tool_name": "calculate_charging_soc_by_time",
                        "arguments": {
                            "start_state_of_charge": 20,
                            "charging_time": 40,
                        },
                    }
                ]
            }
        )

        data = MessageToDict(part.data)
        arguments = data["tool_calls"][0]["arguments"]

        self.assertEqual(arguments["charging_time"], 40.0)
        self.assertIs(type(arguments["charging_time"]), float)
        self.assertEqual(arguments["start_state_of_charge"], 20.0)
        self.assertIs(type(arguments["start_state_of_charge"]), float)

    def test_evaluator_normalizes_integral_float_tool_arguments(self) -> None:
        schemas = _tool_parameter_schemas(
            [
                {
                    "function": {
                        "name": "calculate_charging_soc_by_time",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "start_state_of_charge": {"type": "integer"},
                                "charging_time": {"type": "integer"},
                            },
                        },
                    }
                }
            ]
        )

        normalized = _normalize_tool_arguments(
            "calculate_charging_soc_by_time",
            {
                "start_state_of_charge": 20.0,
                "charging_time": 40.0,
                "unknown_numeric_argument": 1.0,
            },
            schemas,
        )

        self.assertEqual(normalized["start_state_of_charge"], 20)
        self.assertIs(type(normalized["start_state_of_charge"]), int)
        self.assertEqual(normalized["charging_time"], 40)
        self.assertIs(type(normalized["charging_time"]), int)
        self.assertEqual(normalized["unknown_numeric_argument"], 1.0)
        self.assertIs(type(normalized["unknown_numeric_argument"]), float)

    def test_evaluator_normalizes_nested_planning_tool_integer_arguments(self) -> None:
        schemas = _tool_parameter_schemas(
            [
                {
                    "function": {
                        "name": "planning_tool",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "steps": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "step_dependent_on": {
                                                "type": "array",
                                                "items": {"type": "integer"},
                                            }
                                        },
                                    },
                                },
                                "step_updates": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "step_index": {"type": "integer"}
                                        },
                                    },
                                },
                            },
                        },
                    }
                }
            ]
        )

        normalized = _normalize_tool_arguments(
            "planning_tool",
            {
                "steps": [
                    {
                        "step_description": "a",
                        "step_dependent_on": [0.0],
                    }
                ],
                "step_updates": [{"step_index": 0.0}],
            },
            schemas,
        )

        dependency = normalized["steps"][0]["step_dependent_on"][0]
        step_index = normalized["step_updates"][0]["step_index"]
        self.assertEqual(dependency, 0)
        self.assertIs(type(dependency), int)
        self.assertEqual(step_index, 0)
        self.assertIs(type(step_index), int)

    def test_evaluator_preserves_fractional_floats_and_bools(self) -> None:
        schemas = {
            "test_tool": {
                "type": "object",
                "properties": {
                    "integer_value": {"type": "integer"},
                    "boolean_value": {"type": "integer"},
                },
            }
        }

        normalized = _normalize_tool_arguments(
            "test_tool",
            {"integer_value": 40.5, "boolean_value": True},
            schemas,
        )

        self.assertEqual(normalized["integer_value"], 40.5)
        self.assertIs(type(normalized["integer_value"]), float)
        self.assertIs(normalized["boolean_value"], True)

    def test_parse_tool_calls_serializes_normalized_integer_arguments(self) -> None:
        schemas = {
            "calculate_charging_soc_by_time": {
                "type": "object",
                "properties": {
                    "charging_time": {"type": "integer"},
                },
            }
        }

        parsed = _parse_tool_calls_data(
            [
                {
                    "tool_name": "calculate_charging_soc_by_time",
                    "arguments": {"charging_time": 40.0},
                }
            ],
            schemas,
        )

        arguments = json.loads(parsed[0]["function"]["arguments"])
        self.assertEqual(arguments["charging_time"], 40)
        self.assertIs(type(arguments["charging_time"]), int)

    def test_generic_tool_exception_is_recorded_in_tool_execution_errors(self) -> None:
        class FailingTool:
            @staticmethod
            def invoke(**kwargs):
                raise TypeError("boom")

        env = Env.__new__(Env)
        env.actions = []
        env.data = {}
        env.task = fake_task()
        env.tools_map = {"failing_tool": FailingTool}

        token = tool_execution_errors_during_runtime.set([])
        try:
            response = env.step(
                SimpleNamespace(name="failing_tool", kwargs={}),
                [],
            )
            errors = tool_execution_errors_during_runtime.get()
        finally:
            tool_execution_errors_during_runtime.reset(token)

        self.assertEqual(response.observation, "Error: boom")
        self.assertEqual(errors, ["failing_tool: TypeError: boom"])

    def test_async_generic_tool_exception_is_recorded_in_tool_execution_errors(self) -> None:
        class FailingTool:
            @staticmethod
            def invoke(**kwargs):
                raise TypeError("boom")

        env = Env.__new__(Env)
        env.terminate_tools = []
        env.task = fake_task()

        token = tool_execution_errors_during_runtime.set([])
        try:
            result = asyncio.run(
                env._run_action(
                    SimpleNamespace(name="failing_tool", kwargs={}),
                    {"failing_tool": FailingTool},
                    {},
                )
            )
            errors = tool_execution_errors_during_runtime.get()
        finally:
            tool_execution_errors_during_runtime.reset(token)

        self.assertEqual(result["observation"], "Error: boom")
        self.assertEqual(errors, ["failing_tool: TypeError: boom"])

    def test_explicit_tool_execution_errors_are_preserved(self) -> None:
        class ExplicitFailureTool:
            @staticmethod
            def invoke(**kwargs):
                tool_execution_errors_during_runtime.get().append("explicit failure")
                return '{"status":"FAILURE"}'

        env = Env.__new__(Env)
        env.actions = []
        env.data = {}
        env.task = fake_task()
        env.tools_map = {"explicit_failure_tool": ExplicitFailureTool}

        token = tool_execution_errors_during_runtime.set([])
        try:
            response = env.step(
                SimpleNamespace(name="explicit_failure_tool", kwargs={}),
                [],
            )
            errors = tool_execution_errors_during_runtime.get()
        finally:
            tool_execution_errors_during_runtime.reset(token)

        self.assertEqual(response.observation, '{"status":"FAILURE"}')
        self.assertEqual(errors, ["explicit failure"])

    def test_cerebras_respond_action_returns_text_part(self) -> None:
        parts, history_message = CerebrasCARBenchAgentExecutor._build_a2a_response_parts(
            {
                "action": "respond",
                "content": "Done.",
                "tool_calls": [],
            }
        )

        self.assertEqual(len(parts), 1)
        self.assertEqual(parts[0].WhichOneof("content"), "text")
        self.assertEqual(parts[0].text, "Done.")
        self.assertEqual(history_message, {"role": "assistant", "content": "Done."})

    def test_cerebras_tool_action_returns_tool_calls_data_part(self) -> None:
        parts, history_message = CerebrasCARBenchAgentExecutor._build_a2a_response_parts(
            {
                "action": "tool_calls",
                "content": "",
                "tool_calls": [
                    {
                        "tool_name": "open_close_sunshade",
                        "arguments": {"percentage": 50},
                    }
                ],
            }
        )

        self.assertEqual(len(parts), 1)
        self.assertEqual(parts[0].WhichOneof("content"), "data")
        data = MessageToDict(parts[0].data)
        self.assertEqual(
            data,
            {
                "tool_calls": [
                    {
                        "tool_name": "open_close_sunshade",
                        "arguments": {"percentage": 50.0},
                    }
                ]
            },
        )
        percentage = data["tool_calls"][0]["arguments"]["percentage"]
        self.assertIs(type(percentage), float)
        self.assertIsNone(history_message["content"])
        self.assertEqual(
            history_message["tool_calls"][0]["function"]["name"],
            "open_close_sunshade",
        )

    def test_cerebras_turn_metrics_are_public_metadata_shape(self) -> None:
        executor = CerebrasCARBenchAgentExecutor(model="gpt-oss-120b")

        executor._record_turn_metrics(
            "ctx",
            100.0,
            token_usage=TokenUsage(
                input_tokens=1200,
                cached_input_tokens=400,
                output_tokens=80,
                reasoning_output_tokens=25,
                total_tokens=1305,
            ),
            cost=0.25,
        )
        metrics = executor._public_turn_metrics(
            executor.ctx_id_to_turn_metrics.pop("ctx")
        )

        self.assertEqual(metrics[PROMPT_TOKENS], 1200)
        self.assertEqual(metrics[COMPLETION_TOKENS], 80)
        self.assertEqual(metrics[THINKING_TOKENS], 25)
        self.assertEqual(metrics[COST], 0.25)
        self.assertEqual(metrics[MODEL], "gpt-oss-120b")
        self.assertEqual(metrics[NUM_LLM_CALLS], 1)
        self.assertEqual(metrics[AVG_LLM_CALL_TIME_MS], 100.0)
        self.assertEqual(metrics[NUM_PASSES], 1)
        self.assertEqual(metrics[QUOTA_WAIT_TIME_MS], 0.0)
        self.assertNotIn("_total_llm_time_ms", metrics)

    def test_cerebras_planner_executor_metrics_report_internal_passes(self) -> None:
        executor = CerebrasPlannerExecutor(
            planner_model="gpt-oss-120b",
            executor_model="gpt-oss-120b",
        )
        executor._last_internal_call_count = 2

        executor._record_turn_metrics(
            "ctx",
            300.0,
            token_usage=TokenUsage(
                input_tokens=3000,
                output_tokens=200,
                reasoning_output_tokens=75,
                total_tokens=3275,
            ),
            cost=0.5,
        )
        metrics = executor._public_turn_metrics(
            executor.ctx_id_to_turn_metrics.pop("ctx")
        )

        self.assertEqual(metrics[MODEL], "gpt-oss-120b->gpt-oss-120b")
        self.assertEqual(metrics[PROMPT_TOKENS], 3000)
        self.assertEqual(metrics[COMPLETION_TOKENS], 200)
        self.assertEqual(metrics[THINKING_TOKENS], 75)
        self.assertEqual(metrics[COST], 0.5)
        self.assertEqual(metrics[NUM_LLM_CALLS], 2)
        self.assertEqual(metrics[AVG_LLM_CALL_TIME_MS], 150.0)
        self.assertEqual(metrics[NUM_PASSES], 2)
        self.assertEqual(metrics[QUOTA_WAIT_TIME_MS], 0.0)

    def test_cerebras_planner_executor_defaults_to_reasoning_split(self) -> None:
        executor = CerebrasPlannerExecutor()

        self.assertEqual(executor.planner_model, "gpt-oss-120b")
        self.assertEqual(executor.executor_model, "gpt-oss-120b")
        self.assertEqual(executor.planner_reasoning_effort, "high")
        self.assertEqual(executor.executor_reasoning_effort, "medium")


if __name__ == "__main__":
    unittest.main()
