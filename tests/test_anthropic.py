"""OpenAI <-> Anthropic Messages translation, including stream event mapping."""

from __future__ import annotations

import json

import httpx
import pytest

from llm_gateway.config import ProviderConfig
from llm_gateway.errors import ProviderError
from llm_gateway.providers.anthropic import (
    AnthropicProvider,
    StreamTranslator,
    TranslationError,
    from_anthropic_response,
    to_anthropic_request,
)
from llm_gateway.schemas import ChatCompletionRequest


def req(**kwargs) -> ChatCompletionRequest:
    kwargs.setdefault("model", "claude")
    kwargs.setdefault("messages", [{"role": "user", "content": "hi"}])
    return ChatCompletionRequest.model_validate(kwargs)


def translate(**kwargs) -> dict:
    return to_anthropic_request(
        req(**kwargs), "claude-haiku-4-5", default_max_tokens=1024, stream=False
    )


# -- request ------------------------------------------------------------------------------


def test_system_and_developer_messages_become_top_level_system() -> None:
    payload = translate(
        messages=[
            {"role": "system", "content": "You are a support agent for a Dhaka e-shop."},
            {"role": "developer", "content": "Answer in one sentence."},
            {"role": "user", "content": "Where is my order?"},
        ]
    )
    assert (
        payload["system"]
        == "You are a support agent for a Dhaka e-shop.\n\nAnswer in one sentence."
    )
    assert payload["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "Where is my order?"}]}
    ]
    assert payload["model"] == "claude-haiku-4-5"


def test_consecutive_same_role_turns_are_merged() -> None:
    payload = translate(
        messages=[
            {"role": "user", "content": "first"},
            {"role": "user", "content": [{"type": "text", "text": "second"}]},
            {"role": "assistant", "content": "ok"},
        ]
    )
    assert [t["role"] for t in payload["messages"]] == ["user", "assistant"]
    assert [b["text"] for b in payload["messages"][0]["content"]] == ["first", "second"]


def test_tool_calls_and_tool_results_round_trip() -> None:
    payload = translate(
        messages=[
            {"role": "user", "content": "Track order 1042"},
            {
                "role": "assistant",
                "content": "Checking.",
                "tool_calls": [
                    {
                        "id": "toolu_01",
                        "type": "function",
                        "function": {"name": "track_order", "arguments": '{"order_id": 1042}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "toolu_01", "content": '{"status": "shipped"}'},
            {"role": "tool", "tool_call_id": "toolu_02", "content": "second result"},
        ]
    )
    assistant, results = payload["messages"][1], payload["messages"][2]
    assert assistant["content"] == [
        {"type": "text", "text": "Checking."},
        {"type": "tool_use", "id": "toolu_01", "name": "track_order", "input": {"order_id": 1042}},
    ]
    assert results["role"] == "user"
    assert [b["tool_use_id"] for b in results["content"]] == ["toolu_01", "toolu_02"]
    assert results["content"][0] == {
        "type": "tool_result",
        "tool_use_id": "toolu_01",
        "content": '{"status": "shipped"}',
    }


def test_invalid_tool_arguments_are_a_client_error() -> None:
    with pytest.raises(TranslationError) as info:
        translate(
            messages=[
                {"role": "user", "content": "x"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "t1",
                            "type": "function",
                            "function": {"name": "f", "arguments": "{oops"},
                        }
                    ],
                },
            ]
        )
    assert info.value.fallback is False and info.value.status == 400


def test_images_become_image_blocks() -> None:
    payload = translate(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is on this receipt?"},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,QUJD"}},
                    {"type": "image_url", "image_url": {"url": "https://cdn.example.com/r.png"}},
                ],
            }
        ]
    )
    blocks = payload["messages"][0]["content"]
    assert blocks[1] == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/jpeg", "data": "QUJD"},
    }
    assert blocks[2] == {
        "type": "image",
        "source": {"type": "url", "url": "https://cdn.example.com/r.png"},
    }


def test_sampling_parameters_and_metadata() -> None:
    payload = translate(temperature=1.6, top_p=0.9, stop="END", user="customer-42")
    assert payload["max_tokens"] == 1024  # required upstream, filled from provider default
    assert payload["temperature"] == 1.0  # clamped to Anthropic's 0..1 range
    assert payload["top_p"] == 0.9
    assert payload["stop_sequences"] == ["END"]
    assert payload["metadata"] == {"user_id": "customer-42"}
    assert translate(max_completion_tokens=50)["max_tokens"] == 50
    assert translate(max_tokens=70)["max_tokens"] == 70


def test_tools_translate_to_input_schema() -> None:
    schema = {
        "type": "object",
        "properties": {"order_id": {"type": "integer"}},
        "required": ["order_id"],
    }
    payload = translate(
        tools=[
            {
                "type": "function",
                "function": {"name": "track_order", "description": "Track", "parameters": schema},
            }
        ]
    )
    assert payload["tools"] == [
        {"name": "track_order", "description": "Track", "input_schema": schema}
    ]


@pytest.mark.parametrize(
    ("choice", "expected"),
    [
        ("auto", {"type": "auto"}),
        ("required", {"type": "any"}),
        ("none", {"type": "none"}),
        (
            {"type": "function", "function": {"name": "track_order"}},
            {"type": "tool", "name": "track_order"},
        ),
    ],
)
def test_tool_choice_mapping(choice, expected) -> None:
    assert translate(tool_choice=choice)["tool_choice"] == expected


# -- response -----------------------------------------------------------------------------


def test_text_response_and_usage_with_cache_tokens() -> None:
    response = from_anthropic_response(
        {
            "id": "msg_01",
            "model": "claude-haiku-4-5",
            "content": [
                {"type": "text", "text": "Your order "},
                {"type": "text", "text": "has shipped."},
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 20, "cache_read_input_tokens": 100, "output_tokens": 7},
        }
    )
    assert response.choices[0].message.content == "Your order has shipped."
    assert response.choices[0].finish_reason == "stop"
    assert response.usage.model_dump() == {
        "prompt_tokens": 120,
        "completion_tokens": 7,
        "total_tokens": 127,
    }


def test_tool_use_response_becomes_tool_calls() -> None:
    response = from_anthropic_response(
        {
            "id": "msg_02",
            "model": "claude-haiku-4-5",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_9",
                    "name": "track_order",
                    "input": {"order_id": 7},
                }
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 5, "output_tokens": 3},
        }
    )
    message = response.choices[0].message
    assert message.content is None
    assert message.tool_calls[0].id == "toolu_9"
    assert json.loads(message.tool_calls[0].function.arguments) == {"order_id": 7}
    assert response.choices[0].finish_reason == "tool_calls"


@pytest.mark.parametrize(
    ("reason", "finish"),
    [
        ("end_turn", "stop"),
        ("stop_sequence", "stop"),
        ("max_tokens", "length"),
        ("refusal", "content_filter"),
    ],
)
def test_stop_reason_mapping(reason: str, finish: str) -> None:
    data = {
        "id": "m",
        "model": "x",
        "content": [{"type": "text", "text": "a"}],
        "stop_reason": reason,
        "usage": {},
    }
    assert from_anthropic_response(data).choices[0].finish_reason == finish


# -- streaming ----------------------------------------------------------------------------

TEXT_EVENTS = [
    (
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": "msg_s1",
                "model": "claude-haiku-4-5",
                "usage": {"input_tokens": 25, "output_tokens": 1},
            },
        },
    ),
    (
        "content_block_start",
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    ),
    ("ping", {"type": "ping"}),
    (
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hello"},
        },
    ),
    (
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": " world"},
        },
    ),
    ("content_block_stop", {"type": "content_block_stop", "index": 0}),
    (
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 12},
        },
    ),
    ("message_stop", {"type": "message_stop"}),
]


def test_stream_text_events_map_to_openai_chunks() -> None:
    translator = StreamTranslator()
    chunks = [c for event, data in TEXT_EVENTS for c in translator.feed(event, data)]
    assert chunks[0].choices[0].delta.role == "assistant"
    assert "".join(c.choices[0].delta.content or "" for c in chunks if c.choices) == "Hello world"
    finish = [
        c.choices[0].finish_reason for c in chunks if c.choices and c.choices[0].finish_reason
    ]
    assert finish == ["stop"]
    assert chunks[-1].choices == [] and chunks[-1].usage.model_dump() == {
        "prompt_tokens": 25,
        "completion_tokens": 12,
        "total_tokens": 37,
    }
    assert {c.id for c in chunks} == {"msg_s1"} and translator.done


def test_stream_tool_use_events_map_to_tool_call_deltas() -> None:
    events = [
        {
            "type": "message_start",
            "message": {"id": "msg_t", "model": "m", "usage": {"input_tokens": 9}},
        },
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Let me check."},
        },
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_A",
                "name": "track_order",
                "input": {},
            },
        },
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": '{"order_'},
        },
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": 'id": 5}'},
        },
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use"},
            "usage": {"output_tokens": 30},
        },
        {"type": "message_stop"},
    ]
    translator = StreamTranslator()
    chunks = [c for e in events for c in translator.feed(None, e)]
    calls = [tc for c in chunks if c.choices for tc in c.choices[0].delta.tool_calls or []]
    assert (
        calls[0].index == 0
        and calls[0].id == "toolu_A"
        and calls[0].function["name"] == "track_order"
    )
    assert "".join(tc.function.get("arguments", "") for tc in calls) == '{"order_id": 5}'
    assert any(c.choices and c.choices[0].finish_reason == "tool_calls" for c in chunks)


def test_stream_error_event_raises_retryable_for_overload() -> None:
    translator = StreamTranslator()
    with pytest.raises(ProviderError) as info:
        translator.feed(
            "error",
            {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}},
        )
    assert info.value.retryable and info.value.status == 529
    with pytest.raises(ProviderError) as info:
        translator.feed(
            "error", {"type": "error", "error": {"type": "invalid_request_error", "message": "bad"}}
        )
    assert not info.value.retryable


# -- adapter over HTTP --------------------------------------------------------------------


def sse(events) -> bytes:
    return "".join(f"event: {e}\ndata: {json.dumps(d)}\n\n" for e, d in events).encode()


@pytest.fixture
def anthropic_config(monkeypatch: pytest.MonkeyPatch) -> ProviderConfig:
    monkeypatch.setenv("TEST_ANTHROPIC_KEY", "sk-ant-test")
    return ProviderConfig(
        type="anthropic", base_url="https://anthropic.test", api_key_env="TEST_ANTHROPIC_KEY"
    )


@pytest.mark.anyio
async def test_adapter_sends_headers_and_parses_response(anthropic_config: ProviderConfig) -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["headers"] = request.headers
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "msg_x",
                "model": "claude-haiku-4-5",
                "content": [{"type": "text", "text": "Hi!"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 3, "output_tokens": 2},
            },
        )

    provider = AnthropicProvider(
        "anthropic", anthropic_config, transport=httpx.MockTransport(handler)
    )
    response = await provider.chat(req(), "claude-haiku-4-5")
    await provider.aclose()
    assert seen["path"] == "/v1/messages"
    assert seen["headers"]["x-api-key"] == "sk-ant-test"
    assert seen["headers"]["anthropic-version"] == "2023-06-01"
    assert seen["body"]["max_tokens"] == 1024
    assert response.choices[0].message.content == "Hi!" and response.usage.total_tokens == 5


@pytest.mark.anyio
async def test_adapter_streams_sse(anthropic_config: ProviderConfig) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["stream"] is True
        return httpx.Response(
            200, content=sse(TEXT_EVENTS), headers={"content-type": "text/event-stream"}
        )

    provider = AnthropicProvider(
        "anthropic", anthropic_config, transport=httpx.MockTransport(handler)
    )
    chunks = [c async for c in provider.stream(req(stream=True), "claude-haiku-4-5")]
    await provider.aclose()
    assert "".join(c.choices[0].delta.content or "" for c in chunks if c.choices) == "Hello world"
    assert chunks[-1].usage.completion_tokens == 12


@pytest.mark.anyio
async def test_adapter_error_classification(anthropic_config: ProviderConfig) -> None:
    responses = iter(
        [
            httpx.Response(
                429,
                json={
                    "type": "error",
                    "error": {"type": "rate_limit_error", "message": "slow down"},
                },
                headers={"retry-after": "3"},
            ),
            httpx.Response(
                400,
                json={
                    "type": "error",
                    "error": {"type": "invalid_request_error", "message": "bad"},
                },
            ),
            httpx.Response(
                401,
                json={"type": "error", "error": {"type": "authentication_error", "message": "key"}},
            ),
        ]
    )
    provider = AnthropicProvider(
        "anthropic", anthropic_config, transport=httpx.MockTransport(lambda r: next(responses))
    )
    with pytest.raises(ProviderError) as limited:
        await provider.chat(req(), "m")
    with pytest.raises(ProviderError) as bad:
        await provider.chat(req(), "m")
    with pytest.raises(ProviderError) as unauthorised:
        await provider.chat(req(), "m")
    await provider.aclose()
    assert (
        limited.value.retryable
        and limited.value.retry_after == 3
        and limited.value.message == "slow down"
    )
    assert not bad.value.retryable and not bad.value.fallback
    assert not unauthorised.value.retryable and unauthorised.value.fallback
