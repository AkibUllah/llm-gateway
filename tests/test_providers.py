"""OpenAI-compatible, Ollama and fake provider adapters."""

from __future__ import annotations

import json

import httpx
import pytest

from llm_gateway.config import ProviderConfig
from llm_gateway.errors import ProviderError
from llm_gateway.providers.fake import FakeProvider, fake_vector
from llm_gateway.providers.ollama import OllamaProvider, from_ollama_response, to_ollama_request
from llm_gateway.providers.openai import OpenAIProvider
from llm_gateway.schemas import ChatCompletionRequest, EmbeddingRequest

pytestmark = pytest.mark.anyio


def req(**kwargs) -> ChatCompletionRequest:
    kwargs.setdefault("model", "alias")
    kwargs.setdefault("messages", [{"role": "user", "content": "hello"}])
    return ChatCompletionRequest.model_validate(kwargs)


@pytest.fixture
def openai_config(monkeypatch: pytest.MonkeyPatch) -> ProviderConfig:
    monkeypatch.setenv("TEST_OPENAI_KEY", "sk-test-openai")
    return ProviderConfig(
        type="openai", base_url="https://openai.test/v1", api_key_env="TEST_OPENAI_KEY"
    )


async def test_openai_passes_through_with_upstream_model_and_auth(openai_config) -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers["authorization"]
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hi"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
            },
        )

    provider = OpenAIProvider("openai", openai_config, transport=httpx.MockTransport(handler))
    response = await provider.chat(req(temperature=0, seed=7, logprobs=True), "gpt-4o-mini")
    await provider.aclose()
    assert seen["auth"] == "Bearer sk-test-openai"
    assert seen["url"] == "https://openai.test/v1/chat/completions"
    assert seen["body"]["model"] == "gpt-4o-mini" and seen["body"]["seed"] == 7
    assert seen["body"]["logprobs"] is True  # unknown fields pass through to OpenAI upstreams
    assert response.usage.total_tokens == 6


async def test_openai_stream_requests_usage_and_parses_done(openai_config) -> None:
    lines = [
        {
            "id": "c",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "m",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hi"}}],
        },
        {
            "id": "c",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "m",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        },
        {
            "id": "c",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "m",
            "choices": [],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        },
    ]
    body = "".join(f"data: {json.dumps(x)}\n\n" for x in lines) + "data: [DONE]\n\n"

    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["stream_options"] == {"include_usage": True}
        return httpx.Response(
            200, content=body.encode(), headers={"content-type": "text/event-stream"}
        )

    provider = OpenAIProvider("openai", openai_config, transport=httpx.MockTransport(handler))
    chunks = [c async for c in provider.stream(req(stream=True), "m")]
    await provider.aclose()
    assert len(chunks) == 3 and chunks[-1].usage.total_tokens == 4


async def test_openai_errors_and_timeouts(openai_config) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("embeddings"):
            raise httpx.ReadTimeout("slow", request=request)
        return httpx.Response(503, json={"error": {"message": "overloaded"}})

    provider = OpenAIProvider("openai", openai_config, transport=httpx.MockTransport(handler))
    with pytest.raises(ProviderError) as server_error:
        await provider.chat(req(), "m")
    with pytest.raises(ProviderError) as timeout:
        await provider.embed(EmbeddingRequest(model="e", input="x"), "e")
    await provider.aclose()
    assert server_error.value.retryable and server_error.value.message == "overloaded"
    assert timeout.value.retryable and "timed out" in timeout.value.message


def test_ollama_request_translation() -> None:
    payload = to_ollama_request(
        req(
            messages=[
                {"role": "developer", "content": "be brief"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                    ],
                },
            ],
            temperature=0.2,
            max_tokens=64,
            stop=["\n\n"],
            seed=3,
            response_format={"type": "json_object"},
        ),
        "llama3.2",
        stream=False,
    )
    assert payload["messages"][0] == {"role": "system", "content": "be brief"}
    assert payload["messages"][1]["images"] == ["AAAA"]
    assert payload["options"] == {
        "temperature": 0.2,
        "num_predict": 64,
        "stop": ["\n\n"],
        "seed": 3,
    }
    assert payload["format"] == "json"


def test_ollama_response_translation() -> None:
    response = from_ollama_response(
        {
            "model": "llama3.2",
            "message": {"role": "assistant", "content": "Done"},
            "done": True,
            "done_reason": "length",
            "prompt_eval_count": 11,
            "eval_count": 4,
        }
    )
    assert response.choices[0].finish_reason == "length"
    assert response.usage.model_dump() == {
        "prompt_tokens": 11,
        "completion_tokens": 4,
        "total_tokens": 15,
    }


async def test_ollama_ndjson_stream_and_embeddings() -> None:
    lines = [
        {"model": "llama3.2", "message": {"role": "assistant", "content": "Sal"}, "done": False},
        {"model": "llama3.2", "message": {"role": "assistant", "content": "aam"}, "done": False},
        {
            "model": "llama3.2",
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 6,
            "eval_count": 2,
        },
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/embed":
            return httpx.Response(
                200, json={"model": "nomic", "embeddings": [[0.1, 0.2]], "prompt_eval_count": 2}
            )
        return httpx.Response(200, content="\n".join(json.dumps(x) for x in lines).encode())

    config = ProviderConfig(type="ollama", base_url="http://ollama.test")
    provider = OllamaProvider("ollama", config, transport=httpx.MockTransport(handler))
    chunks = [c async for c in provider.stream(req(stream=True), "llama3.2")]
    embedded = await provider.embed(EmbeddingRequest(model="e", input=["a"]), "nomic")
    await provider.aclose()
    assert "".join(c.choices[0].delta.content or "" for c in chunks if c.choices) == "Salaam"
    assert chunks[0].choices[0].delta.role == "assistant"
    assert chunks[-1].usage.total_tokens == 8
    assert embedded.data[0].embedding == [0.1, 0.2] and embedded.usage.prompt_tokens == 2


async def test_fake_provider_is_deterministic_and_scriptable() -> None:
    fake = FakeProvider("fake", ProviderConfig(type="fake", replies={"bkash": "Settles T+1."}))
    first = await fake.chat(
        req(messages=[{"role": "user", "content": "When does bKash settle?"}]), "fake-large"
    )
    again = await fake.chat(
        req(messages=[{"role": "user", "content": "When does bKash settle?"}]), "fake-large"
    )
    echo = await fake.chat(
        req(messages=[{"role": "user", "content": "  hello   world "}]), "fake-large"
    )
    assert first.choices[0].message.content == again.choices[0].message.content == "Settles T+1."
    assert echo.choices[0].message.content == "[fake-large] You said: hello world"
    assert first.usage.prompt_tokens > 0 and first.usage.completion_tokens > 0


async def test_fake_provider_limits_tools_and_failures() -> None:
    fake = FakeProvider("fake", ProviderConfig(type="fake", fail_first=1, failure_status=429))
    with pytest.raises(ProviderError) as info:
        await fake.chat(req(), "m")
    assert info.value.status == 429 and info.value.retryable
    truncated = await fake.chat(
        req(messages=[{"role": "user", "content": "one two three four five six"}], max_tokens=4),
        "m",
    )
    assert truncated.choices[0].finish_reason == "length"
    tool = await fake.chat(
        req(tools=[{"type": "function", "function": {"name": "lookup"}}], tool_choice="required"),
        "m",
    )
    assert tool.choices[0].message.tool_calls[0].function.name == "lookup"
    assert tool.choices[0].finish_reason == "tool_calls"


def test_fake_vectors_are_stable_unit_vectors() -> None:
    a, b = fake_vector("ঢাকা", 16), fake_vector("ঢাকা", 16)
    assert a == b and len(a) == 16
    assert abs(sum(x * x for x in a) - 1) < 1e-4
