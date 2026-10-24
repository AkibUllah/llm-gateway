"""Deterministic offline provider for tests, demos and load experiments.

Same request in, same reply out. Latency, streaming speed and failures (first N calls, or a
seeded failure rate with a chosen HTTP status) are configurable per provider in gateway.yaml,
which is how the demo shows retries, fallbacks and the circuit breaker without any API key.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import random
import re
import uuid
from collections.abc import AsyncIterator

from llm_gateway.config import ProviderConfig
from llm_gateway.errors import classify_http_error
from llm_gateway.providers.base import Provider
from llm_gateway.schemas import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ChunkChoice,
    Delta,
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    FinishReason,
    FunctionCall,
    ResponseMessage,
    ToolCall,
    ToolCallDelta,
    Usage,
)
from llm_gateway.tokens import count_messages, count_text


class FakeProvider(Provider):
    def __init__(self, name: str, config: ProviderConfig) -> None:
        super().__init__(name, config)
        self.calls = 0
        self._rng = random.Random(config.seed)
        self._replies = [(re.compile(p, re.I), r) for p, r in config.replies.items()]

    # -- behaviour --------------------------------------------------------------------------

    async def _before_call(self) -> None:
        self.calls += 1
        if self.config.latency_ms:
            await asyncio.sleep(self.config.latency_ms / 1000)
        failing = self.calls <= self.config.fail_first or (
            self.config.failure_rate > 0 and self._rng.random() < self.config.failure_rate
        )
        if failing:
            status = self.config.failure_status
            raise classify_http_error(
                self.name, status, f"simulated upstream failure ({status})", retry_after=None
            )

    def _reply(self, request: ChatCompletionRequest, model: str) -> tuple[str, FinishReason]:
        last_user = next((m.text() for m in reversed(request.messages) if m.role == "user"), "")
        text = next((reply for pattern, reply in self._replies if pattern.search(last_user)), None)
        if text is None:
            snippet = " ".join(last_user.split())[:240]
            text = f"[{model}] You said: {snippet}" if snippet else f"[{model}] Hello!"
        finish: FinishReason = "stop"
        for stop in request.stop_list:
            if stop in text:
                text = text.split(stop, 1)[0]
        if request.output_limit:
            words = text.split(" ")
            kept: list[str] = []
            for word in words:
                if count_text(" ".join([*kept, word])) > request.output_limit:
                    finish = "length"
                    break
                kept.append(word)
            text = " ".join(kept)
        return text, finish

    def _tool_call(self, request: ChatCompletionRequest) -> ToolCall | None:
        if not request.tools or request.tool_choice in (None, "auto", "none"):
            return None
        name = request.tools[0].function.name
        if isinstance(request.tool_choice, dict):
            name = (request.tool_choice.get("function") or {}).get("name", name)
        return ToolCall(
            id="call_" + uuid.uuid4().hex[:12], function=FunctionCall(name=name, arguments="{}")
        )

    # -- API --------------------------------------------------------------------------------

    async def chat(self, request: ChatCompletionRequest, model: str) -> ChatCompletionResponse:
        await self._before_call()
        prompt = count_messages(request.messages)
        call = self._tool_call(request)
        if call is not None:
            message = ResponseMessage(content=None, tool_calls=[call])
            completion = count_text(call.function.name) + 2
            finish: FinishReason = "tool_calls"
        else:
            text, finish = self._reply(request, model)
            message = ResponseMessage(content=text)
            completion = count_text(text)
        return ChatCompletionResponse(
            model=model,
            choices=[Choice(message=message, finish_reason=finish)],
            usage=Usage.of(prompt, completion),
        )

    async def stream(
        self, request: ChatCompletionRequest, model: str
    ) -> AsyncIterator[ChatCompletionChunk]:
        await self._before_call()
        chunk_id = "chatcmpl-" + uuid.uuid4().hex[:24]
        prompt = count_messages(request.messages)
        yield ChatCompletionChunk(
            id=chunk_id,
            model=model,
            choices=[ChunkChoice(delta=Delta(role="assistant", content=""))],
        )
        call = self._tool_call(request)
        if call is not None:
            delta = ToolCallDelta(
                index=0,
                id=call.id,
                type="function",
                function={"name": call.function.name, "arguments": "{}"},
            )
            yield ChatCompletionChunk(
                id=chunk_id, model=model, choices=[ChunkChoice(delta=Delta(tool_calls=[delta]))]
            )
            finish: FinishReason = "tool_calls"
            completion = count_text(call.function.name) + 2
        else:
            text, finish = self._reply(request, model)
            words = text.split(" ")
            for index, word in enumerate(words):
                piece = word if index == 0 else " " + word
                if self.config.stream_chunk_delay_ms:
                    await asyncio.sleep(self.config.stream_chunk_delay_ms / 1000)
                yield ChatCompletionChunk(
                    id=chunk_id, model=model, choices=[ChunkChoice(delta=Delta(content=piece))]
                )
            completion = count_text(text)
        yield ChatCompletionChunk(
            id=chunk_id, model=model, choices=[ChunkChoice(finish_reason=finish)]
        )
        yield ChatCompletionChunk(
            id=chunk_id, model=model, choices=[], usage=Usage.of(prompt, completion)
        )

    async def embed(self, request: EmbeddingRequest, model: str) -> EmbeddingResponse:
        await self._before_call()
        texts = request.texts()
        dims = request.dimensions or 64
        data = [EmbeddingData(index=i, embedding=fake_vector(t, dims)) for i, t in enumerate(texts)]
        tokens = sum(count_text(t) for t in texts)
        return EmbeddingResponse(model=model, data=data, usage=Usage.of(tokens, 0))


def fake_vector(text: str, dims: int) -> list[float]:
    """Stable unit vector derived from the text's hash; similar inputs are not similar vectors."""
    values: list[float] = []
    counter = 0
    while len(values) < dims:
        digest = hashlib.sha256(f"{counter}:{text}".encode()).digest()
        values.extend((b - 127.5) / 127.5 for b in digest)
        counter += 1
    values = values[:dims]
    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return [round(v / norm, 6) for v in values]
