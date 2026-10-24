"""OpenAI-compatible upstreams (OpenAI, Azure-style proxies, vLLM, LM Studio, Groq, ...)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from llm_gateway.errors import ProviderError
from llm_gateway.providers.base import HTTPProvider, iter_sse
from llm_gateway.schemas import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    EmbeddingRequest,
    EmbeddingResponse,
)


class OpenAIProvider(HTTPProvider):
    def auth_headers(self) -> dict[str, str]:
        key = self.config.api_key()
        return {"Authorization": f"Bearer {key}"} if key else {}

    @staticmethod
    def build_payload(
        request: ChatCompletionRequest, model: str, *, stream: bool
    ) -> dict[str, Any]:
        payload = request.model_dump(exclude_none=True)
        payload["model"] = model
        payload["stream"] = stream
        if stream:
            # Always ask for usage so spend tracking works; the gateway strips it if the
            # client did not request it.
            payload["stream_options"] = {"include_usage": True}
        else:
            payload.pop("stream_options", None)
        return payload

    async def chat(self, request: ChatCompletionRequest, model: str) -> ChatCompletionResponse:
        data = await self.post_json(
            "/chat/completions", self.build_payload(request, model, stream=False)
        )
        try:
            return ChatCompletionResponse.model_validate(data)
        except ValueError as exc:
            raise ProviderError(self.name, f"unexpected response shape: {exc}", status=502) from exc

    async def stream(
        self, request: ChatCompletionRequest, model: str
    ) -> AsyncIterator[ChatCompletionChunk]:
        payload = self.build_payload(request, model, stream=True)
        async with self.open_stream("/chat/completions", payload) as response:
            async for _event, data in iter_sse(response):
                if data.strip() == "[DONE]":
                    return
                try:
                    raw = json.loads(data)
                except ValueError:
                    continue
                if "error" in raw:
                    message = raw["error"].get("message", "stream error")
                    raise ProviderError(self.name, message, status=502)
                yield ChatCompletionChunk.model_validate(raw)

    async def embed(self, request: EmbeddingRequest, model: str) -> EmbeddingResponse:
        payload = request.model_dump(exclude_none=True)
        payload["model"] = model
        data = await self.post_json("/embeddings", payload)
        return EmbeddingResponse.model_validate(data)
