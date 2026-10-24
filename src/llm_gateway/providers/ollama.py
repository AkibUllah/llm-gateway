"""Ollama native API adapter (``/api/chat`` with NDJSON streaming, ``/api/embed``)."""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

from llm_gateway.errors import ProviderError
from llm_gateway.providers.base import HTTPProvider
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


def to_ollama_request(
    request: ChatCompletionRequest, model: str, *, stream: bool
) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    for message in request.messages:
        entry: dict[str, Any] = {
            "role": "system" if message.role == "developer" else message.role,
            "content": message.text(),
        }
        if isinstance(message.content, list):
            images = [
                str(p.get("image_url", {}).get("url", "")).partition(",")[2]
                for p in message.content
                if p.get("type") == "image_url"
                and str(p.get("image_url", {}).get("url", "")).startswith("data:")
            ]
            if images:
                entry["images"] = images
        if message.tool_calls:
            entry["tool_calls"] = [
                {
                    "function": {
                        "name": c.function.name,
                        "arguments": json.loads(c.function.arguments or "{}"),
                    }
                }
                for c in message.tool_calls
            ]
        messages.append(entry)

    options: dict[str, Any] = {}
    if request.temperature is not None:
        options["temperature"] = request.temperature
    if request.top_p is not None:
        options["top_p"] = request.top_p
    if request.output_limit:
        options["num_predict"] = request.output_limit
    if request.stop_list:
        options["stop"] = request.stop_list
    if request.seed is not None:
        options["seed"] = request.seed

    payload: dict[str, Any] = {"model": model, "messages": messages, "stream": stream}
    if options:
        payload["options"] = options
    if request.tools:
        payload["tools"] = [t.model_dump() for t in request.tools]
    if request.response_format and request.response_format.get("type") == "json_object":
        payload["format"] = "json"
    return payload


def _finish(data: dict[str, Any], has_tools: bool) -> FinishReason:
    if has_tools:
        return "tool_calls"
    return "length" if data.get("done_reason") == "length" else "stop"


def _tool_calls(message: dict[str, Any]) -> list[ToolCall]:
    calls = []
    for raw in message.get("tool_calls") or []:
        fn = raw.get("function") or {}
        calls.append(
            ToolCall(
                id="call_" + uuid.uuid4().hex[:12],
                function=FunctionCall(
                    name=fn.get("name", ""),
                    arguments=json.dumps(fn.get("arguments") or {}, ensure_ascii=False),
                ),
            )
        )
    return calls


def from_ollama_response(data: dict[str, Any]) -> ChatCompletionResponse:
    message = data.get("message") or {}
    calls = _tool_calls(message)
    return ChatCompletionResponse(
        model=str(data.get("model") or ""),
        choices=[
            Choice(
                message=ResponseMessage(
                    content=message.get("content") or "", tool_calls=calls or None
                ),
                finish_reason=_finish(data, bool(calls)),
            )
        ],
        usage=Usage.of(int(data.get("prompt_eval_count") or 0), int(data.get("eval_count") or 0)),
    )


class OllamaProvider(HTTPProvider):
    async def chat(self, request: ChatCompletionRequest, model: str) -> ChatCompletionResponse:
        data = await self.post_json("/api/chat", to_ollama_request(request, model, stream=False))
        return from_ollama_response(data)

    async def stream(
        self, request: ChatCompletionRequest, model: str
    ) -> AsyncIterator[ChatCompletionChunk]:
        chunk_id = "chatcmpl-" + uuid.uuid4().hex[:24]
        first = True
        tool_index = 0
        async with self.open_stream(
            "/api/chat", to_ollama_request(request, model, stream=True)
        ) as response:
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                data = json.loads(line)
                if "error" in data:
                    raise ProviderError(self.name, str(data["error"]), status=502)
                message = data.get("message") or {}
                delta = Delta(
                    role="assistant" if first else None, content=message.get("content") or None
                )
                calls = _tool_calls(message)
                if calls:
                    delta.tool_calls = [
                        ToolCallDelta(
                            index=tool_index + i,
                            id=c.id,
                            type="function",
                            function={"name": c.function.name, "arguments": c.function.arguments},
                        )
                        for i, c in enumerate(calls)
                    ]
                    tool_index += len(calls)
                first = False
                model_name = str(data.get("model") or model)
                if delta.content or delta.role or delta.tool_calls:
                    yield ChatCompletionChunk(
                        id=chunk_id, model=model_name, choices=[ChunkChoice(delta=delta)]
                    )
                if data.get("done"):
                    yield ChatCompletionChunk(
                        id=chunk_id,
                        model=model_name,
                        choices=[ChunkChoice(finish_reason=_finish(data, tool_index > 0))],
                    )
                    yield ChatCompletionChunk(
                        id=chunk_id,
                        model=model_name,
                        choices=[],
                        usage=Usage.of(
                            int(data.get("prompt_eval_count") or 0),
                            int(data.get("eval_count") or 0),
                        ),
                    )
                    return

    async def embed(self, request: EmbeddingRequest, model: str) -> EmbeddingResponse:
        data = await self.post_json("/api/embed", {"model": model, "input": request.texts()})
        vectors = data.get("embeddings") or []
        tokens = int(data.get("prompt_eval_count") or 0)
        return EmbeddingResponse(
            model=str(data.get("model") or model),
            data=[EmbeddingData(index=i, embedding=v) for i, v in enumerate(vectors)],
            usage=Usage.of(tokens, 0),
        )
