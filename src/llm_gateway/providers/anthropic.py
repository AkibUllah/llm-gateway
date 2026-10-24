"""Anthropic Messages API adapter: OpenAI chat format <-> Messages format, including streaming.

Translation rules (request):
* ``system`` / ``developer`` messages become the top-level ``system`` string.
* ``tool`` messages become ``tool_result`` blocks inside a user turn; assistant ``tool_calls``
  become ``tool_use`` blocks. Consecutive turns with the same role are merged, as the API
  requires alternating roles.
* ``image_url`` parts become ``image`` blocks (base64 data URLs or plain URLs).
* ``max_tokens`` is mandatory upstream, so the provider's ``default_max_tokens`` fills it in.
* ``stop`` -> ``stop_sequences``, ``user`` -> ``metadata.user_id``, ``tools`` -> ``input_schema``,
  ``tool_choice`` auto/required/none/function -> auto/any/none/tool.

Response: text blocks are concatenated, ``tool_use`` blocks become ``tool_calls``, stop reasons map
to OpenAI finish reasons, and cache read/write tokens count as prompt tokens.
"""

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
    ChatMessage,
    Choice,
    ChunkChoice,
    Delta,
    FinishReason,
    FunctionCall,
    ResponseMessage,
    ToolCall,
    ToolCallDelta,
    Usage,
)

ANTHROPIC_VERSION = "2023-06-01"

STOP_REASONS: dict[str, FinishReason] = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "pause_turn": "stop",
    "max_tokens": "length",
    "model_context_window_exceeded": "length",
    "tool_use": "tool_calls",
    "refusal": "content_filter",
}


class TranslationError(ProviderError):
    def __init__(self, message: str) -> None:
        super().__init__("anthropic", message, status=400, retryable=False, fallback=False)


# -- request ------------------------------------------------------------------------------


def _image_block(part: dict[str, Any]) -> dict[str, Any]:
    image = part.get("image_url") or {}
    url = image.get("url", "") if isinstance(image, dict) else str(image)
    if url.startswith("data:"):
        header, _, data = url.partition(",")
        media_type = header[5:].split(";")[0] or "image/png"
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": data},
        }
    return {"type": "image", "source": {"type": "url", "url": url}}


def _content_blocks(content: str | list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    blocks: list[dict[str, Any]] = []
    for part in content:
        kind = part.get("type")
        if kind == "text":
            blocks.append({"type": "text", "text": str(part.get("text", ""))})
        elif kind == "image_url":
            blocks.append(_image_block(part))
        else:
            raise TranslationError(f"unsupported content part type {kind!r} for Anthropic")
    return blocks


def _tool_use_block(call: ToolCall) -> dict[str, Any]:
    try:
        arguments = json.loads(call.function.arguments or "{}")
    except ValueError as exc:
        raise TranslationError(f"tool call {call.id} has invalid JSON arguments") from exc
    if not isinstance(arguments, dict):
        raise TranslationError(f"tool call {call.id} arguments must be a JSON object")
    return {"type": "tool_use", "id": call.id, "name": call.function.name, "input": arguments}


def convert_messages(messages: list[ChatMessage]) -> tuple[str | None, list[dict[str, Any]]]:
    system_parts: list[str] = []
    turns: list[dict[str, Any]] = []

    def add(role: str, blocks: list[dict[str, Any]]) -> None:
        if not blocks:
            return
        if turns and turns[-1]["role"] == role:
            turns[-1]["content"].extend(blocks)
        else:
            turns.append({"role": role, "content": blocks})

    for message in messages:
        if message.role in ("system", "developer"):
            text = message.text()
            if text:
                system_parts.append(text)
        elif message.role == "user":
            add("user", _content_blocks(message.content))
        elif message.role == "assistant":
            blocks = _content_blocks(message.content)
            blocks += [_tool_use_block(call) for call in message.tool_calls or []]
            add("assistant", blocks)
        elif message.role == "tool":
            if not message.tool_call_id:
                raise TranslationError("tool messages need tool_call_id")
            add(
                "user",
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": message.tool_call_id,
                        "content": message.text(),
                    }
                ],
            )
    system = "\n\n".join(system_parts) if system_parts else None
    return system, turns


def convert_tool_choice(choice: str | dict[str, Any] | None) -> dict[str, Any] | None:
    if choice is None:
        return None
    if isinstance(choice, str):
        mapping = {"auto": {"type": "auto"}, "required": {"type": "any"}, "none": {"type": "none"}}
        if choice not in mapping:
            raise TranslationError(f"unsupported tool_choice {choice!r}")
        return mapping[choice]
    name = (choice.get("function") or {}).get("name")
    if not name:
        raise TranslationError("tool_choice object needs function.name")
    return {"type": "tool", "name": name}


def to_anthropic_request(
    request: ChatCompletionRequest, model: str, *, default_max_tokens: int, stream: bool
) -> dict[str, Any]:
    system, messages = convert_messages(request.messages)
    if not messages:
        raise TranslationError("at least one user or assistant message is required")
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": request.output_limit or default_max_tokens,
    }
    if system:
        payload["system"] = system
    if stream:
        payload["stream"] = True
    if request.temperature is not None:
        payload["temperature"] = min(request.temperature, 1.0)  # Anthropic's range is 0..1
    if request.top_p is not None:
        payload["top_p"] = request.top_p
    if request.stop_list:
        payload["stop_sequences"] = request.stop_list
    if request.user:
        payload["metadata"] = {"user_id": request.user}
    if request.tools:
        payload["tools"] = [
            {
                "name": tool.function.name,
                "description": tool.function.description or "",
                "input_schema": tool.function.parameters,
            }
            for tool in request.tools
        ]
    tool_choice = convert_tool_choice(request.tool_choice)
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice
    return payload


# -- response -----------------------------------------------------------------------------


def _usage(raw: dict[str, Any] | None) -> tuple[int, int]:
    raw = raw or {}
    prompt = (
        int(raw.get("input_tokens") or 0)
        + int(raw.get("cache_creation_input_tokens") or 0)
        + int(raw.get("cache_read_input_tokens") or 0)
    )
    return prompt, int(raw.get("output_tokens") or 0)


def from_anthropic_response(data: dict[str, Any]) -> ChatCompletionResponse:
    texts: list[str] = []
    calls: list[ToolCall] = []
    for block in data.get("content") or []:
        if block.get("type") == "text":
            texts.append(block.get("text", ""))
        elif block.get("type") == "tool_use":
            calls.append(
                ToolCall(
                    id=block["id"],
                    function=FunctionCall(
                        name=block["name"],
                        arguments=json.dumps(block.get("input") or {}, ensure_ascii=False),
                    ),
                )
            )
    prompt, completion = _usage(data.get("usage"))
    finish = STOP_REASONS.get(str(data.get("stop_reason")), "stop")
    return ChatCompletionResponse(
        id=str(data.get("id") or "msg_unknown"),
        model=str(data.get("model") or ""),
        choices=[
            Choice(
                message=ResponseMessage(
                    content="".join(texts) if texts or not calls else None, tool_calls=calls or None
                ),
                finish_reason=finish,
            )
        ],
        usage=Usage.of(prompt, completion),
    )


class StreamTranslator:
    """Turn Anthropic stream events into OpenAI ``chat.completion.chunk`` objects.

    message_start          -> first chunk with ``role: assistant``
    content_block_start    -> (tool_use) chunk announcing the tool call id and name
    content_block_delta    -> text_delta: ``content``; input_json_delta: tool ``arguments``
    message_delta          -> chunk with ``finish_reason``; records output token usage
    message_stop           -> final chunk with ``usage`` and no choices
    error                  -> ProviderError (``overloaded_error`` is retryable)
    """

    def __init__(self, provider: str = "anthropic") -> None:
        self.provider = provider
        self.id = "msg_unknown"
        self.model = ""
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self._tool_index: dict[int, int] = {}
        self.done = False

    def _chunk(
        self, delta: Delta | None = None, finish: FinishReason | None = None
    ) -> ChatCompletionChunk:
        return ChatCompletionChunk(
            id=self.id,
            model=self.model,
            choices=[ChunkChoice(delta=delta or Delta(), finish_reason=finish)],
        )

    def feed(self, event: str | None, data: dict[str, Any]) -> list[ChatCompletionChunk]:
        kind = data.get("type") or event
        if kind == "message_start":
            message = data.get("message") or {}
            self.id = str(message.get("id") or self.id)
            self.model = str(message.get("model") or self.model)
            self.prompt_tokens, self.completion_tokens = _usage(message.get("usage"))
            return [self._chunk(Delta(role="assistant", content=""))]
        if kind == "content_block_start":
            block = data.get("content_block") or {}
            if block.get("type") == "tool_use":
                index = len(self._tool_index)
                self._tool_index[int(data.get("index", 0))] = index
                call = ToolCallDelta(
                    index=index,
                    id=block.get("id"),
                    type="function",
                    function={"name": block.get("name", ""), "arguments": ""},
                )
                return [self._chunk(Delta(tool_calls=[call]))]
            if block.get("type") == "text" and block.get("text"):
                return [self._chunk(Delta(content=block["text"]))]
            return []
        if kind == "content_block_delta":
            delta = data.get("delta") or {}
            if delta.get("type") == "text_delta":
                return [self._chunk(Delta(content=delta.get("text", "")))]
            if delta.get("type") == "input_json_delta":
                index = self._tool_index.get(int(data.get("index", 0)), 0)
                call = ToolCallDelta(
                    index=index, function={"arguments": delta.get("partial_json", "")}
                )
                return [self._chunk(Delta(tool_calls=[call]))]
            return []  # thinking / signature deltas are not part of the OpenAI format
        if kind == "message_delta":
            usage = data.get("usage") or {}
            if "output_tokens" in usage:
                self.completion_tokens = int(usage["output_tokens"])
            if usage.get("input_tokens"):
                self.prompt_tokens = _usage(usage)[0]
            reason = (data.get("delta") or {}).get("stop_reason")
            if reason:
                return [self._chunk(finish=STOP_REASONS.get(reason, "stop"))]
            return []
        if kind == "message_stop":
            self.done = True
            return [
                ChatCompletionChunk(
                    id=self.id,
                    model=self.model,
                    choices=[],
                    usage=Usage.of(self.prompt_tokens, self.completion_tokens),
                )
            ]
        if kind == "error":
            error = data.get("error") or {}
            err_type = error.get("type", "api_error")
            retryable = err_type in ("overloaded_error", "api_error", "rate_limit_error")
            status = {"overloaded_error": 529, "rate_limit_error": 429}.get(err_type, 502)
            raise ProviderError(
                self.provider, error.get("message", err_type), status=status, retryable=retryable
            )
        return []  # ping, content_block_stop


class AnthropicProvider(HTTPProvider):
    def auth_headers(self) -> dict[str, str]:
        headers = {"anthropic-version": ANTHROPIC_VERSION}
        key = self.config.api_key()
        if key:
            headers["x-api-key"] = key
        return headers

    async def chat(self, request: ChatCompletionRequest, model: str) -> ChatCompletionResponse:
        payload = to_anthropic_request(
            request, model, default_max_tokens=self.config.default_max_tokens, stream=False
        )
        return from_anthropic_response(await self.post_json("/v1/messages", payload))

    async def stream(
        self, request: ChatCompletionRequest, model: str
    ) -> AsyncIterator[ChatCompletionChunk]:
        payload = to_anthropic_request(
            request, model, default_max_tokens=self.config.default_max_tokens, stream=True
        )
        translator = StreamTranslator(self.name)
        async with self.open_stream("/v1/messages", payload) as response:
            async for event, raw in iter_sse(response):
                try:
                    data = json.loads(raw)
                except ValueError:
                    continue
                for chunk in translator.feed(event, data):
                    yield chunk
                if translator.done:
                    return
