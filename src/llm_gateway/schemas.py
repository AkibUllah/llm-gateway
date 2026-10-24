"""OpenAI-compatible request/response models.

These are the gateway's canonical format: every provider adapter translates to and from them,
so clients written for the OpenAI API work unchanged.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Role = Literal["system", "developer", "user", "assistant", "tool"]


class _Open(BaseModel):
    """Accept and keep unknown fields so newer client options pass through to OpenAI upstreams."""

    model_config = ConfigDict(extra="allow")


class FunctionCall(BaseModel):
    name: str
    arguments: str = ""


class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class ChatMessage(_Open):
    role: Role
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None

    def text(self) -> str:
        """Plain text of the message (text parts joined), used for caching and token counts."""
        if self.content is None:
            return ""
        if isinstance(self.content, str):
            return self.content
        return "\n".join(
            str(part.get("text", "")) for part in self.content if part.get("type") == "text"
        )


class FunctionDef(BaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})


class ToolDef(BaseModel):
    type: Literal["function"] = "function"
    function: FunctionDef


class StreamOptions(BaseModel):
    include_usage: bool = False


class ChatCompletionRequest(_Open):
    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    max_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    stream: bool = False
    stream_options: StreamOptions | None = None
    stop: str | list[str] | None = None
    n: int = Field(default=1, ge=1, le=1)
    seed: int | None = None
    user: str | None = None
    tools: list[ToolDef] | None = None
    tool_choice: str | dict[str, Any] | None = None
    response_format: dict[str, Any] | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None

    @field_validator("stop")
    @classmethod
    def _stop_list(cls, value: str | list[str] | None) -> str | list[str] | None:
        if isinstance(value, list) and len(value) > 4:
            raise ValueError("at most 4 stop sequences are allowed")
        return value

    @property
    def output_limit(self) -> int | None:
        return self.max_completion_tokens or self.max_tokens

    @property
    def stop_list(self) -> list[str]:
        if self.stop is None:
            return []
        return [self.stop] if isinstance(self.stop, str) else list(self.stop)


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def of(cls, prompt: int, completion: int) -> Usage:
        return cls(
            prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion
        )


class ResponseMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str | None = None
    tool_calls: list[ToolCall] | None = None


FinishReason = Literal["stop", "length", "tool_calls", "content_filter"]


class Choice(BaseModel):
    index: int = 0
    message: ResponseMessage
    finish_reason: FinishReason | None = "stop"


def new_completion_id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex[:24]


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=new_completion_id)
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[Choice]
    usage: Usage = Field(default_factory=Usage)
    system_fingerprint: str | None = None


class ToolCallDelta(BaseModel):
    index: int
    id: str | None = None
    type: Literal["function"] | None = None
    function: dict[str, str] | None = None


class Delta(BaseModel):
    role: Literal["assistant"] | None = None
    content: str | None = None
    tool_calls: list[ToolCallDelta] | None = None


class ChunkChoice(BaseModel):
    index: int = 0
    delta: Delta = Field(default_factory=Delta)
    finish_reason: FinishReason | None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChunkChoice] = Field(default_factory=list)
    usage: Usage | None = None

    def to_sse(self) -> bytes:
        return f"data: {self.model_dump_json(exclude_none=True)}\n\n".encode()


class EmbeddingRequest(_Open):
    model: str
    input: str | list[str] | list[int] | list[list[int]]
    encoding_format: Literal["float", "base64"] = "float"
    dimensions: int | None = Field(default=None, ge=1)
    user: str | None = None

    def texts(self) -> list[str]:
        if isinstance(self.input, str):
            return [self.input]
        if self.input and all(isinstance(x, str) for x in self.input):
            return list(self.input)  # type: ignore[arg-type]
        raise ValueError("token-array input is not supported; send strings")


class EmbeddingData(BaseModel):
    object: Literal["embedding"] = "embedding"
    index: int
    embedding: list[float]


class EmbeddingResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[EmbeddingData]
    model: str
    usage: Usage = Field(default_factory=Usage)


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = 0
    owned_by: str = "llm-gateway"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]
