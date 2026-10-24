"""Provider protocol and shared HTTP plumbing for the adapters."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx

from llm_gateway.config import ProviderConfig
from llm_gateway.errors import (
    ProviderError,
    UnsupportedOperation,
    classify_http_error,
    parse_retry_after,
)
from llm_gateway.schemas import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    EmbeddingRequest,
    EmbeddingResponse,
)


class Provider(ABC):
    """One upstream. Adapters translate the OpenAI-shaped request into the provider's API."""

    def __init__(self, name: str, config: ProviderConfig) -> None:
        self.name = name
        self.config = config

    @property
    def type(self) -> str:
        return self.config.type

    @abstractmethod
    async def chat(self, request: ChatCompletionRequest, model: str) -> ChatCompletionResponse: ...

    @abstractmethod
    def stream(
        self, request: ChatCompletionRequest, model: str
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Yield OpenAI-shaped chunks. The final chunk carries ``usage`` when known."""

    async def embed(self, request: EmbeddingRequest, model: str) -> EmbeddingResponse:
        raise UnsupportedOperation(self.name, f"{self.type} provider does not serve embeddings")

    async def aclose(self) -> None:  # noqa: B027 - optional hook
        """Release network resources."""


class HTTPProvider(Provider):
    def __init__(
        self,
        name: str,
        config: ProviderConfig,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        super().__init__(name, config)
        base_url = (config.resolved_base_url() or "").rstrip("/")
        self.client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(config.timeout_s, connect=min(10.0, config.timeout_s)),
            headers=config.headers,
            transport=transport,
        )

    def auth_headers(self) -> dict[str, str]:
        return {}

    async def aclose(self) -> None:
        await self.client.aclose()

    def _error_from_response(self, response: httpx.Response, body: bytes) -> ProviderError:
        message = body.decode("utf-8", "replace")[:500] or response.reason_phrase
        try:
            payload = json.loads(body)
            err = payload.get("error", payload)
            if isinstance(err, dict):
                message = str(err.get("message") or err)
            elif isinstance(err, str):
                message = err
        except (ValueError, AttributeError):
            pass
        retry_after = parse_retry_after(response.headers.get("retry-after"))
        return classify_http_error(self.name, response.status_code, message, retry_after)

    async def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = await self.client.post(path, json=payload, headers=self.auth_headers())
        except httpx.TimeoutException as exc:
            raise ProviderError(self.name, "upstream timed out", retryable=True) from exc
        except httpx.TransportError as exc:
            raise ProviderError(self.name, f"connection failed: {exc}", retryable=True) from exc
        if response.status_code >= 400:
            raise self._error_from_response(response, response.content)
        try:
            return response.json()
        except ValueError as exc:
            raise ProviderError(self.name, "upstream returned invalid JSON", status=502) from exc

    @asynccontextmanager
    async def open_stream(
        self, path: str, payload: dict[str, Any]
    ) -> AsyncIterator[httpx.Response]:
        try:
            async with self.client.stream(
                "POST", path, json=payload, headers=self.auth_headers()
            ) as response:
                if response.status_code >= 400:
                    body = await response.aread()
                    raise self._error_from_response(response, body)
                yield response
        except httpx.TimeoutException as exc:
            raise ProviderError(self.name, "upstream timed out", retryable=True) from exc
        except httpx.TransportError as exc:
            raise ProviderError(self.name, f"connection failed: {exc}", retryable=True) from exc


async def iter_sse(response: httpx.Response) -> AsyncIterator[tuple[str | None, str]]:
    """Parse a text/event-stream body into ``(event, data)`` pairs."""
    event: str | None = None
    data: list[str] = []
    async for line in response.aiter_lines():
        if line == "":
            if data:
                yield event, "\n".join(data)
            event, data = None, []
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        value = value[1:] if value.startswith(" ") else value
        if field == "event":
            event = value
        elif field == "data":
            data.append(value)
    if data:
        yield event, "\n".join(data)
