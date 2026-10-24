"""Request orchestration: key policy -> cache -> router -> accounting (usage, spend, metrics)."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import anyio
import httpx

from llm_gateway.cache import ResponseCache
from llm_gateway.config import GatewayConfig, Settings
from llm_gateway.errors import GatewayError, ProviderError
from llm_gateway.keys import KeyContext, KeyService
from llm_gateway.metrics import Metrics
from llm_gateway.pricing import PriceTable
from llm_gateway.ratelimit import KeyRateLimiter
from llm_gateway.routing.router import RouteInfo, Router
from llm_gateway.schemas import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ChunkChoice,
    Delta,
    EmbeddingRequest,
    EmbeddingResponse,
    FinishReason,
    FunctionCall,
    ModelCard,
    ModelList,
    ResponseMessage,
    ToolCall,
    Usage,
)
from llm_gateway.tokens import count_messages, count_text
from llm_gateway.usage import UsageStore

logger = logging.getLogger(__name__)

DONE = b"data: [DONE]\n\n"


@dataclass
class ChunkAccumulator:
    """Rebuild a full response from stream chunks (for caching and token accounting)."""

    id: str = ""
    model: str = ""
    content: list[str] = field(default_factory=list)
    tools: dict[int, dict[str, str]] = field(default_factory=dict)
    finish_reason: FinishReason | None = None
    usage: Usage | None = None

    def add(self, chunk: ChatCompletionChunk) -> None:
        self.id = self.id or chunk.id
        self.model = self.model or chunk.model
        if chunk.usage is not None:
            self.usage = chunk.usage
        for choice in chunk.choices:
            if choice.delta.content:
                self.content.append(choice.delta.content)
            for call in choice.delta.tool_calls or []:
                slot = self.tools.setdefault(call.index, {"id": "", "name": "", "arguments": ""})
                if call.id:
                    slot["id"] = call.id
                if call.function:
                    slot["name"] += call.function.get("name", "")
                    slot["arguments"] += call.function.get("arguments", "")
            if choice.finish_reason:
                self.finish_reason = choice.finish_reason

    @property
    def text(self) -> str:
        return "".join(self.content)

    def response(self) -> ChatCompletionResponse:
        calls = [
            ToolCall(id=t["id"], function=FunctionCall(name=t["name"], arguments=t["arguments"]))
            for _, t in sorted(self.tools.items())
        ]
        return ChatCompletionResponse(
            id=self.id,
            model=self.model,
            choices=[
                Choice(
                    message=ResponseMessage(
                        content=self.text if self.content or not calls else None,
                        tool_calls=calls or None,
                    ),
                    finish_reason=self.finish_reason or "stop",
                )
            ],
            usage=self.usage or Usage(),
        )


def replay_chunks(
    response: ChatCompletionResponse, include_usage: bool
) -> list[ChatCompletionChunk]:
    """Serve a cached response to a streaming client."""
    choice = response.choices[0]
    chunks = [
        ChatCompletionChunk(
            id=response.id,
            model=response.model,
            choices=[
                ChunkChoice(delta=Delta(role="assistant", content=choice.message.content or ""))
            ],
        )
    ]
    if choice.message.tool_calls:
        from llm_gateway.schemas import ToolCallDelta

        chunks.append(
            ChatCompletionChunk(
                id=response.id,
                model=response.model,
                choices=[
                    ChunkChoice(
                        delta=Delta(
                            tool_calls=[
                                ToolCallDelta(
                                    index=i,
                                    id=c.id,
                                    type="function",
                                    function={
                                        "name": c.function.name,
                                        "arguments": c.function.arguments,
                                    },
                                )
                                for i, c in enumerate(choice.message.tool_calls)
                            ]
                        )
                    )
                ],
            )
        )
    chunks.append(
        ChatCompletionChunk(
            id=response.id,
            model=response.model,
            choices=[ChunkChoice(finish_reason=choice.finish_reason)],
        )
    )
    if include_usage:
        chunks.append(
            ChatCompletionChunk(id=response.id, model=response.model, usage=response.usage)
        )
    return chunks


@dataclass
class CallResult:
    status: int = 200
    error_code: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0
    cached: bool = False
    info: RouteInfo | None = None


class Gateway:
    def __init__(
        self,
        *,
        settings: Settings,
        config: GatewayConfig,
        router: Router,
        keys: KeyService,
        usage: UsageStore,
        limiter: KeyRateLimiter,
        prices: PriceTable,
        metrics: Metrics,
        cache: ResponseCache | None,
    ) -> None:
        self.settings = settings
        self.config = config
        self.router = router
        self.keys = keys
        self.usage = usage
        self.limiter = limiter
        self.prices = prices
        self.metrics = metrics
        self.cache = cache
        self._background: set[asyncio.Task[Any]] = set()

    # -- shared steps -----------------------------------------------------------------------

    async def _preflight(self, key: KeyContext, alias: str, estimated_tokens: int) -> None:
        self.router.resolve(alias)  # 404 / 503 before charging the rate limit
        if not key.allows(alias):
            self.metrics.rejections.labels(reason="model_not_allowed").inc()
            raise GatewayError(
                403,
                f"key {key.name!r} may not use model {alias!r}",
                type="permission_error",
                code="model_not_allowed",
                param="model",
            )
        try:
            key.check_budget()
            await self.limiter.check(
                key.id, rpm=key.rpm_limit, tpm=key.tpm_limit, tokens=estimated_tokens
            )
        except GatewayError as exc:
            self.metrics.rejections.labels(reason=exc.code or "rejected").inc()
            raise

    async def _account(
        self,
        *,
        endpoint: str,
        key: KeyContext,
        alias: str,
        request_id: str,
        started: float,
        result: CallResult,
        stream: bool = False,
    ) -> None:
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        deployment = result.info.deployment if result.info else None
        provider = deployment.provider.name if deployment else None
        try:
            await self.usage.record(
                request_id=request_id,
                key_id=key.id,
                endpoint=endpoint,
                alias=alias,
                provider=provider,
                provider_model=deployment.model if deployment else None,
                stream=stream,
                status_code=result.status,
                error_code=result.error_code,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                cost_usd=result.cost,
                latency_ms=latency_ms,
                cached=result.cached,
                fallback=bool(result.info and result.info.fallback_used),
                attempts=len(result.info.attempts) if result.info else 0,
            )
            if result.cost > 0:
                crossed = await self.keys.add_spend(key.id, result.cost)
                for threshold in crossed:
                    self._notify_budget(key, threshold)
            await self.limiter.add_tokens(key.id, result.prompt_tokens + result.completion_tokens)
        except Exception:  # accounting must never fail the user's request
            logger.exception("failed to record usage", extra={"request_id": request_id})

        status = "ok" if result.status < 400 else str(result.status)
        cache_label = "hit" if result.cached else "miss"
        self.metrics.requests.labels(endpoint, alias, provider or "-", status, cache_label).inc()
        self.metrics.latency.labels(endpoint, alias).observe(latency_ms / 1000)
        if result.prompt_tokens:
            self.metrics.tokens.labels(alias, "prompt").inc(result.prompt_tokens)
        if result.completion_tokens:
            self.metrics.tokens.labels(alias, "completion").inc(result.completion_tokens)
        if result.cost:
            self.metrics.cost.labels(alias).inc(result.cost)
        logger.info(
            "request completed",
            extra={
                "endpoint": endpoint,
                "key_id": key.id,
                "model": alias,
                "provider": provider,
                "status": result.status,
                "cached": result.cached,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "cost_usd": result.cost,
                "latency_ms": latency_ms,
                "attempts": len(result.info.attempts) if result.info else 0,
            },
        )

    def _notify_budget(self, key: KeyContext, threshold: float) -> None:
        url = self.settings.budget_webhook_url
        if not url:
            return

        async def send() -> None:
            payload = {
                "event": "budget.threshold",
                "key_id": key.id,
                "key_name": key.name,
                "threshold": threshold,
                "budget_usd": key.budget_usd,
            }
            try:
                async with httpx.AsyncClient(timeout=5) as client:
                    await client.post(url, json=payload)
            except httpx.HTTPError as exc:
                logger.warning("budget webhook failed", extra={"error": str(exc)})

        task = asyncio.create_task(send())
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    def _headers(self, request_id: str, result: CallResult, *, cost: bool = True) -> dict[str, str]:
        headers = {
            "x-request-id": request_id,
            "x-gateway-cache": "hit" if result.cached else "miss",
        }
        if result.info and result.info.deployment:
            headers["x-gateway-provider"] = result.info.deployment.provider.name
            headers["x-gateway-model"] = result.info.deployment.model
            headers["x-gateway-attempts"] = str(len(result.info.attempts))
            headers["x-gateway-fallback"] = str(result.info.fallback_used).lower()
        if cost:  # streamed responses are priced after the last byte; see /admin/usage
            headers["x-gateway-cost-usd"] = f"{result.cost:.8f}"
        return headers

    def _log_payload(self, request: ChatCompletionRequest) -> None:
        if self.settings.log_payloads:
            logger.info(
                "request payload",
                extra={
                    "model": request.model,
                    "messages": [m.model_dump(exclude_none=True) for m in request.messages],
                },
            )

    def _cost(self, info: RouteInfo, prompt: int, completion: int) -> float:
        assert info.deployment is not None
        return self.prices.cost(info.deployment.model, prompt, completion)

    # -- endpoints --------------------------------------------------------------------------

    def models(self, key: KeyContext) -> ModelList:
        return ModelList(
            data=[ModelCard(id=alias) for alias in self.router.aliases() if key.allows(alias)]
        )

    async def chat(
        self,
        request: ChatCompletionRequest,
        key: KeyContext,
        *,
        request_id: str,
        use_cache: bool = True,
    ) -> tuple[ChatCompletionResponse, dict[str, str]]:
        started = time.perf_counter()
        result = CallResult()
        estimate = count_messages(request.messages)
        cache_key = self.cache.key_for(request) if self.cache and use_cache else None
        try:
            await self._preflight(key, request.model, estimate)
            self._log_payload(request)
            if cache_key:
                cached = await self.cache.get(cache_key)  # type: ignore[union-attr]
                if cached is not None:
                    self.metrics.cache_events.labels(event="hit").inc()
                    result.cached = True
                    result.prompt_tokens = cached.usage.prompt_tokens
                    result.completion_tokens = cached.usage.completion_tokens
                    return cached, self._headers(request_id, result)
                self.metrics.cache_events.labels(event="miss").inc()

            response, info = await self.router.chat(request)
            result.info = info
            usage = response.usage
            if not usage.total_tokens:
                text = response.choices[0].message.content or "" if response.choices else ""
                usage = Usage.of(estimate, count_text(text))
                response.usage = usage
            result.prompt_tokens, result.completion_tokens = (
                usage.prompt_tokens,
                usage.completion_tokens,
            )
            result.cost = self._cost(info, usage.prompt_tokens, usage.completion_tokens)
            if cache_key:
                await self.cache.put(cache_key, response)  # type: ignore[union-attr]
            return response, self._headers(request_id, result)
        except GatewayError as exc:
            result.status, result.error_code = exc.status, exc.code
            raise
        finally:
            await self._account(
                endpoint="chat",
                key=key,
                alias=request.model,
                request_id=request_id,
                started=started,
                result=result,
            )

    async def chat_stream(
        self,
        request: ChatCompletionRequest,
        key: KeyContext,
        *,
        request_id: str,
        use_cache: bool = True,
    ) -> tuple[AsyncIterator[bytes], dict[str, str]]:
        """Return the SSE byte stream. Errors before the first byte raise normally (JSON error)."""
        started = time.perf_counter()
        result = CallResult()
        estimate = count_messages(request.messages)
        include_usage = bool(request.stream_options and request.stream_options.include_usage)
        try:
            await self._preflight(key, request.model, estimate)
        except GatewayError as exc:
            result.status, result.error_code = exc.status, exc.code
            await self._account(
                endpoint="chat",
                key=key,
                alias=request.model,
                request_id=request_id,
                started=started,
                result=result,
                stream=True,
            )
            raise
        self._log_payload(request)
        cache_key = self.cache.key_for(request) if self.cache and use_cache else None

        if cache_key:
            cached = await self.cache.get(cache_key)  # type: ignore[union-attr]
            if cached is not None:
                self.metrics.cache_events.labels(event="hit").inc()
                result.cached = True
                result.prompt_tokens = cached.usage.prompt_tokens
                result.completion_tokens = cached.usage.completion_tokens

                async def replay() -> AsyncIterator[bytes]:
                    try:
                        for chunk in replay_chunks(cached, include_usage):
                            yield chunk.to_sse()
                        yield DONE
                    finally:
                        with anyio.CancelScope(shield=True):
                            await self._account(
                                endpoint="chat",
                                key=key,
                                alias=request.model,
                                request_id=request_id,
                                started=started,
                                result=result,
                                stream=True,
                            )

                return replay(), self._headers(request_id, result, cost=False)
            self.metrics.cache_events.labels(event="miss").inc()

        try:
            handle = await self.router.stream(request)
        except GatewayError as exc:
            result.status, result.error_code = exc.status, exc.code
            await self._account(
                endpoint="chat",
                key=key,
                alias=request.model,
                request_id=request_id,
                started=started,
                result=result,
                stream=True,
            )
            raise
        result.info = handle.info

        async def body() -> AsyncIterator[bytes]:
            acc = ChunkAccumulator()
            failed = False
            try:
                async for chunk in _prepend(handle.first, handle.rest):
                    acc.add(chunk)
                    if chunk.usage is not None and not chunk.choices:
                        if include_usage:
                            yield chunk.to_sse()
                        continue
                    if chunk.usage is not None and not include_usage:
                        chunk = chunk.model_copy(update={"usage": None})
                    yield chunk.to_sse()
            except ProviderError as exc:
                failed = True
                result.status, result.error_code = 502, "upstream_stream_error"
                error = GatewayError(
                    502,
                    f"upstream stream failed: {exc.message}",
                    type="api_error",
                    code="upstream_stream_error",
                )
                yield f"data: {json.dumps(error.body())}\n\n".encode()
            finally:
                with anyio.CancelScope(shield=True):
                    usage = acc.usage
                    if usage is None or not usage.total_tokens:
                        usage = Usage.of(estimate, count_text(acc.text))
                    result.prompt_tokens, result.completion_tokens = (
                        usage.prompt_tokens,
                        usage.completion_tokens,
                    )
                    if handle.info.deployment is not None:
                        result.cost = self._cost(
                            handle.info, usage.prompt_tokens, usage.completion_tokens
                        )
                    if cache_key and not failed and acc.finish_reason:
                        acc.usage = usage
                        await self.cache.put(cache_key, acc.response())  # type: ignore[union-attr]
                    await self._account(
                        endpoint="chat",
                        key=key,
                        alias=request.model,
                        request_id=request_id,
                        started=started,
                        result=result,
                        stream=True,
                    )
            yield DONE

        return body(), self._headers(request_id, result, cost=False)

    async def embeddings(
        self, request: EmbeddingRequest, key: KeyContext, *, request_id: str
    ) -> tuple[EmbeddingResponse, dict[str, str]]:
        started = time.perf_counter()
        result = CallResult()
        try:
            texts = request.texts()
        except ValueError as exc:
            raise GatewayError(400, str(exc), param="input") from exc
        estimate = sum(count_text(t) for t in texts)
        try:
            await self._preflight(key, request.model, estimate)
            response, info = await self.router.embed(request)
            result.info = info
            prompt = response.usage.prompt_tokens or estimate
            response.usage = Usage.of(prompt, 0)
            result.prompt_tokens = prompt
            result.cost = self._cost(info, prompt, 0)
            return response, self._headers(request_id, result)
        except GatewayError as exc:
            result.status, result.error_code = exc.status, exc.code
            raise
        finally:
            await self._account(
                endpoint="embeddings",
                key=key,
                alias=request.model,
                request_id=request_id,
                started=started,
                result=result,
            )

    async def aclose(self) -> None:
        for task in list(self._background):
            task.cancel()
        await self.router.aclose()


async def _prepend(
    first: ChatCompletionChunk, rest: AsyncIterator[ChatCompletionChunk]
) -> AsyncIterator[ChatCompletionChunk]:
    yield first
    async for chunk in rest:
        yield chunk
