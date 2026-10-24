"""FastAPI application factory."""

from __future__ import annotations

import logging
import random
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from importlib import resources
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response

from llm_gateway import __version__
from llm_gateway.api import admin_routes, openai_routes
from llm_gateway.api.deps import AppState
from llm_gateway.cache import MemoryCache, RedisCache, ResponseCache
from llm_gateway.config import GatewayConfig, Settings
from llm_gateway.db import Database
from llm_gateway.errors import GatewayError
from llm_gateway.gateway import Gateway
from llm_gateway.keys import KeyCreate, KeyService
from llm_gateway.logs import configure_logging, request_id_var
from llm_gateway.metrics import Metrics
from llm_gateway.pricing import PriceTable
from llm_gateway.providers import Provider, build_provider
from llm_gateway.ratelimit import KeyRateLimiter, MemoryCounters, RedisCounters
from llm_gateway.routing.router import Router
from llm_gateway.tokens import backend as token_backend
from llm_gateway.usage import UsageStore

logger = logging.getLogger(__name__)

DESCRIPTION = """\
OpenAI-compatible gateway in front of OpenAI, Anthropic, Ollama and a deterministic fake provider.
Virtual keys with budgets, rate limits and model allow-lists; aliases with weighted load balancing,
retries, fallbacks and circuit breakers; response caching; usage and cost reporting.

Authenticate `/v1/*` with `Authorization: Bearer sk-gw-...` (a virtual key) and `/admin/*` with
the `ADMIN_API_KEY`.
"""


def build_providers(
    config: GatewayConfig, transport: httpx.AsyncBaseTransport | None
) -> tuple[dict[str, Provider], dict[str, str]]:
    providers: dict[str, Provider] = {}
    unavailable: dict[str, str] = {}
    for name, provider_config in config.providers.items():
        ok, reason = provider_config.availability()
        if not ok:
            unavailable[name] = reason or "unavailable"
            logger.info("provider %s disabled: %s", name, reason)
            continue
        providers[name] = build_provider(name, provider_config, transport=transport)
    return providers, unavailable


def create_app(
    settings: Settings | None = None,
    *,
    provider_transport: httpx.AsyncBaseTransport | None = None,
    rng: random.Random | None = None,
    sleep: Callable[[float], Awaitable[None]] | None = None,
    redis_client: Any | None = None,
    configure_logs: bool = True,
) -> FastAPI:
    settings = settings or Settings()
    if configure_logs:
        configure_logging(
            settings.log_level, json_logs=settings.log_json, redact_pii=settings.log_redact_pii
        )
    config = GatewayConfig.load(settings.gateway_config)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        metrics = Metrics()
        db = Database(settings.database_url)
        await db.create_all()

        redis = redis_client
        if redis is None and settings.redis_url:
            from redis.asyncio import Redis

            redis = Redis.from_url(settings.redis_url)

        providers, unavailable = build_providers(config, provider_transport)
        router_kwargs: dict[str, Any] = {"unavailable": unavailable, "rng": rng, "metrics": metrics}
        if sleep is not None:
            router_kwargs["sleep"] = sleep
        router = Router(config, providers, **router_kwargs)

        keys = KeyService(db, alert_thresholds=settings.alert_thresholds)
        usage = UsageStore(db)
        cache = None
        if config.cache.enabled:
            backend = (
                RedisCache(redis) if redis is not None else MemoryCache(config.cache.max_entries)
            )
            cache = ResponseCache(backend, config.cache)
        limiter = KeyRateLimiter(RedisCounters(redis) if redis is not None else MemoryCounters())
        gateway = Gateway(
            settings=settings,
            config=config,
            router=router,
            keys=keys,
            usage=usage,
            limiter=limiter,
            prices=PriceTable.load(settings.pricing_config),
            metrics=metrics,
            cache=cache,
        )
        if settings.demo_api_key:
            await keys.ensure(
                settings.demo_api_key,
                KeyCreate(name="demo", budget_usd=settings.demo_key_budget_usd, rpm_limit=120),
            )
        app.state.gateway_state = AppState(
            settings=settings,
            config=config,
            db=db,
            keys=keys,
            usage=usage,
            gateway=gateway,
            metrics=metrics,
            redis=redis,
        )
        logger.info(
            "gateway started",
            extra={
                "providers": sorted(providers),
                "disabled_providers": unavailable,
                "aliases": router.aliases(),
                "cache": type(cache.backend).__name__ if cache else None,
                "token_counter": token_backend(),
            },
        )
        try:
            yield
        finally:
            await gateway.aclose()
            await db.dispose()
            if redis is not None and redis_client is None:
                await redis.aclose()

    app = FastAPI(
        title="LLM Gateway",
        version=__version__,
        description=DESCRIPTION,
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def request_id(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        rid = request.headers.get("x-request-id") or "req_" + uuid.uuid4().hex[:20]
        token = request_id_var.set(rid)
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers.setdefault("x-request-id", rid)
        return response

    @app.exception_handler(GatewayError)
    async def gateway_error(_: Request, exc: GatewayError) -> JSONResponse:
        return JSONResponse(exc.body(), status_code=exc.status, headers=exc.headers)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        param = ".".join(str(p) for p in first.get("loc", ())[1:]) or None
        message = (
            f"{param}: {first.get('msg')}" if param else str(first.get("msg", "invalid request"))
        )
        error = GatewayError(400, message, param=param, code="invalid_request")
        return JSONResponse(error.body(), status_code=400)

    app.include_router(openai_routes.router)
    app.include_router(admin_routes.router)

    @app.get("/health", tags=["Ops"], summary="Liveness and dependency status")
    async def health(request: Request) -> JSONResponse:
        state: AppState = request.app.state.gateway_state
        db_ok = await state.db.ping()
        redis_ok: bool | None = None
        if state.redis is not None:
            try:
                redis_ok = bool(await state.redis.ping())
            except Exception:  # noqa: BLE001
                redis_ok = False
        providers = state.gateway.router.health()
        body = {
            "status": "ok" if db_ok and redis_ok is not False else "degraded",
            "version": __version__,
            "database": db_ok,
            "redis": redis_ok,
            "providers_available": sum(1 for p in providers if p.get("available")),
            "models": state.gateway.router.aliases(),
        }
        return JSONResponse(body, status_code=200 if body["status"] == "ok" else 503)

    @app.get("/metrics", tags=["Ops"], summary="Prometheus metrics", response_class=Response)
    async def metrics(request: Request) -> Response:
        state: AppState = request.app.state.gateway_state
        return Response(
            state.metrics.render(), media_type="text/plain; version=0.0.4; charset=utf-8"
        )

    @app.get("/dashboard", include_in_schema=False)
    async def dashboard() -> HTMLResponse:
        html = resources.files("llm_gateway").joinpath("static/dashboard.html").read_text("utf-8")
        return HTMLResponse(html)

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse("/dashboard")

    return app
