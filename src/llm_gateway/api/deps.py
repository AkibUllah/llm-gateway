"""Shared request dependencies: application state, API key and admin authentication."""

from __future__ import annotations

import hmac
from dataclasses import dataclass
from typing import Any

from fastapi import Request

from llm_gateway.config import GatewayConfig, Settings
from llm_gateway.db import Database
from llm_gateway.errors import GatewayError, auth_error
from llm_gateway.gateway import Gateway
from llm_gateway.keys import KeyContext, KeyService
from llm_gateway.metrics import Metrics
from llm_gateway.usage import UsageStore


@dataclass
class AppState:
    settings: Settings
    config: GatewayConfig
    db: Database
    keys: KeyService
    usage: UsageStore
    gateway: Gateway
    metrics: Metrics
    redis: Any | None = None


def get_state(request: Request) -> AppState:
    return request.app.state.gateway_state


def bearer_token(request: Request) -> str | None:
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() == "bearer" and token.strip():
        return token.strip()
    return request.headers.get("x-api-key") or None


async def require_key(request: Request) -> KeyContext:
    state = get_state(request)
    key = await state.keys.authenticate(bearer_token(request))
    request.state.key_id = key.id
    return key


async def require_admin(request: Request) -> None:
    state = get_state(request)
    expected = state.settings.admin_api_key
    if not expected:
        raise GatewayError(
            503,
            "admin API is disabled; set ADMIN_API_KEY",
            type="service_unavailable",
            code="admin_disabled",
        )
    supplied = request.headers.get("x-admin-key") or bearer_token(request) or ""
    if not hmac.compare_digest(supplied.encode(), expected.encode()):
        raise auth_error("invalid admin key")


def wants_cache(request: Request) -> bool:
    directive = (request.headers.get("x-gateway-cache") or "").lower()
    cache_control = (request.headers.get("cache-control") or "").lower()
    return (
        directive not in ("no-store", "bypass", "off")
        and "no-cache" not in cache_control
        and ("no-store" not in cache_control)
    )
