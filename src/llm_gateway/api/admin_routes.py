"""Admin API: virtual keys, usage and cost reports, budget alerts, provider health, cache."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, Query, Request

from llm_gateway.api.deps import get_state, require_admin
from llm_gateway.keys import KeyCreate, KeyCreated, KeyOut, KeyUpdate
from llm_gateway.usage import GroupBy, UsageReport

router = APIRouter(prefix="/admin", tags=["Admin"], dependencies=[Depends(require_admin)])


@router.post("/keys", response_model=KeyCreated, status_code=201, summary="Create a virtual key")
async def create_key(body: KeyCreate, request: Request) -> KeyCreated:
    return await get_state(request).keys.create(body)


@router.get("/keys", response_model=list[KeyOut], summary="List keys (secrets are never returned)")
async def list_keys(request: Request) -> list[KeyOut]:
    return await get_state(request).keys.list()


@router.get("/keys/{key_id}", response_model=KeyOut)
async def get_key(key_id: str, request: Request) -> KeyOut:
    return await get_state(request).keys.get(key_id)


@router.patch("/keys/{key_id}", response_model=KeyOut, summary="Change limits, budget or status")
async def update_key(key_id: str, body: KeyUpdate, request: Request) -> KeyOut:
    return await get_state(request).keys.update(key_id, body)


@router.delete("/keys/{key_id}", response_model=KeyOut, summary="Revoke a key")
async def revoke_key(key_id: str, request: Request) -> KeyOut:
    return await get_state(request).keys.revoke(key_id)


@router.get("/usage", response_model=UsageReport, summary="Usage and cost, grouped")
async def usage(
    request: Request,
    group_by: GroupBy = "model",
    since: datetime | None = None,
    until: datetime | None = None,
    key_id: str | None = None,
) -> UsageReport:
    return await get_state(request).usage.report(group_by, since, until, key_id)


@router.get("/alerts", summary="Budget threshold alerts, newest first")
async def alerts(
    request: Request, key_id: str | None = Query(default=None)
) -> list[dict[str, Any]]:
    return await get_state(request).keys.alerts(key_id)


@router.get("/providers", summary="Provider availability and circuit-breaker state")
async def providers(request: Request) -> list[dict[str, Any]]:
    return get_state(request).gateway.router.health()


@router.post("/cache/clear", summary="Drop every cached response")
async def clear_cache(request: Request) -> dict[str, int]:
    cache = get_state(request).gateway.cache
    return {"cleared": await cache.backend.clear() if cache else 0}
