"""End-to-end through the FastAPI app with httpx's ASGI transport (streaming included)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from tests.conftest import ADMIN, DEMO, admin, auth, make_client, write_config

pytestmark = pytest.mark.anyio

CHAT = {"model": "chat", "temperature": 0, "messages": [{"role": "user", "content": "ping"}]}


def sse_events(body: str) -> list[dict | str]:
    events: list[dict | str] = []
    for block in body.strip().split("\n\n"):
        data = block.removeprefix("data: ")
        events.append(data if data == "[DONE]" else json.loads(data))
    return events


async def create_key(client: httpx.AsyncClient, **body) -> dict:
    body.setdefault("name", "test")
    response = await client.post("/admin/keys", json=body, headers=admin())
    assert response.status_code == 201, response.text
    return response.json()


async def test_health_models_and_metrics(client: httpx.AsyncClient) -> None:
    health = await client.get("/health")
    assert health.status_code == 200 and health.json()["database"] is True
    assert "claude" in health.json()["models"] and "gpt" not in health.json()["models"]
    models = await client.get("/v1/models", headers=auth())
    assert {m["id"] for m in models.json()["data"]} == {"chat", "backup-chat", "embed", "claude"}
    await client.post("/v1/chat/completions", json=CHAT, headers=auth())
    metrics = (await client.get("/metrics")).text
    assert 'llmgw_requests_total{cache="miss",endpoint="chat",model="chat"' in metrics
    assert "llmgw_cost_usd_total" in metrics


async def test_authentication_errors_use_openai_shape(client: httpx.AsyncClient) -> None:
    missing = await client.post("/v1/chat/completions", json=CHAT)
    wrong = await client.post("/v1/chat/completions", json=CHAT, headers=auth("sk-gw-nope"))
    assert missing.status_code == wrong.status_code == 401
    assert wrong.json()["error"]["type"] == "authentication_error"
    assert wrong.json()["error"]["code"] == "invalid_api_key"
    assert (await client.get("/admin/keys")).status_code == 401
    assert (await client.get("/admin/keys", headers=auth())).status_code == 401
    assert (await client.get("/admin/keys", headers={"x-admin-key": ADMIN})).status_code == 200


async def test_chat_completion_with_routing_headers(client: httpx.AsyncClient) -> None:
    response = await client.post("/v1/chat/completions", json=CHAT, headers=auth())
    body = response.json()
    assert response.status_code == 200
    assert (
        body["object"] == "chat.completion" and body["choices"][0]["message"]["content"] == "pong"
    )
    assert (
        body["usage"]["total_tokens"]
        == body["usage"]["prompt_tokens"] + body["usage"]["completion_tokens"]
    )
    assert response.headers["x-gateway-provider"] in {"primary", "secondary"}
    assert response.headers["x-gateway-cache"] == "miss"
    assert float(response.headers["x-gateway-cost-usd"]) > 0
    assert response.headers["x-request-id"].startswith("req_")


async def test_cache_hit_and_bypass(client: httpx.AsyncClient) -> None:
    first = await client.post("/v1/chat/completions", json=CHAT, headers=auth())
    second = await client.post("/v1/chat/completions", json=CHAT, headers=auth())
    bypass = await client.post(
        "/v1/chat/completions", json=CHAT, headers={**auth(), "Cache-Control": "no-cache"}
    )
    warm = {**CHAT, "temperature": 0.8}
    not_cached = [
        await client.post("/v1/chat/completions", json=warm, headers=auth()) for _ in range(2)
    ]
    assert first.headers["x-gateway-cache"] == "miss"
    assert second.headers["x-gateway-cache"] == "hit" and second.json()["id"] == first.json()["id"]
    assert float(second.headers["x-gateway-cost-usd"]) == 0
    assert bypass.headers["x-gateway-cache"] == "miss"
    assert {r.headers["x-gateway-cache"] for r in not_cached} == {"miss"}


async def test_streaming_sse_with_usage(client: httpx.AsyncClient) -> None:
    payload = {
        **CHAT,
        "stream": True,
        "stream_options": {"include_usage": True},
        "messages": [{"role": "user", "content": "stream this please"}],
    }
    async with client.stream(
        "POST", "/v1/chat/completions", json=payload, headers=auth()
    ) as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        body = "".join([part async for part in response.aiter_text()])
    events = sse_events(body)
    assert events[-1] == "[DONE]"
    chunks = [e for e in events if isinstance(e, dict)]
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"])
    assert text == "[fake-large] You said: stream this please"
    assert chunks[-1]["choices"] == [] and chunks[-1]["usage"]["completion_tokens"] > 0


async def test_streaming_without_usage_option_and_cache_replay(client: httpx.AsyncClient) -> None:
    payload = {**CHAT, "stream": True}
    first = await client.post("/v1/chat/completions", json=payload, headers=auth())
    replay = await client.post("/v1/chat/completions", json=payload, headers=auth())
    first_chunks = [e for e in sse_events(first.text) if isinstance(e, dict)]
    assert all("usage" not in c for c in first_chunks)
    assert replay.headers["x-gateway-cache"] == "hit"
    replay_text = "".join(
        c["choices"][0]["delta"].get("content", "")
        for c in sse_events(replay.text)
        if isinstance(c, dict)
    )
    assert replay_text == "pong"


async def test_embeddings(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/v1/embeddings",
        json={"model": "embed", "input": ["ঢাকা", "Chattogram"], "dimensions": 8},
        headers=auth(),
    )
    body = response.json()
    assert response.status_code == 200 and len(body["data"]) == 2
    assert len(body["data"][0]["embedding"]) == 8 and body["usage"]["prompt_tokens"] > 0
    bad = await client.post(
        "/v1/embeddings", json={"model": "embed", "input": [[1, 2]]}, headers=auth()
    )
    assert bad.status_code == 400


async def test_validation_errors_are_openai_style(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/v1/chat/completions", json={"model": "chat", "messages": []}, headers=auth()
    )
    assert response.status_code == 400
    assert response.json()["error"]["param"] == "messages"
    unknown = await client.post(
        "/v1/chat/completions", json={**CHAT, "model": "gpt-9"}, headers=auth()
    )
    assert unknown.status_code == 404 and unknown.json()["error"]["code"] == "model_not_found"


async def test_key_lifecycle_allowlist_and_revocation(client: httpx.AsyncClient) -> None:
    created = await create_key(
        client, name="mobile-app", allowed_models=["chat", "embed"], rpm_limit=100
    )
    secret = created["key"]
    assert secret.startswith("sk-gw-") and created["key_prefix"] == secret[:12]
    listed = (await client.get("/admin/keys", headers=admin())).json()
    assert all("key" not in k and "key_hash" not in k for k in listed)
    models = (await client.get("/v1/models", headers=auth(secret))).json()["data"]
    assert {m["id"] for m in models} == {"chat", "embed"}
    denied = await client.post(
        "/v1/chat/completions", json={**CHAT, "model": "claude"}, headers=auth(secret)
    )
    assert denied.status_code == 403 and denied.json()["error"]["code"] == "model_not_allowed"
    patched = await client.patch(
        f"/admin/keys/{created['id']}", json={"rpm_limit": 5}, headers=admin()
    )
    assert patched.json()["rpm_limit"] == 5
    await client.delete(f"/admin/keys/{created['id']}", headers=admin())
    revoked = await client.post("/v1/chat/completions", json=CHAT, headers=auth(secret))
    assert revoked.status_code == 401


async def test_expired_key_is_rejected(client: httpx.AsyncClient) -> None:
    past = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
    created = await create_key(client, expires_at=past)
    response = await client.post("/v1/chat/completions", json=CHAT, headers=auth(created["key"]))
    assert response.status_code == 401 and "expired" in response.json()["error"]["message"]


async def test_budget_enforcement_and_alerts(client: httpx.AsyncClient) -> None:
    created = await create_key(client, name="tight", budget_usd=0.00005)
    secret = created["key"]
    statuses = []
    for i in range(6):
        body = {
            **CHAT,
            "temperature": 0.5,
            "messages": [{"role": "user", "content": f"request number {i}"}],
        }
        statuses.append(
            (await client.post("/v1/chat/completions", json=body, headers=auth(secret))).status_code
        )
    assert statuses[0] == 200 and statuses[-1] == 402
    exhausted = await client.post("/v1/chat/completions", json=CHAT, headers=auth(secret))
    assert exhausted.json()["error"]["type"] == "insufficient_quota"
    key = (await client.get(f"/admin/keys/{created['id']}", headers=admin())).json()
    assert key["spent_usd"] >= key["budget_usd"] and key["budget_remaining_usd"] == 0
    alerts = (await client.get(f"/admin/alerts?key_id={created['id']}", headers=admin())).json()
    assert sorted(a["threshold"] for a in alerts) == [0.5, 0.8, 1.0]
    reset = await client.patch(
        f"/admin/keys/{created['id']}", json={"reset_spend": True}, headers=admin()
    )
    assert reset.json()["spent_usd"] == 0


async def test_rate_limit_per_key(client: httpx.AsyncClient) -> None:
    created = await create_key(client, rpm_limit=2)
    codes = [
        (
            await client.post("/v1/chat/completions", json=CHAT, headers=auth(created["key"]))
        ).status_code
        for _ in range(3)
    ]
    limited = await client.post("/v1/chat/completions", json=CHAT, headers=auth(created["key"]))
    assert codes == [200, 200, 429]
    assert (
        limited.json()["error"]["code"] == "requests_per_minute"
        and "retry-after" in limited.headers
    )


async def test_fallback_when_provider_is_not_configured(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/v1/chat/completions", json={**CHAT, "model": "claude"}, headers=auth()
    )
    assert response.status_code == 200
    assert response.headers["x-gateway-provider"] == "backup"
    assert response.headers["x-gateway-fallback"] == "true"


async def test_usage_reports(client: httpx.AsyncClient) -> None:
    await client.post("/v1/chat/completions", json=CHAT, headers=auth())
    await client.post("/v1/chat/completions", json=CHAT, headers=auth())  # cache hit
    await client.post("/v1/embeddings", json={"model": "embed", "input": "x"}, headers=auth())
    await client.post(
        "/v1/chat/completions", json={**CHAT, "model": "gpt-9"}, headers=auth()
    )  # rejected, still recorded
    by_model = (await client.get("/admin/usage?group_by=model", headers=admin())).json()
    rows = {r["group"]: r for r in by_model["rows"]}
    assert rows["chat"]["requests"] == 2 and rows["chat"]["cache_hits"] == 1
    assert rows["embed"]["requests"] == 1
    by_key = (await client.get("/admin/usage?group_by=key", headers=admin())).json()
    assert by_key["rows"][0]["label"] == "demo"
    by_day = (await client.get("/admin/usage?group_by=day", headers=admin())).json()
    assert len(by_day["rows"]) == 1 and by_day["totals"]["requests"] == 4
    assert by_day["totals"]["errors"] == 1  # the unknown model is recorded as a 404
    assert by_day["totals"]["cost_usd"] > 0
    providers = (await client.get("/admin/providers", headers=admin())).json()
    assert {p["name"] for p in providers} >= {"primary", "anthropic"}
    cleared = (await client.post("/admin/cache/clear", headers=admin())).json()
    assert cleared["cleared"] >= 1


async def test_dashboard_is_served(client: httpx.AsyncClient) -> None:
    page = await client.get("/dashboard")
    assert page.status_code == 200 and "chart.js" in page.text.lower()


async def test_anthropic_end_to_end_stream_translation(settings, sleeps, monkeypatch) -> None:
    monkeypatch.setenv("TEST_ANTHROPIC_KEY", "sk-ant-test")
    events = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_e2e",
                "model": "claude-haiku-4-5",
                "usage": {"input_tokens": 10, "output_tokens": 1},
            },
        },
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Assalamu "},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "alaikum"},
        },
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 4},
        },
        {"type": "message_stop"},
    ]
    body = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    async for client in make_client(
        settings, sleep=sleeps, provider_transport=httpx.MockTransport(handler)
    ):
        payload = {
            "model": "claude",
            "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [
                {"role": "system", "content": "Greet in Bangla."},
                {"role": "user", "content": "Hi"},
            ],
        }
        response = await client.post("/v1/chat/completions", json=payload, headers=auth())
        usage = (await client.get("/admin/usage?group_by=provider", headers=admin())).json()
    chunks = [e for e in sse_events(response.text) if isinstance(e, dict)]
    assert response.headers["x-gateway-provider"] == "anthropic"
    assert seen["body"]["system"] == "Greet in Bangla." and seen["body"]["stream"] is True
    assert (
        "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"])
        == "Assalamu alaikum"
    )
    assert chunks[-1]["usage"] == {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}
    row = next(r for r in usage["rows"] if r["group"] == "anthropic")
    assert row["prompt_tokens"] == 10 and row["completion_tokens"] == 4
    assert row["cost_usd"] == pytest.approx((10 * 1 + 4 * 5) / 1_000_000)


async def test_mid_stream_failure_is_reported_in_band(settings, sleeps, monkeypatch) -> None:
    monkeypatch.setenv("TEST_ANTHROPIC_KEY", "sk-ant-test")
    events = [
        {
            "type": "message_start",
            "message": {"id": "m", "model": "x", "usage": {"input_tokens": 1}},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "partial"},
        },
        {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}},
    ]
    body = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()
    transport = httpx.MockTransport(lambda r: httpx.Response(200, content=body))
    async for client in make_client(settings, sleep=sleeps, provider_transport=transport):
        response = await client.post(
            "/v1/chat/completions",
            headers=auth(),
            json={
                "model": "claude",
                "stream": True,
                "messages": [{"role": "user", "content": "x"}],
            },
        )
        usage = (await client.get("/admin/usage?group_by=model", headers=admin())).json()
    events_out = sse_events(response.text)
    assert events_out[-1] == "[DONE]"
    assert events_out[-2]["error"]["code"] == "upstream_stream_error"
    assert usage["totals"]["errors"] == 1


async def test_redis_backed_cache_and_limits(tmp_path, sleeps) -> None:
    import fakeredis

    from llm_gateway.config import Settings

    redis = fakeredis.FakeAsyncRedis()
    settings = Settings(
        _env_file=None,
        gateway_config=write_config(tmp_path / "g.yaml"),
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'r.db'}",
        admin_api_key=ADMIN,
        demo_api_key=DEMO,
    )
    async for client in make_client(settings, sleep=sleeps, redis_client=redis):
        await client.post("/v1/chat/completions", json=CHAT, headers=auth())
        hit = await client.post("/v1/chat/completions", json=CHAT, headers=auth())
        health = (await client.get("/health")).json()
    assert hit.headers["x-gateway-cache"] == "hit" and health["redis"] is True
    assert [k async for k in redis.scan_iter("llmgw:rpm:*")]
