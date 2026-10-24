from __future__ import annotations

import random
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

from llm_gateway.app import create_app
from llm_gateway.config import GatewayConfig, Settings

ROOT = Path(__file__).resolve().parents[1]
ADMIN = "test-admin-key"
DEMO = "sk-gw-test-demo-key-000000000001"

TEST_CONFIG: dict[str, Any] = {
    "providers": {
        "primary": {"type": "fake", "replies": {"^ping$": "pong"}},
        "secondary": {"type": "fake"},
        "backup": {"type": "fake"},
        "anthropic": {
            "type": "anthropic",
            "base_url": "https://anthropic.test",
            "api_key_env": "TEST_ANTHROPIC_KEY",
        },
        "openai": {
            "type": "openai",
            "base_url": "https://openai.test/v1",
            "api_key_env": "TEST_OPENAI_KEY",
        },
    },
    "models": {
        "chat": {
            "targets": [
                {"provider": "primary", "model": "fake-large", "weight": 3},
                {"provider": "secondary", "model": "fake-large", "weight": 1},
            ],
            "fallbacks": ["backup-chat"],
        },
        "backup-chat": {"targets": [{"provider": "backup", "model": "fake-small"}]},
        "embed": {"targets": [{"provider": "primary", "model": "fake-embed"}]},
        "claude": {
            "targets": [{"provider": "anthropic", "model": "claude-haiku-4-5"}],
            "fallbacks": ["backup-chat"],
        },
        "gpt": {"targets": [{"provider": "openai", "model": "gpt-4o-mini"}]},
    },
    "routing": {
        "max_retries": 2,
        "backoff_base_ms": 100,
        "backoff_max_ms": 1000,
        "request_timeout_s": 5,
        "breaker": {"failure_threshold": 3, "recovery_s": 30},
    },
    "cache": {"enabled": True, "ttl_s": 60, "only_deterministic": True},
}


def write_config(path: Path, overrides: dict[str, Any] | None = None) -> Path:
    data = yaml.safe_load(yaml.safe_dump(TEST_CONFIG))
    for key, value in (overrides or {}).items():
        data[key] = value
    path.write_text(yaml.safe_dump(data))
    return path


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def gateway_config() -> GatewayConfig:
    return GatewayConfig.model_validate(yaml.safe_load(yaml.safe_dump(TEST_CONFIG)))


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        environment="test",
        gateway_config=write_config(tmp_path / "gateway.yaml"),
        pricing_config=ROOT / "config" / "pricing.yaml",
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'gateway.db'}",
        admin_api_key=ADMIN,
        demo_api_key=DEMO,
        demo_key_budget_usd=100,
        log_json=False,
    )


class Sleeps:
    def __init__(self) -> None:
        self.calls: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


@pytest.fixture
def sleeps() -> Sleeps:
    return Sleeps()


async def make_client(settings: Settings, **kwargs: Any) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(settings, rng=random.Random(1), configure_logs=False, **kwargs)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://gateway.test") as client:
            client.app = app  # type: ignore[attr-defined]
            yield client


@pytest.fixture
async def client(settings: Settings, sleeps: Sleeps) -> AsyncIterator[httpx.AsyncClient]:
    async for c in make_client(settings, sleep=sleeps):
        yield c


def auth(key: str = DEMO) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


def admin() -> dict[str, str]:
    return {"Authorization": f"Bearer {ADMIN}"}
