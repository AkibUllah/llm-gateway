"""Settings (environment) and the routing configuration (YAML)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ProviderType = Literal["openai", "anthropic", "ollama", "fake"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    environment: Literal["development", "production", "test"] = "development"
    gateway_config: Path = Path("config/gateway.yaml")
    pricing_config: Path = Path("config/pricing.yaml")
    database_url: str = "sqlite+aiosqlite:///./gateway.db"
    redis_url: str | None = None

    admin_api_key: str | None = None
    demo_api_key: str | None = Field(
        default=None, description="Create (or keep) a virtual key with this exact value at startup"
    )
    demo_key_budget_usd: float = 5.0

    log_level: str = "INFO"
    log_json: bool = True
    log_redact_pii: bool = True
    log_payloads: bool = False

    budget_alert_thresholds: str = "0.5,0.8,1.0"
    budget_webhook_url: str | None = None

    host: str = "0.0.0.0"
    port: int = Field(default=8080, validation_alias=AliasChoices("PORT", "port"))

    @field_validator(
        "redis_url", "admin_api_key", "demo_api_key", "budget_webhook_url", mode="before"
    )
    @classmethod
    def _blank_is_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("database_url")
    @classmethod
    def _async_driver(cls, value: str) -> str:
        """Render/Heroku hand out ``postgres://``; SQLAlchemy async needs an explicit driver."""
        if value.startswith("postgres://"):
            value = "postgresql://" + value[len("postgres://") :]
        if value.startswith("postgresql://"):
            value = "postgresql+asyncpg://" + value[len("postgresql://") :]
        if value.startswith("sqlite:///"):
            value = "sqlite+aiosqlite:///" + value[len("sqlite:///") :]
        return value

    @property
    def alert_thresholds(self) -> list[float]:
        values = sorted({float(v) for v in self.budget_alert_thresholds.split(",") if v.strip()})
        return [v for v in values if 0 < v <= 10]


class ProviderConfig(BaseModel):
    type: ProviderType
    base_url: str | None = None
    base_url_env: str | None = None
    api_key_env: str | None = None
    enabled: bool = True
    timeout_s: float = Field(default=60.0, gt=0)
    headers: dict[str, str] = Field(default_factory=dict)
    default_max_tokens: int = Field(default=1024, ge=1, description="Anthropic requires max_tokens")
    # FakeProvider knobs (ignored by real providers)
    latency_ms: int = Field(default=0, ge=0)
    stream_chunk_delay_ms: int = Field(default=0, ge=0)
    failure_rate: float = Field(default=0.0, ge=0, le=1)
    failure_status: int = 503
    fail_first: int = Field(default=0, ge=0, description="Fail this many calls, then succeed")
    seed: int = 0
    replies: dict[str, str] = Field(
        default_factory=dict, description="Regex on the last user message -> scripted reply"
    )

    def resolved_base_url(self) -> str | None:
        if self.base_url_env and os.environ.get(self.base_url_env):
            return os.environ[self.base_url_env]
        return self.base_url

    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env) if self.api_key_env else None

    def availability(self) -> tuple[bool, str | None]:
        """Whether the provider can be used, and why not."""
        if not self.enabled:
            return False, "disabled in config"
        if self.type == "fake":
            return True, None
        if self.api_key_env and not self.api_key():
            return False, f"{self.api_key_env} is not set"
        if not self.resolved_base_url():
            return False, f"{self.base_url_env or 'base_url'} is not set"
        return True, None


class Target(BaseModel):
    provider: str
    model: str
    weight: float = Field(default=1.0, gt=0)


class ModelRoute(BaseModel):
    targets: list[Target] = Field(min_length=1)
    fallbacks: list[str] = Field(default_factory=list)
    description: str = ""


class BreakerConfig(BaseModel):
    failure_threshold: int = Field(default=5, ge=1)
    recovery_s: float = Field(default=30.0, gt=0)


class LatencyAwareConfig(BaseModel):
    enabled: bool = False
    window: int = Field(default=100, ge=1, description="Samples kept per deployment")
    min_samples: int = Field(default=20, ge=1, description="Fresh samples before p95 counts")
    max_age_s: float = Field(default=300.0, gt=0, description="Samples older than this expire")
    penalty: float = Field(default=1.0, gt=0, description="Exponent on fastest p95 / p95")
    tolerance_ms: float = Field(
        default=25.0, ge=0, description="p95 gaps up to this are treated as noise"
    )
    min_weight_ratio: float = Field(
        default=0.05, gt=0, le=1, description="Floor as a share of the configured weight"
    )

    @model_validator(mode="after")
    def _window_holds_min_samples(self) -> LatencyAwareConfig:
        if self.min_samples > self.window:
            raise ValueError("latency_aware.min_samples cannot exceed latency_aware.window")
        return self


class RoutingConfig(BaseModel):
    max_retries: int = Field(default=2, ge=0, le=10)
    backoff_base_ms: int = Field(default=100, ge=0)
    backoff_max_ms: int = Field(default=2000, ge=0)
    request_timeout_s: float = Field(default=60.0, gt=0)
    breaker: BreakerConfig = Field(default_factory=BreakerConfig)
    latency_aware: LatencyAwareConfig = Field(default_factory=LatencyAwareConfig)


class CacheConfig(BaseModel):
    enabled: bool = True
    ttl_s: int = Field(default=3600, ge=1)
    max_entries: int = Field(default=5000, ge=1)
    only_deterministic: bool = Field(
        default=True, description="Cache only requests with temperature == 0"
    )
    case_insensitive: bool = False


class GatewayConfig(BaseModel):
    providers: dict[str, ProviderConfig]
    models: dict[str, ModelRoute]
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)

    @model_validator(mode="after")
    def _check_references(self) -> GatewayConfig:
        for alias, route in self.models.items():
            for target in route.targets:
                if target.provider not in self.providers:
                    raise ValueError(
                        f"model {alias!r} targets unknown provider {target.provider!r}"
                    )
            for fallback in route.fallbacks:
                if fallback not in self.models:
                    raise ValueError(f"model {alias!r} falls back to unknown model {fallback!r}")
                if fallback == alias:
                    raise ValueError(f"model {alias!r} cannot fall back to itself")
        return self

    @classmethod
    def load(cls, path: Path) -> GatewayConfig:
        if not path.exists():
            raise FileNotFoundError(f"gateway config not found: {path}")
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls.model_validate(data)
