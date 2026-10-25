"""Virtual API keys: hashed at rest, with budgets, rate limits, model allow-lists and expiry."""

from __future__ import annotations

import fnmatch
import hashlib
import logging
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError

from llm_gateway.db import ApiKey, BudgetAlert, Database, utcnow
from llm_gateway.errors import GatewayError, auth_error

logger = logging.getLogger(__name__)

KEY_PREFIX = "sk-gw-"


def hash_key(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


def generate_key() -> str:
    return KEY_PREFIX + secrets.token_urlsafe(32)


def as_utc(value: datetime | None) -> datetime | None:
    """SQLite drops tzinfo on read; treat naive timestamps as UTC."""
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


class KeyCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    allowed_models: list[str] = Field(default_factory=list, description="Globs; empty = all")
    budget_usd: float | None = Field(default=None, ge=0)
    budget_period: Literal["total", "monthly"] = "total"
    rpm_limit: int | None = Field(default=None, ge=1)
    tpm_limit: int | None = Field(default=None, ge=1)
    expires_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class KeyUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    active: bool | None = None
    allowed_models: list[str] | None = None
    budget_usd: float | None = Field(default=None, ge=0)
    budget_period: Literal["total", "monthly"] | None = None
    rpm_limit: int | None = Field(default=None, ge=1)
    tpm_limit: int | None = Field(default=None, ge=1)
    expires_at: datetime | None = None
    metadata: dict[str, Any] | None = None
    reset_spend: bool = False


class KeyOut(BaseModel):
    id: str
    name: str
    key_prefix: str
    active: bool
    allowed_models: list[str]
    budget_usd: float | None
    budget_period: str
    spent_usd: float
    budget_remaining_usd: float | None
    period_start: datetime
    rpm_limit: int | None
    tpm_limit: int | None
    metadata: dict[str, Any]
    created_at: datetime
    expires_at: datetime | None
    last_used_at: datetime | None

    @classmethod
    def of(cls, key: ApiKey) -> KeyOut:
        remaining = None
        if key.budget_usd is not None:
            remaining = round(max(0.0, key.budget_usd - (key.spent_usd or 0.0)), 8)
        return cls(
            id=key.id,
            name=key.name,
            key_prefix=key.key_prefix,
            active=key.active,
            allowed_models=list(key.allowed_models or []),
            budget_usd=key.budget_usd,
            budget_period=key.budget_period,
            spent_usd=round(key.spent_usd or 0.0, 8),
            budget_remaining_usd=remaining,
            period_start=as_utc(key.period_start),  # type: ignore[arg-type]
            rpm_limit=key.rpm_limit,
            tpm_limit=key.tpm_limit,
            metadata=dict(key.metadata_ or {}),
            created_at=as_utc(key.created_at),  # type: ignore[arg-type]
            expires_at=as_utc(key.expires_at),
            last_used_at=as_utc(key.last_used_at),
        )


class KeyCreated(KeyOut):
    key: str = Field(description="The secret. Shown once; only its hash is stored.")


@dataclass(frozen=True)
class KeyContext:
    """What a request needs to know about its key (detached from the ORM session)."""

    id: str
    name: str
    allowed_models: tuple[str, ...]
    budget_usd: float | None
    spent_usd: float
    rpm_limit: int | None
    tpm_limit: int | None

    def allows(self, model: str) -> bool:
        return not self.allowed_models or any(
            fnmatch.fnmatchcase(model, pattern) for pattern in self.allowed_models
        )

    def check_budget(self) -> None:
        if self.budget_usd is not None and self.spent_usd >= self.budget_usd:
            raise GatewayError(
                402,
                f"budget exhausted for key {self.name!r} "
                f"(${self.spent_usd:.4f} of ${self.budget_usd:.2f})",
                type="insufficient_quota",
                code="budget_exceeded",
            )


def _month_start(now: datetime) -> datetime:
    return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


class KeyService:
    def __init__(self, db: Database, *, alert_thresholds: list[float] | None = None) -> None:
        self.db = db
        self.alert_thresholds = alert_thresholds or []

    async def create(self, data: KeyCreate, *, raw_key: str | None = None) -> KeyCreated:
        raw = raw_key or generate_key()
        now = utcnow()
        key = ApiKey(
            name=data.name,
            key_prefix=raw[:12],
            key_hash=hash_key(raw),
            allowed_models=data.allowed_models,
            budget_usd=data.budget_usd,
            budget_period=data.budget_period,
            spent_usd=0.0,
            period_start=_month_start(now) if data.budget_period == "monthly" else now,
            rpm_limit=data.rpm_limit,
            tpm_limit=data.tpm_limit,
            expires_at=data.expires_at,
            metadata_=data.metadata,
            created_at=now,
        )
        async with self.db.session() as session:
            session.add(key)
            try:
                await session.commit()
            except IntegrityError as exc:
                raise GatewayError(
                    409, "a key with this value already exists", code="duplicate_key"
                ) from exc
        return KeyCreated(**KeyOut.of(key).model_dump(), key=raw)

    async def ensure(self, raw_key: str, data: KeyCreate) -> None:
        """Create the key if it does not exist yet (used for the demo key)."""
        async with self.db.session() as session:
            found = await session.scalar(select(ApiKey).where(ApiKey.key_hash == hash_key(raw_key)))
        if found is None:
            created = await self.create(data, raw_key=raw_key)
            logger.info("created demo key %s (%s)", created.id, data.name)

    async def authenticate(self, raw_key: str | None) -> KeyContext:
        if not raw_key:
            raise auth_error("missing API key; send 'Authorization: Bearer sk-gw-...'")
        now = utcnow()
        async with self.db.session() as session:
            key = await session.scalar(select(ApiKey).where(ApiKey.key_hash == hash_key(raw_key)))
            if key is None or not key.active:
                raise auth_error()
            expires = as_utc(key.expires_at)
            if expires is not None and expires <= now:
                raise auth_error("API key has expired")
            if key.budget_period == "monthly" and as_utc(key.period_start) < _month_start(now):
                key.spent_usd = 0.0
                key.period_start = _month_start(now)
            key.last_used_at = now
            await session.commit()
            return KeyContext(
                id=key.id,
                name=key.name,
                allowed_models=tuple(key.allowed_models or ()),
                budget_usd=key.budget_usd,
                spent_usd=key.spent_usd or 0.0,
                rpm_limit=key.rpm_limit,
                tpm_limit=key.tpm_limit,
            )

    async def add_spend(self, key_id: str, cost: float) -> list[float]:
        """Atomically add spend; return the alert thresholds crossed by this request."""
        if cost <= 0:
            return []
        async with self.db.session() as session:
            row = (
                await session.execute(
                    update(ApiKey)
                    .where(ApiKey.id == key_id)
                    .values(spent_usd=ApiKey.spent_usd + cost)
                    .returning(
                        ApiKey.spent_usd, ApiKey.budget_usd, ApiKey.period_start, ApiKey.name
                    )
                )
            ).one_or_none()
            await session.commit()
            if row is None or row.budget_usd in (None, 0):
                return []
            before = (row.spent_usd - cost) / row.budget_usd
            after = row.spent_usd / row.budget_usd
            crossed = [t for t in self.alert_thresholds if before < t <= after]
            for threshold in crossed:
                session.add(
                    BudgetAlert(
                        key_id=key_id,
                        threshold=threshold,
                        period_start=row.period_start,
                        spent_usd=row.spent_usd,
                        budget_usd=row.budget_usd,
                    )
                )
                logger.warning(
                    "budget alert",
                    extra={
                        "key_id": key_id,
                        "key_name": row.name,
                        "threshold": threshold,
                        "spent_usd": round(row.spent_usd, 6),
                        "budget_usd": row.budget_usd,
                    },
                )
            try:
                await session.commit()
            except IntegrityError:
                await session.rollback()  # another replica recorded the same alert
            return crossed

    async def list(self) -> list[KeyOut]:
        async with self.db.session() as session:
            keys = (await session.scalars(select(ApiKey).order_by(ApiKey.created_at))).all()
        return [KeyOut.of(k) for k in keys]

    async def get(self, key_id: str) -> KeyOut:
        async with self.db.session() as session:
            key = await session.get(ApiKey, key_id)
        if key is None:
            raise GatewayError(404, f"key {key_id} not found", code="key_not_found")
        return KeyOut.of(key)

    async def update(self, key_id: str, data: KeyUpdate) -> KeyOut:
        async with self.db.session() as session:
            key = await session.get(ApiKey, key_id)
            if key is None:
                raise GatewayError(404, f"key {key_id} not found", code="key_not_found")
            changes = data.model_dump(exclude_unset=True, exclude={"reset_spend", "metadata"})
            for field, value in changes.items():
                setattr(key, field, value)
            if data.metadata is not None:
                key.metadata_ = data.metadata
            if data.reset_spend:
                key.spent_usd = 0.0
                key.period_start = utcnow()
            await session.commit()
            return KeyOut.of(key)

    async def revoke(self, key_id: str) -> KeyOut:
        return await self.update(key_id, KeyUpdate(active=False))

    async def alerts(self, key_id: str | None = None) -> list[dict[str, Any]]:
        async with self.db.session() as session:
            query = select(BudgetAlert).order_by(BudgetAlert.created_at.desc()).limit(200)
            if key_id:
                query = query.where(BudgetAlert.key_id == key_id)
            rows = (await session.scalars(query)).all()
        return [
            {
                "key_id": a.key_id,
                "threshold": a.threshold,
                "spent_usd": a.spent_usd,
                "budget_usd": a.budget_usd,
                "period_start": as_utc(a.period_start),
                "created_at": as_utc(a.created_at),
            }
            for a in rows
        ]
