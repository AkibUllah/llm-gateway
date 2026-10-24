"""Async SQLAlchemy engine and ORM models (SQLite for dev/tests, PostgreSQL in compose)."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
    event,
)
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

Money = Numeric(18, 8, asdecimal=False)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_key_id() -> str:
    return "key_" + uuid.uuid4().hex[:16]


class Base(DeclarativeBase):
    pass


class ApiKey(Base):
    __tablename__ = "api_keys"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_key_id)
    name: Mapped[str] = mapped_column(String(120))
    key_prefix: Mapped[str] = mapped_column(String(16))
    key_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    allowed_models: Mapped[list[str]] = mapped_column(JSON, default=list)
    budget_usd: Mapped[float | None] = mapped_column(Money, nullable=True)
    budget_period: Mapped[str] = mapped_column(String(10), default="total")
    spent_usd: Mapped[float] = mapped_column(Money, default=0.0)
    period_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    rpm_limit: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tpm_limit: Mapped[int | None] = mapped_column(Integer, nullable=True)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class UsageRecord(Base):
    __tablename__ = "usage_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(String(40), index=True)
    key_id: Mapped[str | None] = mapped_column(
        String(32), ForeignKey("api_keys.id", ondelete="SET NULL"), nullable=True, index=True
    )
    endpoint: Mapped[str] = mapped_column(String(20))
    alias: Mapped[str] = mapped_column(String(120), index=True)
    provider: Mapped[str | None] = mapped_column(String(60), nullable=True)
    provider_model: Mapped[str | None] = mapped_column(String(120), nullable=True)
    stream: Mapped[bool] = mapped_column(Boolean, default=False)
    status_code: Mapped[int] = mapped_column(Integer)
    error_code: Mapped[str | None] = mapped_column(String(60), nullable=True)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Money, default=0.0)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    cached: Mapped[bool] = mapped_column(Boolean, default=False)
    fallback: Mapped[bool] = mapped_column(Boolean, default=False)
    attempts: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True
    )


class BudgetAlert(Base):
    __tablename__ = "budget_alerts"
    __table_args__ = (UniqueConstraint("key_id", "threshold", "period_start"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    key_id: Mapped[str] = mapped_column(String(32), ForeignKey("api_keys.id", ondelete="CASCADE"))
    threshold: Mapped[float] = mapped_column(Float)
    period_start: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    spent_usd: Mapped[float] = mapped_column(Money)
    budget_usd: Mapped[float] = mapped_column(Money)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


def _sqlite_pragmas(dbapi_connection: Any, _record: Any) -> None:
    # WAL lets readers proceed during writes; NORMAL sync is durable across app crashes and
    # avoids an fsync per usage row. Production deployments use PostgreSQL anyway.
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


class Database:
    def __init__(self, url: str) -> None:
        kwargs: dict[str, Any] = {"pool_pre_ping": True}
        if url.startswith("sqlite"):
            kwargs = {"connect_args": {"timeout": 15}}
        self.engine: AsyncEngine = create_async_engine(url, **kwargs)
        if url.startswith("sqlite"):
            event.listen(self.engine.sync_engine, "connect", _sqlite_pragmas)
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)

    async def create_all(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    def session(self) -> AsyncSession:
        return self.sessions()

    async def ping(self) -> bool:
        from sqlalchemy import text

        try:
            async with self.engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return True
        except Exception:  # noqa: BLE001 - health check reports, never raises
            return False

    async def dispose(self) -> None:
        await self.engine.dispose()
