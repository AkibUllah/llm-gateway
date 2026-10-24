"""Usage records and the aggregated reports behind /admin/usage and the dashboard."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Literal

from pydantic import BaseModel
from sqlalchemy import case, func, select

from llm_gateway.db import ApiKey, Database, UsageRecord, utcnow

GroupBy = Literal["key", "model", "provider", "day"]


class UsageRow(BaseModel):
    group: str | None
    label: str | None = None
    requests: int
    errors: int
    cache_hits: int
    fallbacks: int
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    avg_latency_ms: float


class UsageReport(BaseModel):
    group_by: GroupBy
    since: datetime
    until: datetime
    rows: list[UsageRow]
    totals: UsageRow


class UsageStore:
    def __init__(self, db: Database) -> None:
        self.db = db

    async def record(self, **fields: Any) -> None:
        async with self.db.session() as session:
            session.add(UsageRecord(**fields))
            await session.commit()

    async def report(
        self,
        group_by: GroupBy = "model",
        since: datetime | None = None,
        until: datetime | None = None,
        key_id: str | None = None,
    ) -> UsageReport:
        until = until or utcnow()
        since = since or until - timedelta(days=30)
        group_col = {
            "key": UsageRecord.key_id,
            "model": UsageRecord.alias,
            "provider": UsageRecord.provider,
            "day": func.date(UsageRecord.created_at),
        }[group_by]

        metrics = (
            func.count(UsageRecord.id),
            func.sum(case((UsageRecord.status_code >= 400, 1), else_=0)),
            func.sum(case((UsageRecord.cached.is_(True), 1), else_=0)),
            func.sum(case((UsageRecord.fallback.is_(True), 1), else_=0)),
            func.coalesce(func.sum(UsageRecord.prompt_tokens), 0),
            func.coalesce(func.sum(UsageRecord.completion_tokens), 0),
            func.coalesce(func.sum(UsageRecord.cost_usd), 0),
            func.coalesce(func.avg(UsageRecord.latency_ms), 0),
        )
        conditions = [UsageRecord.created_at >= since, UsageRecord.created_at <= until]
        if key_id:
            conditions.append(UsageRecord.key_id == key_id)

        async with self.db.session() as session:
            grouped = (
                await session.execute(
                    select(group_col.label("g"), *metrics)
                    .where(*conditions)
                    .group_by(group_col)
                    .order_by(group_col)
                )
            ).all()
            total = (await session.execute(select(*metrics).where(*conditions))).one()
            names: dict[str, str] = {}
            if group_by == "key":
                ids = [r[0] for r in grouped if r[0]]
                if ids:
                    names = dict(
                        (
                            await session.execute(
                                select(ApiKey.id, ApiKey.name).where(ApiKey.id.in_(ids))
                            )
                        ).all()
                    )

        def row(group: Any, values: Any) -> UsageRow:
            requests, errors, hits, fallbacks, prompt, completion, cost, latency = values
            return UsageRow(
                group=str(group) if group is not None else None,
                label=names.get(group) if group_by == "key" else None,
                requests=int(requests or 0),
                errors=int(errors or 0),
                cache_hits=int(hits or 0),
                fallbacks=int(fallbacks or 0),
                prompt_tokens=int(prompt or 0),
                completion_tokens=int(completion or 0),
                cost_usd=round(float(cost or 0), 6),
                avg_latency_ms=round(float(latency or 0), 2),
            )

        return UsageReport(
            group_by=group_by,
            since=since,
            until=until,
            rows=[row(r[0], r[1:]) for r in grouped],
            totals=row("total", tuple(total)),
        )
