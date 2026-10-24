"""Cost table loaded from YAML: USD per one million input/output tokens, per provider model."""

from __future__ import annotations

import fnmatch
import logging
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class Price(BaseModel):
    input: float = Field(ge=0, description="USD per 1M prompt tokens")
    output: float = Field(default=0, ge=0, description="USD per 1M completion tokens")


class PriceTable:
    def __init__(self, prices: dict[str, Price]) -> None:
        self.prices = prices
        self._warned: set[str] = set()

    @classmethod
    def load(cls, path: Path | None) -> PriceTable:
        if path is None or not path.exists():
            return cls({})
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls({str(k): Price.model_validate(v) for k, v in raw.items()})

    def lookup(self, model: str) -> Price | None:
        """Exact match first, then glob patterns such as ``claude-sonnet-4-*``."""
        if model in self.prices:
            return self.prices[model]
        for pattern, price in self.prices.items():
            if any(ch in pattern for ch in "*?[") and fnmatch.fnmatchcase(model, pattern):
                return price
        return None

    def cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        price = self.lookup(model)
        if price is None:
            if model not in self._warned:
                logger.warning("no price configured for model %s; recording cost 0", model)
                self._warned.add(model)
            return 0.0
        usd = (prompt_tokens * price.input + completion_tokens * price.output) / 1_000_000
        return round(usd, 8)
