"""Structured JSON logs with optional PII redaction."""

from __future__ import annotations

import json
import logging
import re
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()) | {
    "message",
    "asctime",
    "color_message",  # uvicorn's ANSI-coloured duplicate of msg
}

# Order matters: secrets first, then card numbers before phone numbers (both are digit runs).
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("api_key", re.compile(r"\b(?:sk|pk|rk)-[A-Za-z0-9_\-]{16,}\b")),
    ("bearer", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{12,}")),
    ("email", re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")),
    ("card", re.compile(r"\b(?:\d[ -]?){12,18}\d\b")),
    (
        "phone",
        re.compile(
            r"(?<![\w])(?:\+?88)?01[3-9]\d{2}[ -]?\d{6}\b"  # Bangladeshi mobile numbers
            r"|\+\d{1,3}[ -]?\d{3,4}[ -]?\d{3,4}[ -]?\d{3,4}\b"  # other international numbers
        ),
    ),
]


def _luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = ord(ch) - 48
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


def redact(text: str) -> str:
    """Replace secrets and personal data with ``[REDACTED:<kind>]`` markers."""
    for kind, pattern in _PATTERNS:
        if kind == "card":
            text = pattern.sub(
                lambda m: (
                    "[REDACTED:card]" if _luhn_ok(re.sub(r"\D", "", m.group(0))) else m.group(0)
                ),
                text,
            )
        else:
            text = pattern.sub(f"[REDACTED:{kind}]", text)
    return text


def redact_value(value: Any) -> Any:
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {k: redact_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_value(v) for v in value]
    return value


class JsonFormatter(logging.Formatter):
    def __init__(self, *, redact_pii: bool) -> None:
        super().__init__()
        self.redact_pii = redact_pii

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname.lower(),
            "logger": record.name,
            "msg": record.getMessage(),
        }
        request_id = request_id_var.get()
        if request_id:
            payload["request_id"] = request_id
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        if self.redact_pii:
            payload = redact_value(payload)
        return json.dumps(payload, ensure_ascii=False, default=str)


class PlainRedactingFormatter(logging.Formatter):
    def __init__(self, *, redact_pii: bool) -> None:
        super().__init__("%(asctime)s %(levelname)s %(name)s: %(message)s")
        self.redact_pii = redact_pii

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        return redact(text) if self.redact_pii else text


def configure_logging(
    level: str = "INFO", *, json_logs: bool = True, redact_pii: bool = True
) -> None:
    handler = logging.StreamHandler(sys.stdout)
    formatter: logging.Formatter = (
        JsonFormatter(redact_pii=redact_pii)
        if json_logs
        else PlainRedactingFormatter(redact_pii=redact_pii)
    )
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())
    for noisy in ("httpx", "httpcore", "aiosqlite", "sqlalchemy.engine"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
