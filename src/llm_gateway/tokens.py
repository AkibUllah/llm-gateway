"""Token counting: tiktoken when installed (``pip install llm-gateway[tokens]``), else a heuristic.

The heuristic (about four characters per token for Latin text, one token per character for
scripts such as Bangla where BPE vocabularies are sparse) is within ~15% of cl100k on typical
chat traffic, which is enough for rate limiting and pre-flight budget checks. Billing uses the
provider-reported usage whenever the provider sends it.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from functools import lru_cache
from typing import Any

from llm_gateway.schemas import ChatMessage

MESSAGE_OVERHEAD = 4
REPLY_PRIMER = 3
_WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@lru_cache(maxsize=1)
def _encoder() -> Any | None:
    try:
        import tiktoken  # type: ignore[import-not-found]

        return tiktoken.get_encoding("cl100k_base")
    except Exception:  # noqa: BLE001 - optional dependency, any failure means "not available"
        return None


def heuristic_tokens(text: str) -> int:
    if not text:
        return 0
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    other_chars = len(text) - ascii_chars
    by_chars = ascii_chars / 4 + other_chars
    by_words = len(_WORD_RE.findall(text)) * 0.75
    return max(1, math.ceil(max(by_chars, by_words)))


def count_text(text: str) -> int:
    encoder = _encoder()
    if encoder is not None:
        return len(encoder.encode(text))
    return heuristic_tokens(text)


def count_messages(messages: Iterable[ChatMessage]) -> int:
    total = REPLY_PRIMER
    for message in messages:
        total += MESSAGE_OVERHEAD + count_text(message.text())
        if message.name:
            total += 1
        for call in message.tool_calls or []:
            total += count_text(call.function.name) + count_text(call.function.arguments)
    return total


def backend() -> str:
    return "tiktoken" if _encoder() is not None else "heuristic"
