"""Provider registry."""

from __future__ import annotations

import httpx

from llm_gateway.config import ProviderConfig
from llm_gateway.providers.anthropic import AnthropicProvider
from llm_gateway.providers.base import Provider
from llm_gateway.providers.fake import FakeProvider
from llm_gateway.providers.ollama import OllamaProvider
from llm_gateway.providers.openai import OpenAIProvider

__all__ = ["Provider", "build_provider"]


def build_provider(
    name: str, config: ProviderConfig, *, transport: httpx.AsyncBaseTransport | None = None
) -> Provider:
    if config.type == "fake":
        return FakeProvider(name, config)
    if config.type == "openai":
        return OpenAIProvider(name, config, transport=transport)
    if config.type == "anthropic":
        return AnthropicProvider(name, config, transport=transport)
    if config.type == "ollama":
        return OllamaProvider(name, config, transport=transport)
    raise ValueError(f"unknown provider type {config.type!r}")
