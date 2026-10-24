"""Error types. Client-facing errors render in the OpenAI error shape."""

from __future__ import annotations

from typing import Any


class GatewayError(Exception):
    """An error returned to the API client as ``{"error": {...}}`` with an HTTP status."""

    def __init__(
        self,
        status: int,
        message: str,
        *,
        type: str = "invalid_request_error",
        code: str | None = None,
        param: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.type = type
        self.code = code
        self.param = param
        self.headers = headers or {}

    def body(self) -> dict[str, Any]:
        return {
            "error": {
                "message": self.message,
                "type": self.type,
                "param": self.param,
                "code": self.code,
            }
        }


def auth_error(message: str = "invalid or missing API key") -> GatewayError:
    return GatewayError(401, message, type="authentication_error", code="invalid_api_key")


class ProviderError(Exception):
    """An upstream call failed. ``retryable`` decides between retry/fallback and failing fast."""

    def __init__(
        self,
        provider: str,
        message: str,
        *,
        status: int | None = None,
        retryable: bool = True,
        retry_after: float | None = None,
        fallback: bool | None = None,
    ) -> None:
        super().__init__(f"{provider}: {message}")
        self.provider = provider
        self.message = message
        self.status = status
        self.retryable = retryable
        self.retry_after = retry_after
        # Errors that are the provider's fault (auth, missing model) are worth a fallback even
        # though retrying the same provider is pointless. Malformed requests are not.
        self.fallback = retryable if fallback is None else fallback

    @property
    def is_rate_limit(self) -> bool:
        return self.status == 429


class UnsupportedOperation(ProviderError):
    """The provider cannot serve this kind of request (e.g. embeddings on Anthropic)."""

    def __init__(self, provider: str, message: str) -> None:
        super().__init__(provider, message, status=400, retryable=False, fallback=True)


def classify_http_error(
    provider: str, status: int, message: str, retry_after: float | None = None
) -> ProviderError:
    """Map an upstream HTTP status to retry/fallback behaviour."""
    if status == 429 or status == 408 or status == 409 or status >= 500:
        return ProviderError(
            provider, message, status=status, retryable=True, retry_after=retry_after
        )
    if status in (401, 403, 404):
        return ProviderError(provider, message, status=status, retryable=False, fallback=True)
    return ProviderError(provider, message, status=status, retryable=False, fallback=False)


def parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None
