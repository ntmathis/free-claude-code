"""Failover-related helpers shared across providers and routes."""

import httpx
import openai

from providers.exceptions import ProviderError


def is_failover_retryable(exc: Exception) -> bool:
    """Return True when the error is transient and a roster fallback should be attempted."""
    if isinstance(exc, ProviderError):
        status = exc.status_code
        return status == 408 or status == 429 or status >= 500

    if isinstance(exc, (openai.RateLimitError, openai.APITimeoutError)):
        return True

    if isinstance(
        exc,
        (
            httpx.TimeoutException,
            httpx.ConnectError,
            httpx.ReadError,
            httpx.RemoteProtocolError,
        ),
    ):
        return True

    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code == 408 or status_code == 429 or status_code >= 500

    return False
