import httpx

from providers.common.failover import is_failover_retryable
from providers.exceptions import (
    AuthenticationError,
    InvalidRequestError,
    OverloadedError,
    RateLimitError,
)


def test_is_failover_retryable_for_transient_provider_errors():
    assert is_failover_retryable(RateLimitError("rate limited")) is True
    assert is_failover_retryable(OverloadedError("overloaded")) is True


def test_is_failover_retryable_for_non_transient_provider_errors():
    assert is_failover_retryable(AuthenticationError("auth")) is False
    assert is_failover_retryable(InvalidRequestError("bad request")) is False


def test_is_failover_retryable_for_httpx_transient_errors():
    request = httpx.Request("POST", "http://example.com")
    assert is_failover_retryable(httpx.ConnectError("connect", request=request)) is True
    assert is_failover_retryable(httpx.ReadTimeout("timeout", request=request)) is True
