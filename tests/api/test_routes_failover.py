from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from api.app import app
from api.dependencies import get_provider
from config.settings import Settings
from providers.exceptions import InvalidRequestError, RateLimitError


def _build_payload() -> dict:
    return {
        "model": "claude-3-sonnet",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 32,
        "stream": True,
    }


def test_stream_failover_on_retryable_error_uses_next_candidate():
    primary_provider = Mock()
    fallback_provider = Mock()

    async def _primary_fail(*args, **kwargs):
        raise RateLimitError("Too Many Requests")
        if False:
            yield ""

    async def _fallback_ok(*args, **kwargs):
        yield "event: message_start\ndata: {}\n\n"
        yield 'event: content_block_delta\ndata: {"text":"fallback-ok"}\n\n'
        yield "event: message_stop\ndata: {}\n\n"

    primary_provider.stream_response = _primary_fail
    fallback_provider.stream_response = _fallback_ok

    settings = Settings(model="nvidia_nim/default-model")
    settings.provider_type = "nvidia_nim"
    settings.sonnet_model = (
        "open_router/anthropic/claude-3.5-sonnet,nvidia_nim/z-ai/glm4.7"
    )

    app.dependency_overrides[get_provider] = lambda: fallback_provider
    try:
        with (
            patch("api.models.anthropic.get_settings", return_value=settings),
            patch("api.routes.get_provider_for_type", return_value=primary_provider),
        ):
            client = TestClient(app)
            response = client.post("/v1/messages", json=_build_payload())

        assert response.status_code == 200
        assert b"fallback-ok" in b"".join(response.iter_bytes())
    finally:
        app.dependency_overrides.pop(get_provider, None)


def test_stream_failover_skips_non_retryable_errors():
    primary_provider = Mock()
    fallback_provider = Mock()

    async def _primary_invalid(*args, **kwargs):
        raise InvalidRequestError("Bad request")
        if False:
            yield ""

    primary_provider.stream_response = _primary_invalid

    settings = Settings(model="nvidia_nim/default-model")
    settings.provider_type = "nvidia_nim"
    settings.sonnet_model = (
        "open_router/anthropic/claude-3.5-sonnet,nvidia_nim/z-ai/glm4.7"
    )

    app.dependency_overrides[get_provider] = lambda: fallback_provider
    try:
        with (
            patch("api.models.anthropic.get_settings", return_value=settings),
            patch("api.routes.get_provider_for_type", return_value=primary_provider),
        ):
            client = TestClient(app)
            response = client.post("/v1/messages", json=_build_payload())

        assert response.status_code == 400
        assert response.json()["error"]["type"] == "invalid_request_error"
    finally:
        app.dependency_overrides.pop(get_provider, None)


def test_stream_midstream_error_emits_sse_error_instead_of_socket_close():
    provider = Mock()

    async def _midstream_error(*args, **kwargs):
        yield "event: message_start\ndata: {}\n\n"
        raise RateLimitError("The socket connection was closed unexpectedly.")
        if False:
            yield ""

    provider.stream_response = _midstream_error

    settings = Settings(model="nvidia_nim/default-model")
    settings.provider_type = "nvidia_nim"
    settings.sonnet_model = "nvidia_nim/z-ai/glm4.7"

    app.dependency_overrides[get_provider] = lambda: provider
    try:
        with patch("api.models.anthropic.get_settings", return_value=settings):
            client = TestClient(app)
            response = client.post("/v1/messages", json=_build_payload())

        assert response.status_code == 200
        body = b"".join(response.iter_bytes())
        assert b"event: error" in body
    finally:
        app.dependency_overrides.pop(get_provider, None)
