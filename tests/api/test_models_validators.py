from unittest.mock import patch

import pytest

from api.models.anthropic import Message, MessagesRequest, TokenCountRequest
from config.settings import Settings


@pytest.fixture
def mock_settings():
    settings = Settings(model="nvidia_nim/target-model-from-settings")
    settings.opus_model = "open_router/opus-from-settings"
    settings.sonnet_model = "nvidia_nim/sonnet-from-settings"
    settings.haiku_model = None
    return settings


def test_messages_request_map_model_claude_to_default(mock_settings):
    with patch("api.models.anthropic.get_settings", return_value=mock_settings):
        request = MessagesRequest(
            model="claude-3-opus",
            max_tokens=100,
            messages=[Message(role="user", content="hello")],
        )

        assert request.model == "opus-from-settings"
        assert request.target_provider_type == "open_router"
        assert request.original_model == "claude-3-opus"
        assert request.target_candidates is not None
        assert request.target_candidates[0]["provider_type"] == "open_router"


def test_messages_request_map_model_non_claude_unchanged(mock_settings):
    with patch("api.models.anthropic.get_settings", return_value=mock_settings):
        request = MessagesRequest(
            model="gpt-4",
            max_tokens=100,
            messages=[Message(role="user", content="hello")],
        )

        assert request.model == "gpt-4"
        assert request.target_provider_type == "nvidia_nim"


def test_messages_request_map_model_with_provider_prefix(mock_settings):
    with patch("api.models.anthropic.get_settings", return_value=mock_settings):
        request = MessagesRequest(
            model="anthropic/claude-3-haiku",
            max_tokens=100,
            messages=[Message(role="user", content="hello")],
        )

        # Since haiku_model is None in mock_settings, maps to default
        assert request.model == "target-model-from-settings"
        assert request.target_provider_type == "nvidia_nim"


def test_messages_request_uses_ordered_roster_candidates(mock_settings):
    mock_settings.sonnet_model = (
        "open_router/anthropic/claude-3.5-sonnet,nvidia_nim/z-ai/glm4.7"
    )
    with patch("api.models.anthropic.get_settings", return_value=mock_settings):
        request = MessagesRequest(
            model="claude-3-sonnet",
            max_tokens=100,
            messages=[Message(role="user", content="hello")],
        )

        assert request.model == "anthropic/claude-3.5-sonnet"
        assert request.target_provider_type == "open_router"
        assert request.target_candidates is not None
        assert request.target_candidates[1]["provider_model"] == "z-ai/glm4.7"


def test_token_count_request_model_validation(mock_settings):
    with patch("api.models.anthropic.get_settings", return_value=mock_settings):
        request = TokenCountRequest(
            model="claude-3-sonnet", messages=[Message(role="user", content="hello")]
        )

        assert request.model == "sonnet-from-settings"
        assert request.target_provider_type == "nvidia_nim"


def test_messages_request_model_mapping_logs(mock_settings):
    with (
        patch("api.models.anthropic.get_settings", return_value=mock_settings),
        patch("api.models.anthropic.logger.debug") as mock_log,
    ):
        MessagesRequest(
            model="claude-2.1",
            max_tokens=100,
            messages=[Message(role="user", content="hello")],
        )

        mock_log.assert_called()
        args = mock_log.call_args[0][0]
        assert "MODEL MAPPING" in args
        assert "claude-2.1" in args
        assert "target-model-from-settings" in args
