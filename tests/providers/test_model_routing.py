import pytest

from providers.common import parse_prefixed_model, parse_prefixed_model_roster


def test_parse_prefixed_model_valid():
    provider_type, model_id = parse_prefixed_model(
        "open_router/anthropic/claude-3-opus"
    )
    assert provider_type == "open_router"
    assert model_id == "anthropic/claude-3-opus"


@pytest.mark.parametrize(
    "value",
    [
        "stepfun-ai/step-3.5-flash",
        "openrouter/anthropic/claude-3-opus",
        "nvidia_nim/",
    ],
)
def test_parse_prefixed_model_invalid(value):
    with pytest.raises(ValueError):
        parse_prefixed_model(value)


def test_parse_prefixed_model_roster_valid():
    roster = parse_prefixed_model_roster(
        "open_router/anthropic/claude-3-opus, nvidia_nim/z-ai/glm4.7"
    )
    assert roster == [
        ("open_router", "anthropic/claude-3-opus"),
        ("nvidia_nim", "z-ai/glm4.7"),
    ]


def test_parse_prefixed_model_roster_deduplicates_stably():
    roster = parse_prefixed_model_roster(
        "open_router/anthropic/claude-3-opus,open_router/anthropic/claude-3-opus,nvidia_nim/z-ai/glm4.7"
    )
    assert roster == [
        ("open_router", "anthropic/claude-3-opus"),
        ("nvidia_nim", "z-ai/glm4.7"),
    ]


@pytest.mark.parametrize(
    "value",
    [
        "open_router/anthropic/claude-3-opus,",
        "open_router/anthropic/claude-3-opus,,nvidia_nim/z-ai/glm4.7",
    ],
)
def test_parse_prefixed_model_roster_invalid_empty_entry(value):
    with pytest.raises(ValueError):
        parse_prefixed_model_roster(value)
