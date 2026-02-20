import pytest

from providers.common import parse_prefixed_model


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
