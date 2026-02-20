"""Utilities for parsing provider-prefixed model identifiers."""

VALID_PROVIDER_TYPES = ("nvidia_nim", "open_router", "lmstudio")


def is_valid_provider_type(provider_type: str) -> bool:
    """Return True when provider type is one of the supported canonical names."""
    return provider_type in VALID_PROVIDER_TYPES


def parse_prefixed_model(value: str) -> tuple[str, str]:
    """Parse `provider/model/id` into `(provider_type, model_id)`."""
    provider_type, sep, provider_model = value.partition("/")
    if not sep:
        raise ValueError("Model must use 'provider/model' format with provider prefix.")
    if not is_valid_provider_type(provider_type):
        valid = ", ".join(VALID_PROVIDER_TYPES)
        raise ValueError(
            f"Unknown provider prefix '{provider_type}'. Valid prefixes: {valid}."
        )
    if not provider_model.strip():
        raise ValueError("Model id cannot be empty after provider prefix.")
    return provider_type, provider_model
