"""Model name normalization utilities.

Centralizes model name mapping logic to avoid duplication across the codebase.
"""

import os
from dataclasses import dataclass

from providers.common.model_routing import (
    parse_prefixed_model,
    parse_prefixed_model_roster,
)

# Provider prefixes to strip from model names
_PROVIDER_PREFIXES = ["anthropic/", "openai/", "gemini/"]

# Claude model identifiers
_CLAUDE_IDENTIFIERS = ["haiku", "sonnet", "opus", "claude"]


@dataclass(frozen=True)
class ResolvedModelTarget:
    """Resolved target for a request after model mapping."""

    provider_type: str
    provider_model: str
    mapped_model: str


def _resolve_roster(
    roster: str,
) -> list[ResolvedModelTarget]:
    resolved: list[ResolvedModelTarget] = []
    for provider_type, provider_model in parse_prefixed_model_roster(roster):
        resolved.append(
            ResolvedModelTarget(
                provider_type=provider_type,
                provider_model=provider_model,
                mapped_model=f"{provider_type}/{provider_model}",
            )
        )
    return resolved


def strip_provider_prefixes(model: str) -> str:
    """
    Strip provider prefixes from model name.

    Args:
        model: The model name, possibly with prefix

    Returns:
        Model name without provider prefix
    """
    for prefix in _PROVIDER_PREFIXES:
        if model.startswith(prefix):
            return model[len(prefix) :]
    return model


def is_claude_model(model: str) -> bool:
    """
    Check if a model name identifies as a Claude model.

    Args:
        model: The (prefix-stripped) model name

    Returns:
        True if this is a Claude model
    """
    model_lower = model.lower()
    return any(name in model_lower for name in _CLAUDE_IDENTIFIERS)


def normalize_model_name(
    model: str,
    default_model: str | None = None,
    opus_model: str | None = None,
    sonnet_model: str | None = None,
    haiku_model: str | None = None,
) -> str:
    """
    Normalize a model name by stripping prefixes and mapping to specific/default models if needed.

    This is the central function for model name normalization across the API.
    It strips provider prefixes and maps Claude model names to the configured model.

    Args:
        model: The model name (may include provider prefix)
        default_model: The default model to use for Claude models.
                       If None, uses settings.model from config.
        opus_model: Specific override for opus models
        sonnet_model: Specific override for sonnet models
        haiku_model: Specific override for haiku models

    Returns:
        Normalized model name (original if not a Claude model, mapped if Claude)
    """
    return resolve_model_target(
        model,
        default_model=default_model,
        opus_model=opus_model,
        sonnet_model=sonnet_model,
        haiku_model=haiku_model,
    ).mapped_model


def resolve_model_targets(
    model: str,
    default_model: str | None = None,
    opus_model: str | None = None,
    sonnet_model: str | None = None,
    haiku_model: str | None = None,
) -> list[ResolvedModelTarget]:
    """Resolve ordered provider/model candidates for a request model."""
    clean = strip_provider_prefixes(model)

    if default_model is None:
        default_model = os.getenv("MODEL", "nvidia_nim/moonshotai/kimi-k2-thinking")
    default_roster = _resolve_roster(default_model)
    default_provider = default_roster[0].provider_type

    if is_claude_model(clean):
        clean_lower = clean.lower()
        if "opus" in clean_lower and opus_model is not None:
            return _resolve_roster(opus_model)
        if "sonnet" in clean_lower and sonnet_model is not None:
            return _resolve_roster(sonnet_model)
        if "haiku" in clean_lower and haiku_model is not None:
            return _resolve_roster(haiku_model)
        return default_roster

    try:
        provider_type, provider_model = parse_prefixed_model(model)
    except ValueError:
        return [
            ResolvedModelTarget(
                provider_type=default_provider,
                provider_model=model,
                mapped_model=f"{default_provider}/{model}",
            )
        ]

    return [
        ResolvedModelTarget(
            provider_type=provider_type,
            provider_model=provider_model,
            mapped_model=f"{provider_type}/{provider_model}",
        )
    ]


def resolve_model_target(
    model: str,
    default_model: str | None = None,
    opus_model: str | None = None,
    sonnet_model: str | None = None,
    haiku_model: str | None = None,
) -> ResolvedModelTarget:
    """Resolve provider type and provider-local model id for a request model."""
    return resolve_model_targets(
        model,
        default_model=default_model,
        opus_model=opus_model,
        sonnet_model=sonnet_model,
        haiku_model=haiku_model,
    )[0]


def get_original_model(model: str) -> str:
    """
    Get the original model name, storing it before normalization.

    Convenience function that returns the input unchanged, intended to be
    called alongside normalize_model_name to capture the original.

    Args:
        model: The model name

    Returns:
        The model name unchanged (for documentation purposes)
    """
    return model
