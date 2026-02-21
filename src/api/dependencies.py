"""Dependency injection for FastAPI."""

from fastapi import HTTPException
from loguru import logger

from config.settings import NVIDIA_NIM_BASE_URL, Settings
from config.settings import get_settings as _get_settings
from providers.base import BaseProvider, ProviderConfig

# Global provider instances by provider_type
_providers: dict[str, BaseProvider] = {}


def get_settings() -> Settings:
    """Get application settings via dependency injection."""
    return _get_settings()


def get_provider_for_type(provider_type: str) -> BaseProvider:
    """Get or create a provider instance for the given provider type."""
    print(f"DEBUG get_provider_for_type called with {provider_type}")
    if provider_type in _providers:
        return _providers[provider_type]

    settings = get_settings()

    if provider_type == "nvidia_nim":
        if not settings.nvidia_nim_api_key or not settings.nvidia_nim_api_key.strip():
            raise HTTPException(
                status_code=503,
                detail=(
                    "NVIDIA_NIM_API_KEY is not set. Add it to your .env file. "
                    "Get a key at https://build.nvidia.com/settings/api-keys"
                ),
            )
        from providers.nvidia_nim import NvidiaNimProvider

        config = ProviderConfig(
            api_key=settings.nvidia_nim_api_key,
            base_url=NVIDIA_NIM_BASE_URL,
            rate_limit=settings.provider_rate_limit,
            rate_window=settings.provider_rate_window,
            max_concurrency=settings.provider_max_concurrency,
            http_read_timeout=settings.http_read_timeout,
            http_write_timeout=settings.http_write_timeout,
            http_connect_timeout=settings.http_connect_timeout,
        )
        _providers[provider_type] = NvidiaNimProvider(config, nim_settings=settings.nim)
    elif provider_type == "open_router":
        if not settings.open_router_api_key or not settings.open_router_api_key.strip():
            raise HTTPException(
                status_code=503,
                detail=(
                    "OPENROUTER_API_KEY is not set. Add it to your .env file. "
                    "Get a key at https://openrouter.ai/keys"
                ),
            )
        from providers.open_router import OpenRouterProvider

        config = ProviderConfig(
            api_key=settings.open_router_api_key,
            base_url="https://openrouter.ai/api/v1",
            rate_limit=settings.provider_rate_limit,
            rate_window=settings.provider_rate_window,
            max_concurrency=settings.provider_max_concurrency,
            http_read_timeout=settings.http_read_timeout,
            http_write_timeout=settings.http_write_timeout,
            http_connect_timeout=settings.http_connect_timeout,
        )
        _providers[provider_type] = OpenRouterProvider(config)
    elif provider_type == "lmstudio":
        from providers.lmstudio import LMStudioProvider

        config = ProviderConfig(
            api_key="lm-studio",
            base_url=settings.lm_studio_base_url,
            rate_limit=settings.provider_rate_limit,
            rate_window=settings.provider_rate_window,
            max_concurrency=settings.provider_max_concurrency,
            http_read_timeout=settings.http_read_timeout,
            http_write_timeout=settings.http_write_timeout,
            http_connect_timeout=settings.http_connect_timeout,
        )
        _providers[provider_type] = LMStudioProvider(config)
    else:
        logger.error(
            "Unknown provider_type: '%s'. Supported: 'nvidia_nim', 'open_router', 'lmstudio'",
            provider_type,
        )
        raise ValueError(
            f"Unknown provider_type: '{provider_type}'. "
            "Supported: 'nvidia_nim', 'open_router', 'lmstudio'"
        )

    logger.info("Provider initialized: %s", provider_type)
    return _providers[provider_type]


def get_provider() -> BaseProvider:
    """Get provider for default configured provider type."""
    settings = get_settings()
    return get_provider_for_type(settings.provider_type)


async def cleanup_provider():
    """Cleanup provider resources."""
    close_error: Exception | None = None
    for provider in list(_providers.values()):
        client = getattr(provider, "_client", None)
        if client and hasattr(client, "aclose"):
            try:
                await client.aclose()
            except Exception as exc:  # pragma: no cover - explicit propagation path
                if close_error is None:
                    close_error = exc
    _providers.clear()
    logger.debug("Provider cleanup completed")
    if close_error is not None:
        raise close_error
