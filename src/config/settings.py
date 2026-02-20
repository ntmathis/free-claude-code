"""Centralized configuration using Pydantic Settings."""

from functools import lru_cache

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from providers.common.model_routing import parse_prefixed_model_roster

from .nim import NimSettings

load_dotenv()

# Fixed base URL for NVIDIA NIM
NVIDIA_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # ==================== Provider Selection ====================
    # Valid: "nvidia_nim" | "open_router" | "lmstudio"
    provider_type: str = "nvidia_nim"

    # ==================== OpenRouter Config ====================
    open_router_api_key: str = Field(default="", validation_alias="OPENROUTER_API_KEY")

    # ==================== Messaging Platform Selection ====================
    # Valid: "telegram" | "discord"
    messaging_platform: str = Field(
        default="discord", validation_alias="MESSAGING_PLATFORM"
    )

    # ==================== NVIDIA NIM Config ====================
    nvidia_nim_api_key: str = ""

    # ==================== LM Studio Config ====================
    lm_studio_base_url: str = Field(
        default="http://localhost:1234/v1",
        validation_alias="LM_STUDIO_BASE_URL",
    )

    # ==================== Model ====================
    # Default model for all Claude model requests if specific overrides aren't provided
    model: str = "nvidia_nim/stepfun-ai/step-3.5-flash"

    # Specific Claude model overrides
    opus_model: str | None = Field(default=None, validation_alias="OPUS_MODEL")
    sonnet_model: str | None = Field(default=None, validation_alias="SONNET_MODEL")
    haiku_model: str | None = Field(default=None, validation_alias="HAIKU_MODEL")

    # ==================== Provider Rate Limiting ====================
    provider_rate_limit: int = Field(default=40, validation_alias="PROVIDER_RATE_LIMIT")
    provider_rate_window: int = Field(
        default=60, validation_alias="PROVIDER_RATE_WINDOW"
    )
    provider_max_concurrency: int = Field(
        default=5, validation_alias="PROVIDER_MAX_CONCURRENCY"
    )

    # ==================== HTTP Client Timeouts ====================
    http_read_timeout: float = Field(
        default=300.0, validation_alias="HTTP_READ_TIMEOUT"
    )
    http_write_timeout: float = Field(
        default=10.0, validation_alias="HTTP_WRITE_TIMEOUT"
    )
    http_connect_timeout: float = Field(
        default=2.0, validation_alias="HTTP_CONNECT_TIMEOUT"
    )

    # ==================== Fast Prefix Detection ====================
    fast_prefix_detection: bool = True

    # ==================== Optimizations ====================
    enable_network_probe_mock: bool = True
    enable_title_generation_skip: bool = True
    enable_suggestion_mode_skip: bool = True
    enable_filepath_extraction_mock: bool = True

    # ==================== NIM Settings ====================
    nim: NimSettings = Field(default_factory=NimSettings)

    # ==================== Voice Note Transcription ====================
    voice_note_enabled: bool = Field(
        default=True, validation_alias="VOICE_NOTE_ENABLED"
    )
    # Hugging Face token for faster model downloads (optional)
    hf_token: str = Field(default="", validation_alias="HF_TOKEN")
    # Hugging Face Whisper model ID (e.g. openai/whisper-base) or short name
    whisper_model: str = Field(default="base", validation_alias="WHISPER_MODEL")
    # Device: "cpu" | "cuda"
    whisper_device: str = Field(default="cpu", validation_alias="WHISPER_DEVICE")

    # ==================== Discord Feature Flags ====================
    discord_enable_text_attachments: bool = Field(
        default=True, validation_alias="DISCORD_ENABLE_TEXT_ATTACHMENTS"
    )
    discord_enable_stats_interaction: bool = Field(
        default=True, validation_alias="DISCORD_ENABLE_STATS_INTERACTION"
    )
    discord_enable_presence_updates: bool = Field(
        default=True, validation_alias="DISCORD_ENABLE_PRESENCE_UPDATES"
    )

    # ==================== Bot Wrapper Config ====================
    telegram_bot_token: str | None = None
    allowed_telegram_user_id: str | None = None
    discord_bot_token: str | None = Field(
        default=None, validation_alias="DISCORD_BOT_TOKEN"
    )
    allowed_discord_channels: str | None = Field(
        default=None, validation_alias="ALLOWED_DISCORD_CHANNELS"
    )
    claude_workspace: str = "./agent_workspace"
    allowed_dir: str = ""

    # ==================== Server ====================
    host: str = "0.0.0.0"
    port: int = 8082
    log_file: str = "server.log"

    # Handle empty strings for optional string fields
    @field_validator(
        "telegram_bot_token",
        "allowed_telegram_user_id",
        "discord_bot_token",
        "allowed_discord_channels",
        mode="before",
    )
    @classmethod
    def parse_optional_str(cls, v):
        if v == "":
            return None
        return v

    @field_validator(
        "model",
        "opus_model",
        "sonnet_model",
        "haiku_model",
        mode="before",
    )
    @classmethod
    def validate_model_override(cls, v, info):
        if v == "":
            if info.field_name == "model":
                raise ValueError("MODEL cannot be empty.")
            raise ValueError(
                f"Specific model override '{info.field_name}' cannot be empty. Remove the key or specify a valid model."
            )
        if v is None:
            return v
        if not isinstance(v, str):
            return v
        parse_prefixed_model_roster(v)
        return v

    @field_validator("whisper_device")
    @classmethod
    def validate_whisper_device(cls, v: str) -> str:
        if v not in ("cpu", "cuda"):
            raise ValueError(f"whisper_device must be 'cpu' or 'cuda', got {v!r}")
        return v

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
