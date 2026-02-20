"""Shared provider utilities used by NIM, OpenRouter, and LM Studio."""

from .error_mapping import map_error
from .heuristic_tool_parser import HeuristicToolParser
from .message_converter import (
    AnthropicToOpenAIConverter,
    get_block_attr,
    get_block_type,
)
from .model_routing import (
    VALID_PROVIDER_TYPES,
    is_valid_provider_type,
    parse_prefixed_model,
)
from .sse_builder import ContentBlockManager, SSEBuilder, map_stop_reason
from .think_parser import ContentChunk, ContentType, ThinkTagParser

__all__ = [
    "VALID_PROVIDER_TYPES",
    "AnthropicToOpenAIConverter",
    "ContentBlockManager",
    "ContentChunk",
    "ContentType",
    "HeuristicToolParser",
    "SSEBuilder",
    "ThinkTagParser",
    "get_block_attr",
    "get_block_type",
    "is_valid_provider_type",
    "map_error",
    "map_stop_reason",
    "parse_prefixed_model",
]
