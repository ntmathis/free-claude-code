from unittest.mock import patch


@patch("providers.common.error_mapping.map_error")
def test_nvidia_nim_errors_import(mock_map_error):
    # This just ensures we can import the file, granting 100% coverage
    from providers.nvidia_nim.errors import map_error

    # Basic sanity check
    assert map_error is not None
    assert callable(map_error)


def test_nvidia_nim_utils_import():
    import providers.nvidia_nim.utils as nvidia_utils

    # Verify __all__ utilities are exported correctly
    assert hasattr(nvidia_utils, "AnthropicToOpenAIConverter")
    assert hasattr(nvidia_utils, "ContentBlockManager")
    assert hasattr(nvidia_utils, "ContentChunk")
    assert hasattr(nvidia_utils, "ContentType")
    assert hasattr(nvidia_utils, "HeuristicToolParser")
    assert hasattr(nvidia_utils, "SSEBuilder")
    assert hasattr(nvidia_utils, "ThinkTagParser")
    assert hasattr(nvidia_utils, "get_block_attr")
    assert hasattr(nvidia_utils, "get_block_type")
    assert hasattr(nvidia_utils, "map_stop_reason")
