import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from cli.manager import CLISessionManager


@pytest.mark.asyncio
async def test_get_or_create_session_existing_in_sessions():
    manager = CLISessionManager(workspace_path="/tmp", api_url="http://x/v1")

    mock_session = MagicMock()
    manager._sessions["real_session_123"] = mock_session

    # temp_to_real mapping
    manager._temp_to_real["temp_123"] = "real_session_123"

    # Lookup using real ID directly
    session, sid, is_new = await manager.get_or_create_session("real_session_123")
    assert session is mock_session
    assert sid == "real_session_123"
    assert is_new is False

    # Lookup using temp ID
    session2, sid2, is_new2 = await manager.get_or_create_session("temp_123")
    assert session2 is mock_session
    assert sid2 == "real_session_123"
    assert is_new2 is False


@pytest.mark.asyncio
async def test_get_or_create_session_existing_in_pending():
    manager = CLISessionManager(workspace_path="/tmp", api_url="http://x/v1")

    mock_session = MagicMock()
    # It hasn't been mapped to a real ID yet, so it's only in pending
    manager._pending_sessions["temp_session_456"] = mock_session

    session, sid, is_new = await manager.get_or_create_session("temp_session_456")
    assert session is mock_session
    assert sid == "temp_session_456"
    assert is_new is False


@pytest.mark.asyncio
async def test_get_stats():
    manager = CLISessionManager(workspace_path="/tmp", api_url="http://x/v1")

    s1 = MagicMock()
    s1.is_busy = True

    s2 = MagicMock()
    s2.is_busy = False

    manager._sessions["a"] = s1
    manager._sessions["b"] = s2
    manager._pending_sessions["c"] = MagicMock()

    stats = manager.get_stats()
    assert stats["active_sessions"] == 2
    assert stats["pending_sessions"] == 1
    assert stats["busy_count"] == 1
