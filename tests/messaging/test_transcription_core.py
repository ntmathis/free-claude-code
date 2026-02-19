import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from messaging.transcription import (
    transcribe_audio,
    _get_pipeline,
    _resolve_model_id,
    _MODEL_MAP,
    _pipeline_cache,
    MAX_AUDIO_SIZE_BYTES,
)


@pytest.fixture(autouse=True)
def wipe_cache():
    _pipeline_cache.clear()
    yield
    _pipeline_cache.clear()


def test_resolve_model_id():
    assert _resolve_model_id("base") == "openai/whisper-base"
    assert _resolve_model_id("nonexistent") == "nonexistent"


def test_get_pipeline_invalid_device():
    with pytest.raises(ValueError, match="must be 'cpu' or 'cuda'"):
        _get_pipeline("base", "invalid_device")


def test_transcribe_audio_file_not_found():
    with pytest.raises(FileNotFoundError, match="Audio file not found"):
        transcribe_audio(Path("/path/to/nowhere/audio.mp3"), "audio/mpeg")


@patch("pathlib.Path.exists")
@patch("pathlib.Path.stat")
def test_transcribe_audio_file_too_large(mock_stat, mock_exists):
    mock_exists.return_value = True
    mock_stat_obj = MagicMock()
    mock_stat_obj.st_size = MAX_AUDIO_SIZE_BYTES + 1
    mock_stat.return_value = mock_stat_obj

    with pytest.raises(ValueError, match="Audio file too large"):
        transcribe_audio(Path("large_audio.mp3"), "audio/mpeg")


@patch("pathlib.Path.exists")
@patch("pathlib.Path.stat")
@patch("messaging.transcription._transcribe_local")
def test_transcribe_audio_success(mock_transcribe_local, mock_stat, mock_exists):
    mock_exists.return_value = True
    mock_stat_obj = MagicMock()
    mock_stat_obj.st_size = 1024
    mock_stat.return_value = mock_stat_obj

    mock_transcribe_local.return_value = "Hello world"

    res = transcribe_audio(
        Path("audio.mp3"), "audio/mpeg", whisper_model="tiny", whisper_device="cpu"
    )
    assert res == "Hello world"
    mock_transcribe_local.assert_called_once_with(Path("audio.mp3"), "tiny", "cpu")


@patch("messaging.transcription._resolve_model_id")
@patch("messaging.transcription._get_pipeline")
@patch("messaging.transcription._load_audio")
def test_transcribe_local(mock_load_audio, mock_get_pipeline, mock_resolve_model_id):
    from messaging.transcription import _transcribe_local

    mock_resolve_model_id.return_value = "model_id"

    mock_pipe = MagicMock()
    mock_pipe.return_value = {"text": "Transcribed text."}
    mock_get_pipeline.return_value = mock_pipe

    mock_load_audio.return_value = {"array": [1, 2, 3], "sampling_rate": 16000}

    result = _transcribe_local(Path("audio.mp3"), "tiny", "cpu")
    assert result == "Transcribed text."


@patch("messaging.transcription._resolve_model_id")
@patch("messaging.transcription._get_pipeline")
@patch("messaging.transcription._load_audio")
def test_transcribe_local_list_output(
    mock_load_audio, mock_get_pipeline, mock_resolve_model_id
):
    from messaging.transcription import _transcribe_local

    mock_resolve_model_id.return_value = "model_id"

    mock_pipe = MagicMock()
    mock_pipe.return_value = {"text": ["Line 1.", "Line 2."]}
    mock_get_pipeline.return_value = mock_pipe

    mock_load_audio.return_value = {"array": [1, 2, 3], "sampling_rate": 16000}

    result = _transcribe_local(Path("audio.mp3"), "tiny", "cpu")
    assert result == "Line 1. Line 2."


@patch("messaging.transcription._resolve_model_id")
@patch("messaging.transcription._get_pipeline")
@patch("messaging.transcription._load_audio")
def test_transcribe_local_empty(
    mock_load_audio, mock_get_pipeline, mock_resolve_model_id
):
    from messaging.transcription import _transcribe_local

    mock_resolve_model_id.return_value = "model_id"

    mock_pipe = MagicMock()
    mock_pipe.return_value = {"text": None}
    mock_get_pipeline.return_value = mock_pipe

    mock_load_audio.return_value = {"array": [1, 2, 3], "sampling_rate": 16000}

    result = _transcribe_local(Path("audio.mp3"), "tiny", "cpu")
    assert result == "(no speech detected)"
