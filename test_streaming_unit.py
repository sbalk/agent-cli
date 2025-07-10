#!/usr/bin/env python3
"""Unit test for Kokoro TTS streaming functionality."""

import asyncio
import io
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_cli import config
from agent_cli.services.tts import get_synthesizer, _synthesize_speech_kokoro_streaming

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.mark.asyncio
async def test_get_synthesizer_streaming():
    """Test that get_synthesizer returns streaming synthesizer for Kokoro when requested."""
    
    # Create test configuration
    provider_config = config.ProviderSelection(
        llm_provider="local",
        asr_provider="local", 
        tts_provider="kokoro"
    )
    
    audio_output_config = config.AudioOutput(
        output_device_index=None,
        output_device_name=None,
        tts_speed=1.0,
        enable_tts=True
    )
    
    kokoro_tts_config = config.KokoroTTS(
        kokoro_tts_model="kokoro-v2",
        kokoro_tts_voice="en_female_1",
        kokoro_tts_host="http://localhost:8880"
    )
    
    # Dummy configs for other providers
    wyoming_tts_config = config.WyomingTTS(
        wyoming_tts_ip="localhost",
        wyoming_tts_port=10200,
        wyoming_voice=None,
        wyoming_tts_language=None,
        wyoming_speaker=None
    )
    
    openai_tts_config = config.OpenAITTS(
        openai_tts_model="tts-1",
        openai_tts_voice="alloy"
    )
    
    openai_llm_config = config.OpenAILLM(
        openai_llm_model="gpt-4",
        openai_api_key=None
    )
    
    # Test with streaming enabled
    synthesizer = get_synthesizer(
        provider_config=provider_config,
        audio_output_config=audio_output_config,
        wyoming_tts_config=wyoming_tts_config,
        openai_tts_config=openai_tts_config,
        openai_llm_config=openai_llm_config,
        kokoro_tts_config=kokoro_tts_config,
        use_streaming=True
    )
    
    # Verify that the synthesizer function name contains "streaming"
    assert "streaming" in synthesizer.func.__name__
    
    # Test without streaming (should return non-streaming synthesizer)
    synthesizer = get_synthesizer(
        provider_config=provider_config,
        audio_output_config=audio_output_config,
        wyoming_tts_config=wyoming_tts_config,
        openai_tts_config=openai_tts_config,
        openai_llm_config=openai_llm_config,
        kokoro_tts_config=kokoro_tts_config,
        use_streaming=False
    )
    
    # Verify that the synthesizer function name does not contain "streaming"
    assert "streaming" not in synthesizer.func.__name__

@pytest.mark.asyncio
async def test_streaming_synthesizer_mock():
    """Test the streaming synthesizer with mocked HTTP response."""
    
    # Create test configuration
    kokoro_tts_config = config.KokoroTTS(
        kokoro_tts_model="kokoro-v2",
        kokoro_tts_voice="en_female_1",
        kokoro_tts_host="http://localhost:8880"
    )
    
    audio_output_config = config.AudioOutput(
        output_device_index=None,
        output_device_name=None,
        tts_speed=1.0,
        enable_tts=True
    )
    
    # Mock aiohttp session and response
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.content.iter_chunked = AsyncMock(return_value=[
        b"fake_audio_chunk_1",
        b"fake_audio_chunk_2", 
        b"fake_audio_chunk_3"
    ])
    
    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)
    mock_session.post = AsyncMock(return_value=mock_response)
    
    # Mock pyaudio context
    mock_pyaudio = MagicMock()
    mock_stream = MagicMock()
    mock_stream.write = MagicMock()
    
    with patch('agent_cli.services.tts.aiohttp.ClientSession', return_value=mock_session), \
         patch('agent_cli.services.tts.pyaudio_context', return_value=mock_pyaudio), \
         patch('agent_cli.services.tts.setup_output_stream', return_value={}), \
         patch('agent_cli.services.tts.open_pyaudio_stream', return_value=mock_stream), \
         patch('agent_cli.services.tts.live_timer'), \
         patch('agent_cli.services.tts.print_with_style'):
        
        result = await _synthesize_speech_kokoro_streaming(
            text="Hello, this is a test!",
            kokoro_tts_config=kokoro_tts_config,
            logger=logger,
            audio_output_config=audio_output_config,
            quiet=True,
            stop_event=None,
            live=MagicMock()
        )
        
        # Verify that the function returned audio data
        assert result is not None
        assert len(result) > 0
        
        # Verify that the HTTP request was made correctly
        mock_session.post.assert_called_once()
        call_args = mock_session.post.call_args
        assert call_args[0][0] == "http://localhost:8880/v1/audio/speech"
        assert call_args[1]["headers"]["Content-Type"] == "application/json"
        
        # Verify that audio chunks were written to the stream
        assert mock_stream.write.called

if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])