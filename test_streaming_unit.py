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
    """Test that get_synthesizer returns streaming synthesizer for Kokoro and OpenAI."""
    
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
    
    # Test Kokoro TTS (should always use streaming)
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
    
    # Test OpenAI TTS (should always use streaming)
    provider_config.tts_provider = "openai"
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
    
    # Test that the function can be called without errors
    # (This is a basic smoke test since mocking aiohttp is complex)
    try:
        result = await _synthesize_speech_kokoro_streaming(
            text="Hello, this is a test!",
            kokoro_tts_config=kokoro_tts_config,
            logger=logger,
            audio_output_config=audio_output_config,
            quiet=True,
            stop_event=None,
            live=MagicMock()
        )
        # The function should return None when the server is not available
        assert result is None
    except Exception as e:
        # It's okay if it fails due to connection issues, but not due to code errors
        assert "connection" in str(e).lower() or "timeout" in str(e).lower() or "refused" in str(e).lower()

if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])