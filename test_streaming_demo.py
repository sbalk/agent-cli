#!/usr/bin/env python3
"""Demo script to test Kokoro TTS streaming functionality."""

import asyncio
import logging
from pathlib import Path

from agent_cli import config
from agent_cli.services.tts import handle_tts_playback
from rich.live import Live
from rich.console import Console

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_streaming_tts():
    """Test the streaming TTS functionality."""
    
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
    
    # Test text
    test_text = "Hello! This is a test of the streaming Kokoro TTS functionality. The audio should start playing as soon as the first chunks are available, rather than waiting for the complete synthesis to finish."
    
    console = Console()
    
    with Live(console=console) as live:
        try:
            print("🔊 Testing Kokoro TTS streaming...")
            print(f"Text: {test_text}")
            print("=" * 80)
            
            # Test with streaming (should start playing immediately)
            result = await handle_tts_playback(
                text=test_text,
                provider_config=provider_config,
                audio_output_config=audio_output_config,
                wyoming_tts_config=wyoming_tts_config,
                openai_tts_config=openai_tts_config,
                openai_llm_config=openai_llm_config,
                kokoro_tts_config=kokoro_tts_config,
                save_file=None,
                quiet=False,
                logger=logger,
                play_audio=True,
                status_message="🔊 Testing streaming TTS...",
                description="Streaming Audio",
                stop_event=None,
                live=live,
            )
            
            if result:
                print(f"✅ Streaming TTS completed successfully! Audio data size: {len(result)} bytes")
            else:
                print("❌ Streaming TTS failed or returned no audio data")
                
        except Exception as e:
            print(f"❌ Error during streaming TTS test: {e}")
            logger.exception("Streaming TTS test failed")

if __name__ == "__main__":
    asyncio.run(test_streaming_tts())