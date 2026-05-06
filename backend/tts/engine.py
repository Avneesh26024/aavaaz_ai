"""
backend/tts/engine.py

Streams MP3 audio from ElevenLabs HTTP streaming endpoint and forwards
each binary chunk back to the caller via an async callback.

This is intentionally decoupled from the WebSocket consumer — it only
knows about bytes and an async callback. The consumer wires it to
self.send() over Django Channels.
"""

import asyncio
import base64
import json
import logging
import os
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger(__name__)

_ELEVENLABS_BASE = "https://api.elevenlabs.io/v1/text-to-speech"
_DEFAULT_MODEL = "eleven_turbo_v2"
_DEFAULT_OUTPUT_FORMAT = "mp3_44100_128"
_STREAM_CHUNK_SIZE = 4096  # bytes per read


async def stream_tts(
    text: str,
    on_chunk: Callable[[str], Awaitable[None]],
    on_complete: Callable[[], Awaitable[None]],
    voice_id: str | None = None,
    model_id: str = _DEFAULT_MODEL,
    log_enabled: bool = False,
    log_path: Optional[str] = None,
) -> None:
    """
    Fetch TTS audio from ElevenLabs streaming endpoint and call `on_chunk`
    with each base64-encoded MP3 chunk. Calls `on_complete` when the stream ends.

    Args:
        text:        The text to synthesise.
        on_chunk:    Async callable receiving base64 str of each audio chunk.
        on_complete: Async callable invoked once when the full stream has been sent.
        voice_id:    Override voice; falls back to ELEVENLABS_VOICE_ID env var.
        model_id:    ElevenLabs model (default: eleven_turbo_v2).
    """
    api_key = os.getenv("ELEVENLABS_API_KEY", "")
    resolved_voice_id = voice_id or os.getenv("ELEVENLABS_VOICE_ID", "")

    if not api_key:
        raise ValueError("ELEVENLABS_API_KEY environment variable is not set")
    if not resolved_voice_id:
        raise ValueError("ELEVENLABS_VOICE_ID environment variable is not set")

    url = (
        f"{_ELEVENLABS_BASE}/{resolved_voice_id}/stream"
        f"?output_format={_DEFAULT_OUTPUT_FORMAT}"
        f"&optimize_streaming_latency=4"
    )

    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }

    payload: dict[str, Any] = {
        "text": text,
        "model_id": model_id,
        "voice_settings": {
            "stability": 0.9,
            "similarity_boost": 0.75,
            "speed": 1.00,
        },
    }

    log_file = None
    if log_enabled and log_path:
        try:
            log_file = open(log_path, "a", encoding="utf-8")
            log_file.write("payload:\n")
            log_file.write(json.dumps(payload, ensure_ascii=True, indent=2))
            log_file.write("\n\nresponse_chunks:\n")
        except OSError as exc:
            logger.warning("[TTS] failed to open log file %s: %s", log_path, exc)
            log_file = None

    try:
        import httpx  # lazy import — only needed at runtime
    except ImportError as exc:
        raise ImportError(
            "httpx is required for TTS streaming. Install it with: pip install httpx"
        ) from exc

    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream("POST", url, json=payload, headers=headers) as response:
            if response.status_code != 200:
                body = await response.aread()
                raise RuntimeError(
                    f"ElevenLabs TTS error {response.status_code}: {body.decode()}"
                )

            async for chunk in response.aiter_bytes(chunk_size=_STREAM_CHUNK_SIZE):
                if chunk:
                    b64 = base64.b64encode(chunk).decode("ascii")
                    if log_file:
                        log_file.write(b64)
                        log_file.write("\n")
                    await on_chunk(b64)

    await on_complete()
    if log_file:
        log_file.write("\n[complete]\n")
        log_file.close()
    logger.info("[TTS] stream complete for text of length %d", len(text))
