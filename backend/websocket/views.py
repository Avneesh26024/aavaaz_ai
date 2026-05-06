"""
backend/websocket/views.py

REST views for the Aavaaz backend.
"""

import asyncio
import logging
import os
import time
from typing import Optional

from django.http import HttpRequest, JsonResponse
from elevenlabs import ElevenLabs

logger = logging.getLogger(__name__)

_scribe_client: Optional[ElevenLabs] = None


def _get_scribe_client() -> Optional[ElevenLabs]:
    global _scribe_client
    if _scribe_client is not None:
        return _scribe_client

    api_key = os.getenv("ELEVENLABS_API_KEY", "")
    if not api_key:
        return None

    _scribe_client = ElevenLabs(api_key=api_key)
    return _scribe_client


async def scribe_token(request: HttpRequest) -> JsonResponse:
    """
    GET /api/scribe-token/

    Generates a short-lived ElevenLabs signed token for browser-side
    Scribe WebSocket authentication.

    ElevenLabs Scribe WebSocket does NOT accept the raw API key directly —
    it requires a signed token obtained from their token endpoint.
    """
    client = _get_scribe_client()
    if client is None:
        return JsonResponse({"error": "ELEVENLABS_API_KEY not configured"}, status=503)

    try:
        token_obj = await asyncio.to_thread(
            client.tokens.single_use.create,
            token_type="realtime_scribe",
        )
    except Exception as exc:
        logger.warning("[scribe_token] failed to create signed token: %s", exc)
        return JsonResponse({"error": "Failed to create Scribe token"}, status=502)

    token_value = getattr(token_obj, "token", None) or getattr(token_obj, "access_token", None)
    if not token_value:
        logger.warning("[scribe_token] token missing from response")
        return JsonResponse({"error": "Invalid token response"}, status=502)

    return JsonResponse({"token": token_value})


async def health(request: HttpRequest) -> JsonResponse:
    """
    GET /api/health/

    Returns system status: model load state, env vars presence, uptime.
    """
    from backend.engines.singletons import _facial_engine, _voice_model

    status = {
        "status": "ok",
        "timestamp": time.time(),
        "models": {
            "facial_engine_loaded": _facial_engine is not None,
            "voice_model_loaded": _voice_model is not None,
        },
        "env": {
            "ELEVENLABS_API_KEY": bool(os.getenv("ELEVENLABS_API_KEY")),
            "ELEVENLABS_VOICE_ID": bool(os.getenv("ELEVENLABS_VOICE_ID")),
            "GEMINI_API_KEY": bool(os.getenv("GEMINI_API_KEY")),
            "LLM_MODE": os.getenv("LLM_MODE", "genai"),
            "GEMINI_MODEL": os.getenv("GEMINI_MODEL", "gemini-3-flash-preview"),
        },
    }
    return JsonResponse(status)
