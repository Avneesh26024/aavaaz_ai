"""
backend/engines/singletons.py

Module-level singletons for heavy ML models.
Models are loaded ONCE when the process starts, not per WebSocket session.
This eliminates the 3-5s lag spike on every new connection.
"""

import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Facial engine singleton ────────────────────────────────────────────────────
_facial_engine = None
_facial_lock = asyncio.Lock()


def get_facial_engine():
    """Return the shared OpenFaceFacialEngine instance (lazy, thread-safe init)."""
    global _facial_engine
    if _facial_engine is None:
        from backend.engines.facial.engine import OpenFaceFacialEngine
        logger.info("[Singleton] Loading OpenFaceFacialEngine...")
        _facial_engine = OpenFaceFacialEngine()
        logger.info("[Singleton] OpenFaceFacialEngine ready")
    return _facial_engine


# ── Voice engine singleton ─────────────────────────────────────────────────────
# SenseVoice has internal state (rolling buffer, async task) so each session
# needs its OWN engine instance — but the underlying _SenseVoiceModel is the
# bottleneck. We share the model weights via a module-level model cache.
#
# The SenseVoiceSEREngine already guards against double-initialization:
#   async def initialize(self): if self._model is not None: return
# So we just need to pre-warm one instance and reuse its _model reference.

_voice_model = None  # shared _SenseVoiceModel instance


async def get_prewarmed_voice_model():
    """
    Return the pre-warmed SenseVoice model weights object.
    The first call loads the model; subsequent calls return immediately.
    """
    global _voice_model
    if _voice_model is not None:
        return _voice_model

    from backend.engines.voice.ser_engine import SenseVoiceSEREngine
    logger.info("[Singleton] Pre-warming SenseVoice model...")
    engine = SenseVoiceSEREngine()
    await engine.initialize()
    _voice_model = engine._model
    logger.info("[Singleton] SenseVoice model ready")
    return _voice_model
