import asyncio
import base64
import json
import logging
import os
import time
from collections import Counter
from typing import Any, Dict, List, Optional
from uuid import uuid4

from channels.generic.websocket import AsyncWebsocketConsumer

from backend.engines.singletons import get_facial_engine, get_prewarmed_voice_model
from backend.engines.voice.ser_engine import SenseVoiceSEREngine
from backend.fusion.fusion_layer import build_composite_state, detect_au_patterns
from backend.llm.client import GeminiGenAIStreamingClient, GeminiLiveWebSocketClient
from backend.llm.prompt import PromptBuilder
from backend.tts.engine import stream_tts

logger = logging.getLogger(__name__)


class TherapySessionConsumer(AsyncWebsocketConsumer):
    async def connect(self) -> None:
        self.session_id = str(uuid4())
        self.session_active = True
        self.facial_buffer: List[Dict[str, Any]] = []
        self._tts_log_enabled = os.getenv("TTS_LOG_ENABLED", "").lower() in {"1", "true", "yes"}
        self._tts_log_path = None
        self._last_transcript_text = ""
        self._last_transcript_time = 0.0
        self._metrics: Dict[str, Any] = {
            "frames_received": 0,
            "audio_chunks_received": 0,
            "transcripts_committed": 0,
            "llm_calls": 0,
            "errors": 0,
            "connect_time": time.time(),
        }

        try:
            # ── Facial engine: shared singleton (no reload cost) ────────────
            self.facial_engine = get_facial_engine()

            # ── Voice engine: per-session instance, shared model weights ────
            # Injecting the pre-warmed model avoids re-loading on each connect
            self.voice_engine = SenseVoiceSEREngine()
            prewarmed_model = await get_prewarmed_voice_model()
            self.voice_engine._model = prewarmed_model  # share weights
            await self.voice_engine.start()             # only starts the async task

            self.prompt_builder = PromptBuilder()
            api_key = os.getenv("GEMINI_API_KEY")
            llm_mode = os.getenv("LLM_MODE", "genai").lower()
            if llm_mode == "live":
                model_name = os.getenv("GEMINI_MODEL", "models/gemini-2.0-flash-live-001")
                self.llm_client = GeminiLiveWebSocketClient(api_key=api_key, model=model_name)
            else:
                model_name = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
                self.llm_client = GeminiGenAIStreamingClient(api_key=api_key, model=model_name)

            if self._tts_log_enabled:
                base_path = os.getenv("TTS_LOG_PATH", "logs/tts_session.txt")
                resolved_path = base_path.replace("{session_id}", self.session_id)
                os.makedirs(os.path.dirname(resolved_path) or ".", exist_ok=True)
                with open(resolved_path, "w", encoding="utf-8") as log_file:
                    log_file.write(f"session_id: {self.session_id}\n")
                self._tts_log_path = resolved_path

            await self.accept()
            logger.info(
                "[session=%s] WebSocket connected llm_mode=%s model=%s",
                self.session_id, llm_mode, model_name,
            )
        except Exception:
            logger.exception("[session=%s] WebSocket connect failed", self.session_id)
            await self.close(code=1011)

    async def disconnect(self, close_code: int) -> None:
        self.session_active = False
        duration = round(time.time() - self._metrics.get("connect_time", time.time()), 1)
        logger.info(
            "[session=%s] WebSocket disconnected code=%s duration=%.1fs "
            "frames=%d audio_chunks=%d transcripts=%d llm_calls=%d errors=%d",
            self.session_id, close_code, duration,
            self._metrics["frames_received"],
            self._metrics["audio_chunks_received"],
            self._metrics["transcripts_committed"],
            self._metrics["llm_calls"],
            self._metrics["errors"],
        )
        if hasattr(self, "voice_engine") and self.voice_engine is not None:
            await self.voice_engine.stop()

    async def receive(self, text_data: Optional[str] = None, bytes_data: Optional[bytes] = None) -> None:
        try:
            if not text_data:
                return
            try:
                payload = json.loads(text_data)
            except json.JSONDecodeError:
                return

            message_type = payload.get("type")
            if message_type == "video_frame":
                self._metrics["frames_received"] += 1
                await self._handle_video_frame(payload)
            elif message_type == "audio_chunk":
                self._metrics["audio_chunks_received"] += 1
                await self._handle_audio_chunk(payload)
            elif message_type == "transcript_committed":
                self._metrics["transcripts_committed"] += 1
                await self._handle_transcript_committed(payload)
            else:
                logger.warning(
                    "[session=%s] Unknown message type=%s", self.session_id, message_type
                )
        except Exception as exc:
            self._metrics["errors"] += 1
            logger.exception("[session=%s] receive error type=%s", self.session_id, payload.get("type", "?"))
            try:
                await self.send(text_data=json.dumps({"type": "error", "message": str(exc)}))
            except Exception:
                await self.close(code=1011)

    async def _handle_video_frame(self, payload: Dict[str, Any]) -> None:
        data = payload.get("data") or ""
        try:
            image_bytes = base64.b64decode(data)
        except (ValueError, TypeError):
            return

        t0 = time.perf_counter()
        loop = asyncio.get_running_loop()
        facial_output = await loop.run_in_executor(
            None, self.facial_engine.analyze_frame, image_bytes
        )
        elapsed = time.perf_counter() - t0

        face_detected = facial_output.get("face_detected", False)
        logger.debug(
            "[session=%s] facial_analysis face=%s latency=%.3fs",
            self.session_id, face_detected, elapsed,
        )

        self.facial_buffer.append(facial_output)
        if len(self.facial_buffer) > 5:
            self.facial_buffer = self.facial_buffer[-5:]

    async def _handle_audio_chunk(self, payload: Dict[str, Any]) -> None:
        data = payload.get("data") or ""
        try:
            audio_bytes = base64.b64decode(data)
        except (ValueError, TypeError):
            return
        await self.voice_engine.append_audio(audio_bytes)

    async def _handle_transcript_committed(self, payload: Dict[str, Any]) -> None:
        transcript = payload.get("transcript", "")
        trimmed = transcript.strip()
        now = time.monotonic()
        if trimmed and trimmed == self._last_transcript_text and (now - self._last_transcript_time) < 3.0:
            logger.info(
                "[session=%s] transcript_dedup text=%r",
                self.session_id, trimmed[:120],
            )
            return
        if trimmed:
            self._last_transcript_text = trimmed
            self._last_transcript_time = now
        logger.info(
            "[session=%s] transcript_committed text=%r",
            self.session_id, transcript[:120],
        )

        t_pipeline_start = time.perf_counter()
        try:
            # ── Voice SER ─────────────────────────────────────────────────
            t0 = time.perf_counter()
            voice_output = await self.voice_engine.handle_transcript_commit(transcript)
            logger.info(
                "[session=%s] ser_done emotion=%s latency=%.3fs",
                self.session_id,
                voice_output.get("dominant_emotion"),
                time.perf_counter() - t0,
            )

            # ── Fusion ────────────────────────────────────────────────────
            averaged_facial = self._average_facial_buffer(self.facial_buffer)
            composite_state = build_composite_state(averaged_facial, voice_output, self.session_id)

            # ── LLM ───────────────────────────────────────────────────────
            self._metrics["llm_calls"] += 1
            prompt_payload = self.prompt_builder.build_prompt(composite_state)
            t0 = time.perf_counter()
            response_text = await self.llm_client.generate(
                prompt_payload.get("system_prompt", ""),
                prompt_payload.get("user_message", ""),
            )
            llm_latency = time.perf_counter() - t0
            logger.info(
                "[session=%s] llm_done latency=%.3fs chars=%d",
                self.session_id, llm_latency, len(response_text),
            )

            # ── Send text response to frontend ────────────────────────────
            await self.send(text_data=json.dumps({"type": "therapist_response", "text": response_text}))

            # ── TTS stream ────────────────────────────────────────────────
            if response_text.strip():
                tts_chunks = 0
                t0 = time.perf_counter()

                async def _on_chunk(b64: str) -> None:
                    nonlocal tts_chunks
                    tts_chunks += 1
                    await self.send(text_data=json.dumps({"type": "tts_audio_chunk", "data": b64}))

                async def _on_complete() -> None:
                    await self.send(text_data=json.dumps({"type": "tts_complete"}))
                    logger.info(
                        "[session=%s] tts_done chunks=%d latency=%.3fs",
                        self.session_id, tts_chunks, time.perf_counter() - t0,
                    )

                await stream_tts(
                    text=response_text,
                    on_chunk=_on_chunk,
                    on_complete=_on_complete,
                    log_enabled=self._tts_log_enabled,
                    log_path=self._tts_log_path,
                )

            total = time.perf_counter() - t_pipeline_start
            logger.info("[session=%s] pipeline_total latency=%.3fs", self.session_id, total)

        except Exception as exc:
            self._metrics["errors"] += 1
            logger.exception("[session=%s] pipeline error: %s", self.session_id, exc)
            await self.send(text_data=json.dumps({"type": "error", "message": str(exc)}))

    # ── Facial buffer aggregation ──────────────────────────────────────────────

    @staticmethod
    def _average_facial_buffer(buffer: List[Dict[str, Any]]) -> Dict[str, Any]:
        valid_entries = [entry for entry in buffer if entry.get("face_detected")]
        if not valid_entries:
            return {"face_detected": False}

        au_sums: Dict[str, float] = {}
        gaze_sums = {"yaw": 0.0, "pitch": 0.0}
        emotion_values: List[int] = []
        pattern_confidence: Dict[str, float] = {}

        for entry in valid_entries:
            au_intensities = entry.get("au_intensities", {})
            for key, value in au_intensities.items():
                au_sums[key] = au_sums.get(key, 0.0) + float(value)

            gaze = entry.get("gaze", {})
            gaze_sums["yaw"] += float(gaze.get("yaw", 0.0))
            gaze_sums["pitch"] += float(gaze.get("pitch", 0.0))

            emotion_values.append(int(entry.get("emotion_index", -1)))

            for pattern in detect_au_patterns(au_intensities):
                name = str(pattern.get("pattern", ""))
                if not name:
                    continue
                confidence = float(pattern.get("confidence", 0.0))
                if confidence > pattern_confidence.get(name, 0.0):
                    pattern_confidence[name] = confidence

        count = float(len(valid_entries))
        au_avg = {key: value / count for key, value in au_sums.items()}
        gaze_avg = {key: value / count for key, value in gaze_sums.items()}
        emotion_index = TherapySessionConsumer._mode_with_recent_tiebreak(emotion_values)
        detected_patterns = [
            {"pattern": name, "confidence": confidence}
            for name, confidence in pattern_confidence.items()
        ]

        return {
            "face_detected": True,
            "au_intensities": au_avg,
            "gaze": gaze_avg,
            "emotion_index": emotion_index,
            "detected_patterns": detected_patterns,
        }

    @staticmethod
    def _mode_with_recent_tiebreak(values: List[int]) -> int:
        if not values:
            return -1
        counts = Counter(values)
        best_count = max(counts.values())
        candidates = {value for value, count in counts.items() if count == best_count}
        for value in reversed(values):
            if value in candidates:
                return value
        return values[-1]
