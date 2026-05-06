import asyncio
import logging
import os
import re
import tempfile
import time
import wave
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

from funasr import AutoModel

from .base import BaseSEREngine


class _SenseVoiceModel:
	"""SenseVoice SER wrapper that returns dominant emotion and segments."""

	_EMO_TOKENS = {
		"<|HAPPY|>": "happy",
		"<|SAD|>": "sad",
		"<|ANGRY|>": "angry",
		"<|NEUTRAL|>": "neutral",
		"<|FEARFUL|>": "fearful",
		"<|DISGUSTED|>": "disgusted",
		"<|SURPRISED|>": "surprised",
		"<|EMO_UNKNOWN|>": "unknown",
	}

	_TOKEN_RE = re.compile(r"(<\|[A-Z_]+\|>)")

	def __init__(
		self,
		model_dir: str,
		device: str,
		vad_model: str,
		remote_code: str,
		vad_kwargs: Optional[Dict[str, Any]] = None,
		trust_remote_code: bool = True,
		disable_update: bool = True,
	) -> None:
		if vad_kwargs is None:
			vad_kwargs = {"max_single_segment_time": 30000}

		self._model = AutoModel(
			model=model_dir,
			trust_remote_code=trust_remote_code,
			remote_code=remote_code,
			vad_model=vad_model,
			vad_kwargs=vad_kwargs,
			device=device,
			disable_update=disable_update,
		)

	def infer(self, audio_input: Any) -> Dict[str, Any]:
		result = self._model.generate(
			input=audio_input,
			cache={},
			language="auto",
			use_itn=True,
			batch_size_s=60,
			merge_vad=True,
			merge_length_s=15,
			ban_emo_unk=False,
		)

		raw_text = result[0]["text"] if result else ""
		dominant, counts = self._extract_emotion(raw_text)
		transcript = self._strip_tokens(raw_text)
		segments = self._segments_from_raw(raw_text)

		return {
			"dominant_emotion": dominant,
			"emotion_counts": counts,
			"raw_text": raw_text,
			"transcript": transcript,
			"segment_emotions": segments,
		}

	def _extract_emotion(self, text: str) -> Tuple[str, Dict[str, int]]:
		counts: Dict[str, int] = {}
		for token, label in self._EMO_TOKENS.items():
			counts[label] = text.count(token)

		if not counts:
			return "neutral", counts

		best_label = "neutral"
		best_count = counts.get(best_label, 0)
		for label, count in counts.items():
			if label == "unknown":
				continue
			if count > best_count:
				best_label = label
				best_count = count

		return best_label, counts

	def _segments_from_raw(self, raw_text: str) -> List[Dict[str, str]]:
		segments: List[Dict[str, str]] = []
		current_emotion = "neutral"

		for part in self._TOKEN_RE.split(raw_text):
			if part in self._EMO_TOKENS:
				current_emotion = self._EMO_TOKENS[part]
				continue

			cleaned = self._strip_tokens(part)
			if cleaned:
				segments.append({"text": cleaned, "emotion": current_emotion})

		return segments

	@staticmethod
	def _strip_tokens(text: str) -> str:
		text = re.sub(r"<\|.*?\|>", " ", text)
		return re.sub(r"\s+", " ", text).strip()


class SenseVoiceSEREngine(BaseSEREngine):
	"""Streaming SER engine using SenseVoice with a rolling emotion window."""

	engine_name = "sensevoice_ser_engine"

	_logger = logging.getLogger(__name__)

	def __init__(self, config: Dict[str, Any] | None = None) -> None:
		super().__init__(config)
		self._buffer = bytearray()
		self._buffer_lock = asyncio.Lock()
		self._emotion_window: Deque[Dict[str, Any]] = deque(maxlen=5)
		self._task: Optional[asyncio.Task] = None
		self._running = False
		self._model: Optional[_SenseVoiceModel] = None

		self._sample_rate = int(self.config.get("sample_rate", 16000))
		self._channels = int(self.config.get("channels", 1))
		self._sample_width = int(self.config.get("sample_width_bytes", 2))
		self._window_seconds = int(self.config.get("window_seconds", 4))
		self._interval_seconds = int(self.config.get("interval_seconds", 4))
		self._max_buffer_seconds = int(self.config.get("max_buffer_seconds", 30))

		repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
		default_model_dir = os.path.expanduser(
			"~/.cache/modelscope/hub/models/iic/SenseVoiceSmall"
		)
		default_vad_model = os.path.expanduser(
			"~/.cache/modelscope/hub/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
		)
		default_remote_code = os.path.join(
			repo_root,
			"test_check",
			"SenseVoice",
			"model.py",
		)

		self._model_dir = self.config.get(
			"model_dir",
			os.getenv("SENSEVOICE_MODEL_DIR", default_model_dir),
		)
		self._vad_model = self.config.get(
			"vad_model",
			os.getenv("SENSEVOICE_VAD_MODEL_DIR", default_vad_model),
		)
		self._remote_code = self.config.get(
			"remote_code",
			os.getenv("SENSEVOICE_REMOTE_CODE", default_remote_code),
		)
		self._device = self.config.get("device", "cuda:0")

	async def initialize(self) -> None:
		if self._model is not None:
			return

		start_time = time.perf_counter()
		self._model = _SenseVoiceModel(
			model_dir=self._model_dir,
			device=self._device,
			vad_model=self._vad_model,
			remote_code=self._remote_code,
		)
		load_seconds = time.perf_counter() - start_time
		self._logger.info("SenseVoice model loaded in %.2fs", load_seconds)

	async def start(self) -> None:
		await self.initialize()
		if self._running:
			return

		self._running = True
		self._task = asyncio.create_task(self._run_loop())

	async def stop(self) -> None:
		self._running = False
		if self._task is not None:
			self._task.cancel()
			try:
				await self._task
			except asyncio.CancelledError:
				pass
			self._task = None

	async def append_audio(self, chunk: bytes) -> None:
		if not chunk:
			return

		max_bytes = self._bytes_per_second * self._max_buffer_seconds
		async with self._buffer_lock:
			self._buffer.extend(chunk)
			excess = len(self._buffer) - max_bytes
			if excess > 0:
				del self._buffer[:excess]

	async def handle_transcript_commit(self, transcript: str) -> Dict[str, Any]:
		cleaned = self._clean_text(transcript)
		dominant, counts = self._aggregate_window()
		segments = self._latest_segments()

		return {
			"transcript": cleaned,
			"dominant_emotion": dominant,
			"emotion_counts": counts,
			"segment_emotions": segments,
		}

	async def _run_loop(self) -> None:
		loop = asyncio.get_running_loop()
		while self._running:
			start_time = loop.time()
			await self._infer_latest_window()
			elapsed = loop.time() - start_time
			delay = max(0.0, self._interval_seconds - elapsed)
			if delay:
				await asyncio.sleep(delay)

	async def _infer_latest_window(self) -> None:
		if self._model is None:
			return

		window_bytes = self._bytes_per_second * self._window_seconds
		async with self._buffer_lock:
			if len(self._buffer) < window_bytes:
				return
			window_data = bytes(self._buffer[-window_bytes:])

		temp_path = None
		try:
			temp_path = self._write_temp_wav(window_data)
			result = await asyncio.to_thread(self._model.infer, temp_path)
			self._emotion_window.append(result)
		except Exception:
			self._logger.exception("Voice inference failed")
		finally:
			if temp_path and os.path.exists(temp_path):
				os.remove(temp_path)

	def _write_temp_wav(self, pcm_data: bytes) -> str:
		temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
		temp_file.close()

		with wave.open(temp_file.name, "wb") as wav_file:
			wav_file.setnchannels(self._channels)
			wav_file.setsampwidth(self._sample_width)
			wav_file.setframerate(self._sample_rate)
			wav_file.writeframes(pcm_data)

		return temp_file.name

	def _aggregate_window(self) -> Tuple[str, Dict[str, int]]:
		if not self._emotion_window:
			return "neutral", {}

		summed: Dict[str, int] = {}
		for entry in self._emotion_window:
			for label, count in entry.get("emotion_counts", {}).items():
				summed[label] = summed.get(label, 0) + int(count)

		if not summed:
			return "neutral", summed

		dominant = "neutral"
		best = summed.get(dominant, 0)
		for label, count in summed.items():
			if label == "unknown":
				continue
			if count > best:
				dominant = label
				best = count

		return dominant, summed

	def _latest_segments(self) -> List[Dict[str, str]]:
		if not self._emotion_window:
			return []
		return list(self._emotion_window[-1].get("segment_emotions", []))

	@staticmethod
	def _clean_text(text: str) -> str:
		return re.sub(r"\s+", " ", text).strip()

	@property
	def _bytes_per_second(self) -> int:
		return self._sample_rate * self._channels * self._sample_width
