from abc import ABC, abstractmethod
import asyncio
import json
import logging
import time
from typing import Any, AsyncIterator, Dict, Iterable, Optional

from backend.llm.config import LiveSessionConfig

logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
	"""Abstract base class for LLM client implementations."""

	@abstractmethod
	async def generate(self, system_prompt: str, user_message: str) -> str:
		"""Return a full response string for the given prompts."""
		raise NotImplementedError

	@abstractmethod
	async def stream(self, system_prompt: str, user_message: str):
		"""Yield streaming response chunks for the given prompts."""
		raise NotImplementedError


class GeminiLiveWebSocketClient(BaseLLMClient):
	"""Gemini Live API client using raw WebSockets and text streaming."""

	def __init__(
		self,
		api_key: Optional[str] = None,
		model: str = "gemini-3-flash-preview",
		response_modalities: Optional[Iterable[str]] = None,
		config: Optional[LiveSessionConfig] = None,
		use_ephemeral_token: bool = False,
		access_token: Optional[str] = None,
		endpoint: Optional[str] = None,
		request_timeout_s: Optional[float] = None,
	) -> None:
		self._api_key = api_key
		self._model = model
		self._response_modalities = list(response_modalities or ["TEXT"])
		self._config = config or LiveSessionConfig()
		self._use_ephemeral_token = use_ephemeral_token
		self._access_token = access_token
		self._endpoint = endpoint
		self._request_timeout_s = request_timeout_s

	def _build_ws_url(self) -> str:
		if self._endpoint:
			return self._endpoint

		if self._use_ephemeral_token:
			if not self._access_token:
				raise ValueError("access_token is required for ephemeral token auth")
			return (
				"wss://generativelanguage.googleapis.com/ws/"
				"google.ai.generativelanguage.v1alpha.GenerativeService."
				"BidiGenerateContentConstrained?access_token="
				+ self._access_token
			)

		if not self._api_key:
			raise ValueError("api_key is required for Gemini Live API")
		return (
			"wss://generativelanguage.googleapis.com/ws/"
			"google.ai.generativelanguage.v1beta.GenerativeService."
			"BidiGenerateContent?key="
			+ self._api_key
		)

	@staticmethod
	def _load_websockets():
		try:
			import websockets  # type: ignore
		except ImportError as exc:
			raise ImportError(
				"websockets is required for GeminiLiveWebSocketClient. "
				"Install it with `pip install websockets`."
			) from exc
		return websockets

	async def _send_setup(self, websocket, system_prompt: str) -> None:
		config_message = self._config.build_setup_payload(
			model=self._model,
			system_prompt=system_prompt,
			response_modalities=self._response_modalities,
		)
		await websocket.send(json.dumps(config_message))

	async def _send_text(self, websocket, text: str) -> None:
		text_message = {"realtimeInput": {"text": text}}
		await websocket.send(json.dumps(text_message))

	async def _await_setup_complete(self, websocket) -> None:
		while True:
			message = await websocket.recv()
			response = json.loads(message)
			if response.get("setupComplete") is not None:
				return
			if "error" in response:
				raise RuntimeError(response["error"])

	def _extract_text_chunks(self, response: Dict[str, Any]) -> Iterable[str]:
		server_content = response.get("serverContent", {})
		model_turn = server_content.get("modelTurn", {})
		parts = model_turn.get("parts", [])
		for part in parts:
			text = part.get("text")
			if text:
				yield text

		output_transcription = server_content.get("outputTranscription", {})
		output_text = output_transcription.get("text")
		if output_text:
			yield output_text

	async def generate(self, system_prompt: str, user_message: str) -> str:
		start_time = time.perf_counter()
		chunks = []
		async for chunk in self.stream(system_prompt, user_message):
			chunks.append(chunk)
		response = "".join(chunks)
		elapsed = time.perf_counter() - start_time
		logger.info("Gemini generate completed in %.2fs", elapsed)
		if not response.strip():
			logger.warning("Gemini returned an empty response")
		return response

	async def stream(self, system_prompt: str, user_message: str) -> AsyncIterator[str]:
		websockets = self._load_websockets()
		ws_url = self._build_ws_url()

		connect_kwargs: Dict[str, Any] = {}
		if self._request_timeout_s is not None:
			connect_kwargs["open_timeout"] = self._request_timeout_s

		try:
			async with websockets.connect(ws_url, **connect_kwargs) as websocket:
				await self._send_setup(websocket, system_prompt)
				await self._await_setup_complete(websocket)
				await self._send_text(websocket, user_message)

				while True:
					message = await websocket.recv()
					response = json.loads(message)
					if "error" in response:
						raise RuntimeError(response["error"])

					for chunk in self._extract_text_chunks(response):
						yield chunk

					server_content = response.get("serverContent", {})
					if server_content.get("turnComplete") is True:
						break
		except websockets.exceptions.ConnectionClosedError as exc:
			logger.error(
				"Gemini websocket closed code=%s reason=%s",
				exc.code,
				exc.reason,
			)
			raise


class GeminiGenAIStreamingClient(BaseLLMClient):
	"""Gemini GenAI client using generate_content_stream (non-live)."""

	def __init__(
		self,
		api_key: Optional[str] = None,
		model: str = "gemini-3-flash-preview",
	) -> None:
		self._api_key = api_key
		self._model = model

	@staticmethod
	def _load_genai():
		try:
			from google import genai  # type: ignore
		except ImportError as exc:
			raise ImportError(
				"google-genai is required for GeminiGenAIStreamingClient. "
				"Install it with `pip install google-genai`."
			) from exc
		return genai

	def _generate_sync(self, system_prompt: str, user_message: str) -> str:
		if not self._api_key:
			raise ValueError("api_key is required for GenAI client")

		genai = self._load_genai()
		client = genai.Client(api_key=self._api_key)
		contents = [system_prompt, user_message]
		response = client.models.generate_content_stream(
			model=self._model,
			contents=contents,
		)
		chunks = []
		for chunk in response:
			text = getattr(chunk, "text", "")
			if text:
				chunks.append(text)
		return "".join(chunks)

	async def generate(self, system_prompt: str, user_message: str) -> str:
		start_time = time.perf_counter()
		response = await asyncio.to_thread(self._generate_sync, system_prompt, user_message)
		elapsed = time.perf_counter() - start_time
		logger.info("GenAI generate completed in %.2fs", elapsed)
		if not response.strip():
			logger.warning("GenAI returned an empty response")
		return response

	async def stream(self, system_prompt: str, user_message: str) -> AsyncIterator[str]:
		response = await self.generate(system_prompt, user_message)
		if response:
			yield response
