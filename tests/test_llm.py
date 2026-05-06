import json
import unittest

from backend.llm.client import GeminiLiveWebSocketClient
from backend.llm.prompt import PromptBuilder


class _FakeWebSocket:
	def __init__(self, responses):
		self._responses = iter(responses)
		self.sent = []

	async def __aenter__(self):
		return self

	async def __aexit__(self, exc_type, exc, tb):
		return False

	def __aiter__(self):
		return self

	async def __anext__(self):
		try:
			return next(self._responses)
		except StopIteration as exc:
			raise StopAsyncIteration from exc

	async def send(self, message: str) -> None:
		self.sent.append(message)


class _FakeWebSocketsModule:
	def __init__(self, responses):
		self._responses = responses
		self.last_ws = None

	def connect(self, _url: str, **_kwargs):
		self.last_ws = _FakeWebSocket(self._responses)
		return self.last_ws


class PromptBuilderTests(unittest.TestCase):
	def test_build_prompt_round_trip_json(self):
		composite_state = {
			"session_id": "fusion_test",
			"timestamp": "2026-05-06T04:16:28.870410Z",
			"facial": {
				"au_intensities": {
					"AU1": 0.0023823543451726437,
					"AU2": 0.004540294408798218,
					"AU4": 0.0003541225742083043,
					"AU6": 0.0028886005748063326,
					"AU9": 0.0,
					"AU12": 0.002464796882122755,
					"AU25": 0.010783981531858444,
					"AU26": 0.012458564713597298,
				},
				"gaze": {"yaw": -0.24864783883094788, "pitch": 0.187225341796875},
				"emotion_index": 0,
				"detected_patterns": [],
				"gaze_pattern": "avoidant_lateral",
				"facial_valence": -0.1,
			},
			"voice": {
				"transcript": "",
				"dominant_emotion": "neutral",
				"emotion_counts": {
					"happy": 0,
					"sad": 0,
					"angry": 0,
					"neutral": 2,
					"fearful": 0,
					"disgusted": 0,
					"surprised": 0,
					"unknown": 3,
				},
				"segment_emotions": [
					{
						"text": "are my favorite. A zestful food is the hot cross.",
						"emotion": "unknown",
					}
				],
				"acoustic_valence": 0.0,
				"semantic_valence": 0.0,
			},
			"fusion": {
				"cross_modal_dissonance_score": 0.057770161861646054,
				"clinical_flags": [],
				"facial_valence": -0.1,
				"acoustic_valence": 0.0,
				"semantic_valence": 0.0,
			},
		}

		builder = PromptBuilder()
		result = builder.build_prompt(composite_state)

		self.assertIsInstance(result.get("system_prompt"), str)
		self.assertIsInstance(result.get("user_message"), str)
		self.assertTrue(result["system_prompt"])
		self.assertTrue(result["user_message"])
		self.assertEqual(json.loads(result["user_message"]), composite_state)


class GeminiLiveWebSocketClientTests(unittest.IsolatedAsyncioTestCase):
	async def test_streaming_text_response(self):
		composite_state = {"session_id": "fusion_test", "timestamp": "2026-05-06T04:16:28Z"}
		prompt_builder = PromptBuilder()
		prompts = prompt_builder.build_prompt(composite_state)

		responses = [
			json.dumps({"setupComplete": {}}),
			json.dumps(
				{
					"serverContent": {
						"modelTurn": {"parts": [{"text": "Hello "}]},
					}
				}
			),
			json.dumps(
				{
					"serverContent": {
						"outputTranscription": {"text": "there"},
						"turnComplete": True,
					}
				}
			),
		]

		fake_ws_module = _FakeWebSocketsModule(responses)

		client = GeminiLiveWebSocketClient(api_key="test-key", model="test-model")
		client._load_websockets = lambda: fake_ws_module

		chunks = []
		async for chunk in client.stream(prompts["system_prompt"], prompts["user_message"]):
			chunks.append(chunk)

		self.assertEqual("".join(chunks), "Hello there")
		self.assertIsNotNone(fake_ws_module.last_ws)
		self.assertGreaterEqual(len(fake_ws_module.last_ws.sent), 2)
		setup_payload = json.loads(fake_ws_module.last_ws.sent[0])
		input_payload = json.loads(fake_ws_module.last_ws.sent[1])
		self.assertIn("setup", setup_payload)
		self.assertIn("realtimeInput", input_payload)
