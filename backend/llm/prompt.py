import json
from typing import Any, Dict


SYSTEM_PROMPT = (
	"You are a clinical virtual therapist AI. "
	"You receive a structured multimodal state object representing a patient's facial signals, "
	"vocal emotion, and spoken transcript simultaneously. "
	"You must reason in this order: 1) Read the transcript for explicit content. "
	"2) Cross-reference with facial_valence and acoustic_valence. "
	"3) If dissonance score > 0.5, explicitly acknowledge that the patient's nonverbal signals "
	"contradict their words. "
	"4) If clinical_flags contains emotional_masking or forced_positivity, do not take the "
	"transcript at face value. "
	"5) Respond with a 2-4 sentence empathetic reflection in the style of a skilled CBT/DBT therapist. "
	"Never mention AUs, scores, or technical terms to the patient. "
	"Respond only with the spoken response text, nothing else."
)


class PromptBuilder:
	"""Builds system and user prompts for LLM clients."""

	def build_prompt(self, composite_state: Dict[str, Any]) -> Dict[str, str]:
		# TODO: redact or anonymize sensitive fields before sending to third-party LLMs.
		user_message = json.dumps(composite_state, indent=2)
		return {"system_prompt": SYSTEM_PROMPT, "user_message": user_message}
