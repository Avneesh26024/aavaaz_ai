import os
import re
from typing import Any, Dict, Optional, Tuple

from funasr import AutoModel


class SenseVoiceSER:
	"""SenseVoice SER wrapper that returns a dominant emotion and cleaned transcript."""

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

	def __init__(
		self,
		model_dir: str = "iic/SenseVoiceSmall",
		device: str = "cuda:0",
		vad_model: Optional[str] = None,
		vad_kwargs: Optional[Dict[str, Any]] = None,
		trust_remote_code: bool = True,
		remote_code: Optional[str] = None,
		disable_update: bool = True,
	) -> None:
		if vad_kwargs is None:
			vad_kwargs = {"max_single_segment_time": 30000}

		if trust_remote_code and not remote_code:
			base_dir = os.path.dirname(os.path.abspath(__file__))
			remote_code = os.path.join(base_dir, "SenseVoice", "model.py")

		self.model = AutoModel(
			model=model_dir,
			trust_remote_code=trust_remote_code,
			remote_code=remote_code,
			vad_model=vad_model,
			vad_kwargs=vad_kwargs,
			device=device,
			disable_update=disable_update,
		)

	def predict(
		self,
		audio_input: Any,
		language: str = "auto",
		use_itn: bool = True,
		batch_size_s: int = 60,
		merge_vad: bool = True,
		merge_length_s: int = 15,
		ban_emo_unk: bool = False,
	) -> Dict[str, Any]:
		result = self.model.generate(
			input=audio_input,
			cache={},
			language=language,
			use_itn=use_itn,
			batch_size_s=batch_size_s,
			merge_vad=merge_vad,
			merge_length_s=merge_length_s,
			ban_emo_unk=ban_emo_unk,
		)

		raw_text = result[0]["text"] if result else ""
		emotion, counts = self._extract_emotion(raw_text)
		transcript = self._strip_tokens(raw_text)

		return {
			"emotion": emotion,
			"transcript": transcript,
			"raw_text": raw_text,
			"emotion_counts": counts,
		}

	def _extract_emotion(self, text: str) -> Tuple[str, Dict[str, int]]:
		counts: Dict[str, int] = {}
		for token, label in self._EMO_TOKENS.items():
			counts[label] = text.count(token)

		# Prefer the most frequent emotion token, default to neutral.
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

	@staticmethod
	def _strip_tokens(text: str) -> str:
		# Remove all <|...|> tokens and normalize whitespace.
		text = re.sub(r"<\|.*?\|>", " ", text)
		return re.sub(r"\s+", " ", text).strip()


if __name__ == "__main__":
	# Example usage with local cache paths.
	ser = SenseVoiceSER(
		model_dir="/home/avneesh/.cache/modelscope/hub/models/iic/SenseVoiceSmall",
		vad_model="/home/avneesh/.cache/modelscope/hub/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
		device="cuda:0",
	)

	result = ser.predict("/home/avneesh/Aavaaz_Engine/test_check/harvard.wav")
	print(result)
