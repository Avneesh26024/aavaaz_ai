import datetime as dt
import logging
from typing import Any, Dict, List

import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

AU_THRESHOLD = 0.15
_DISSONANCE_MAX_STD = 0.816
_HIGH_DISSONANCE_THRESHOLD = 0.6

logger = logging.getLogger(__name__)

_sentiment_analyzer = SentimentIntensityAnalyzer()


def detect_au_patterns(au_intensities: Dict[str, float]) -> List[Dict[str, float]]:
	patterns: List[Dict[str, float]] = []

	def get_au(key: str) -> float:
		return float(au_intensities.get(key, 0.0))

	au1 = get_au("AU1")
	au2 = get_au("AU2")
	au4 = get_au("AU4")
	au6 = get_au("AU6")
	au9 = get_au("AU9")
	au12 = get_au("AU12")
	au26 = get_au("AU26")

	if au12 > AU_THRESHOLD and au6 < AU_THRESHOLD:
		patterns.append({"pattern": "forced_smile", "confidence": au12})
	if au12 > AU_THRESHOLD and au6 > AU_THRESHOLD:
		patterns.append({"pattern": "genuine_smile", "confidence": min(au12, au6)})
	if au1 > AU_THRESHOLD and au4 > AU_THRESHOLD:
		patterns.append({"pattern": "sadness_signature", "confidence": min(au1, au4)})
	if au4 > AU_THRESHOLD and au12 < AU_THRESHOLD:
		patterns.append({"pattern": "dysphoria_marker", "confidence": au4})
	if au1 > AU_THRESHOLD and au2 > AU_THRESHOLD:
		patterns.append({"pattern": "fear_anxiety", "confidence": min(au1, au2)})
	if au9 > AU_THRESHOLD:
		patterns.append({"pattern": "disgust_contempt", "confidence": au9})
	if au26 > AU_THRESHOLD:
		patterns.append({"pattern": "disengagement", "confidence": au26})

	return patterns


def classify_gaze(gaze: Dict[str, float]) -> str:
	pitch = float(gaze.get("pitch", 0.0))
	yaw = float(gaze.get("yaw", 0.0))

	if pitch < -0.15:
		return "avoidant_downward"
	if abs(yaw) > 0.2:
		return "avoidant_lateral"
	if abs(yaw) < 0.1 and abs(pitch) < 0.1:
		return "engaged"
	return "neutral"


def compute_facial_valence(patterns: List[Dict[str, float]], gaze_pattern: str) -> float:
	value = 0.0
	pattern_names = {entry.get("pattern") for entry in patterns}

	if "genuine_smile" in pattern_names:
		value += 0.5
	if "forced_smile" in pattern_names:
		value -= 0.3
	if "sadness_signature" in pattern_names:
		value -= 0.4
	if "dysphoria_marker" in pattern_names:
		value -= 0.3
	if "fear_anxiety" in pattern_names:
		value -= 0.2
	if "disgust_contempt" in pattern_names:
		value -= 0.1
	if gaze_pattern in {"avoidant_downward", "avoidant_lateral"}:
		value -= 0.1

	return float(max(-1.0, min(1.0, value)))


def compute_acoustic_valence(dominant_emotion: str, emotion_counts: Dict[str, int]) -> float:
	_ = emotion_counts
	mapping = {
		"happy": 1.0,
		"surprised": 0.3,
		"neutral": 0.0,
		"unknown": 0.0,
		"fearful": -0.5,
		"sad": -0.7,
		"angry": -0.6,
		"disgusted": -0.5,
	}
	return float(mapping.get(dominant_emotion, 0.0))


def compute_semantic_valence(transcript: str) -> float:
	if not transcript or not transcript.strip():
		return 0.0
	return float(_sentiment_analyzer.polarity_scores(transcript).get("compound", 0.0))


def compute_dissonance(
	facial_valence: float, acoustic_valence: float, semantic_valence: float
) -> float:
	std = float(np.std([facial_valence, acoustic_valence, semantic_valence]))
	if _DISSONANCE_MAX_STD <= 0:
		return 0.0
	normalized = std / _DISSONANCE_MAX_STD
	return float(max(0.0, min(1.0, normalized)))


def compute_clinical_flags(
	patterns: List[Dict[str, float]],
	gaze_pattern: str,
	dominant_emotion: str,
	dissonance_score: float,
	semantic_valence: float,
	facial_valence: float,
	acoustic_valence: float,
) -> List[str]:
	pattern_names = {entry.get("pattern") for entry in patterns}
	flags: List[str] = []

	if dissonance_score > 0.6 and semantic_valence > 0.2 and facial_valence < -0.2:
		flags.append("emotional_masking")
	if "dysphoria_marker" in pattern_names and dominant_emotion in {"neutral", "sad"}:
		flags.append("blunted_affect")
	if "fear_anxiety" in pattern_names and dominant_emotion in {"fearful", "neutral"}:
		flags.append("anxiety_marker")
	if "forced_smile" in pattern_names and semantic_valence > 0.3:
		flags.append("forced_positivity")
	if "sadness_signature" in pattern_names and acoustic_valence < -0.3:
		flags.append("genuine_distress")
	if "disengagement" in pattern_names and gaze_pattern in {
		"avoidant_downward",
		"avoidant_lateral",
	}:
		flags.append("disengaged")

	return flags


def build_composite_state(
	facial_output: Dict[str, Any],
	voice_output: Dict[str, Any],
	session_id: str,
) -> Dict[str, Any]:
	face_detected = bool(facial_output.get("face_detected", False))
	au_intensities = facial_output.get("au_intensities", {}) if face_detected else {}
	gaze = facial_output.get("gaze", {}) if face_detected else {}
	emotion_index = int(facial_output.get("emotion_index", -1)) if face_detected else -1

	if face_detected:
		patterns = detect_au_patterns(au_intensities)
		gaze_pattern = classify_gaze(gaze)
		facial_valence = compute_facial_valence(patterns, gaze_pattern)
	else:
		patterns = []
		gaze_pattern = "neutral"
		facial_valence = 0.0

	transcript = voice_output.get("transcript", "")
	dominant_emotion = voice_output.get("dominant_emotion", "unknown")
	emotion_counts = voice_output.get("emotion_counts", {})
	segment_emotions = voice_output.get("segment_emotions", [])

	acoustic_valence = compute_acoustic_valence(dominant_emotion, emotion_counts)
	semantic_valence = compute_semantic_valence(transcript)
	dissonance_score = compute_dissonance(
		facial_valence, acoustic_valence, semantic_valence
	)
	clinical_flags = compute_clinical_flags(
		patterns,
		gaze_pattern,
		dominant_emotion,
		dissonance_score,
		semantic_valence,
		facial_valence,
		acoustic_valence,
	)

	if clinical_flags and dissonance_score >= _HIGH_DISSONANCE_THRESHOLD:
		logger.info(
			"High dissonance with clinical flags session=%s score=%.3f flags=%s",
			session_id,
			dissonance_score,
			clinical_flags,
		)

	return {
		"session_id": session_id,
		"timestamp": _utc_timestamp(),
		"facial": {
			"au_intensities": au_intensities,
			"gaze": gaze,
			"emotion_index": emotion_index,
			"detected_patterns": patterns,
			"gaze_pattern": gaze_pattern,
			"facial_valence": facial_valence,
		},
		"voice": {
			"transcript": transcript,
			"dominant_emotion": dominant_emotion,
			"emotion_counts": emotion_counts,
			"segment_emotions": segment_emotions,
			"acoustic_valence": acoustic_valence,
			"semantic_valence": semantic_valence,
		},
		"fusion": {
			"cross_modal_dissonance_score": dissonance_score,
			"clinical_flags": clinical_flags,
			"facial_valence": facial_valence,
			"acoustic_valence": acoustic_valence,
			"semantic_valence": semantic_valence,
		},
	}


def _utc_timestamp() -> str:
	return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
