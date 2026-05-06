
import asyncio
import os
import sys
import time
import wave
from typing import Any, Dict, Optional

import cv2

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
	sys.path.insert(0, PROJECT_ROOT)

from backend.engines.facial.engine import OpenFaceFacialEngine
from backend.engines.voice.ser_engine import SenseVoiceSEREngine
from backend.fusion.fusion_layer import build_composite_state


def _default_wav_path() -> str:
	return os.path.join(PROJECT_ROOT, "test_check", "harvard.wav")


async def _stream_wav(engine: SenseVoiceSEREngine, wav_path: str) -> None:
	with wave.open(wav_path, "rb") as wav_file:
		rate = wav_file.getframerate()
		channels = wav_file.getnchannels()
		width = wav_file.getsampwidth()
		frames_per_buffer = int(rate * 0.2)

		if rate != 44100 or channels != 2 or width != 2:
			print(
				"Warning: wav format does not match engine defaults "
				f"(rate={rate}, channels={channels}, width={width})."
			)

		while True:
			chunk = wav_file.readframes(frames_per_buffer)
			if not chunk:
				break
			await engine.append_audio(chunk)
			await asyncio.sleep(frames_per_buffer / rate)


def _format_overlay(result: Dict[str, Any]) -> list[str]:
	lines = [
		f"valence (facial/acoustic/semantic): {result['fusion']['facial_valence']:.2f} / "
		f"{result['fusion']['acoustic_valence']:.2f} / {result['fusion']['semantic_valence']:.2f}",
		f"dissonance: {result['fusion']['cross_modal_dissonance_score']:.2f}",
	]
	if result["fusion"]["clinical_flags"]:
		lines.append(f"flags: {', '.join(result['fusion']['clinical_flags'])}")
	return lines


async def main() -> None:
	wav_path = sys.argv[1] if len(sys.argv) > 1 else _default_wav_path()
	if not os.path.exists(wav_path):
		raise FileNotFoundError(f"Audio file not found: {wav_path}")

	model_dir = os.path.abspath(
		os.path.join(
			os.path.dirname(__file__),
			"..",
			"backend",
			"engines",
			"facial",
			"weights",
		)
	)

	facial_engine = OpenFaceFacialEngine(model_dir=model_dir, device="cuda")
	voice_engine = SenseVoiceSEREngine()

	await voice_engine.start()
	audio_task = asyncio.create_task(_stream_wav(voice_engine, wav_path))

	cap = cv2.VideoCapture(0)
	cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
	if not cap.isOpened():
		audio_task.cancel()
		await voice_engine.stop()
		raise RuntimeError("Unable to open webcam at index 0")

	last_voice_poll = 0.0
	last_print = 0.0
	final_printed = False
	voice_output: Dict[str, Any] = {
		"transcript": "",
		"dominant_emotion": "unknown",
		"emotion_counts": {},
		"segment_emotions": [],
	}

	try:
		while True:
			ret, frame = cap.read()
			if not ret:
				break

			encode_ok, buffer = cv2.imencode(".jpg", frame)
			if not encode_ok:
				continue

			facial_output = facial_engine.analyze_frame(buffer.tobytes())
			now = time.monotonic()
			if now - last_voice_poll >= 4.0:
				voice_output = await voice_engine.handle_transcript_commit("")
				last_voice_poll = now

			if now - last_print >= 2.0:
				composite = build_composite_state(
					facial_output, voice_output, session_id="fusion_test"
				)
				print("\n[Composite State]")
				print(composite)
				last_print = now

			composite = build_composite_state(
				facial_output, voice_output, session_id="fusion_test"
			)
			overlay_lines = _format_overlay(composite)
			y = 24
			for line in overlay_lines:
				cv2.putText(
					frame,
					line,
					(12, y),
					cv2.FONT_HERSHEY_SIMPLEX,
					0.6,
					(0, 255, 0),
					2,
					cv2.LINE_AA,
				)
				y += 24

			cv2.imshow("Fusion Layer", frame)
			if cv2.waitKey(1) & 0xFF == ord("q"):
				break

			if audio_task.done():
				if not final_printed:
					voice_output = await voice_engine.handle_transcript_commit("")
					final_composite = build_composite_state(
						facial_output, voice_output, session_id="fusion_test"
					)
					print("\n[Composite State - Final]")
					print(final_composite)
					final_printed = True
					break
			await asyncio.sleep(0)
	finally:
		audio_task.cancel()
		cap.release()
		cv2.destroyAllWindows()
		await voice_engine.stop()


if __name__ == "__main__":
	asyncio.run(main())
