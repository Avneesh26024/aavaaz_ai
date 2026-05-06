import os
import sys
import time
from typing import Dict, Any

import cv2

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
	sys.path.insert(0, PROJECT_ROOT)

from backend.engines.facial.engine import OpenFaceFacialEngine


def _read_latest_frame(cap: cv2.VideoCapture, max_grabs: int) -> tuple[bool, Any]:
	if max_grabs < 1:
		max_grabs = 1

	if not cap.grab():
		return False, None

	for _ in range(max_grabs - 1):
		if not cap.grab():
			break

	return cap.retrieve()


def _format_overlay(result: Dict[str, Any], latency_ms: float) -> list[str]:
	lines = [f"latency_ms: {latency_ms:.1f}"]

	if not result.get("face_detected"):
		lines.append("face_detected: False")
		return lines

	gaze = result.get("gaze", {})
	lines.append("face_detected: True")
	lines.append(f"emotion_index: {result.get('emotion_index', -1)}")
	lines.append(
		f"gaze yaw/pitch: {gaze.get('yaw', 0.0):.3f}, {gaze.get('pitch', 0.0):.3f}"
	)

	au_intensities = result.get("au_intensities", {})
	for au_name in ("AU1", "AU2", "AU12", "AU26"):
		if au_name in au_intensities:
			lines.append(f"{au_name}: {au_intensities[au_name]:.4f}")

	return lines


def main() -> None:
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

	engine = OpenFaceFacialEngine(model_dir=model_dir, device="cuda")

	cap = cv2.VideoCapture(0)
	cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

	if not cap.isOpened():
		raise RuntimeError("Unable to open webcam at index 0")

	last_latency_ms = 0.0

	try:
		while True:
			max_grabs = 2 if last_latency_ms < 120.0 else 6
			ret, frame = _read_latest_frame(cap, max_grabs=max_grabs)
			if not ret:
				break

			encode_ok, buffer = cv2.imencode(".jpg", frame)
			if not encode_ok:
				continue

			image_bytes = buffer.tobytes()

			inference_start = time.perf_counter()
			result = engine.analyze_frame(image_bytes)
			last_latency_ms = (time.perf_counter() - inference_start) * 1000.0

			overlay_lines = _format_overlay(result, last_latency_ms)
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

			cv2.imshow("Facial Engine", frame)
			if cv2.waitKey(1) & 0xFF == ord("q"):
				break
	finally:
		cap.release()
		cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
