from typing import Dict, Any, Optional
import logging
import time

import os

import cv2
import tempfile
import numpy as np
import torch
from openface.face_detection import FaceDetector
from openface.multitask_model import MultitaskPredictor

from backend.engines.facial.base import AU_ORDER, BaseFacialEngine

logger = logging.getLogger(__name__)
_NO_FACE_LOG_THRESHOLD = 5

class OpenFaceFacialEngine(BaseFacialEngine):
    """
    Facial engine powered by OpenFace-3.0.
    """

    engine_name = "openface_facial_engine"

    def __init__(
        self,
        model_dir: Optional[str] = None,
        device: str = "cuda",
        config: Dict[str, Any] | None = None,
    ) -> None:
        super().__init__(config=config)
        self._no_face_count = 0

        resolved_model_dir = model_dir or self.config.get("model_dir")
        if not resolved_model_dir:
            resolved_model_dir = os.path.join(
                os.path.dirname(__file__),
                "weights",
            )

        self.device = device
        self.face_model_path = os.path.join(resolved_model_dir, "Alignment_RetinaFace.pth")
        self.multitask_model_path = os.path.join(resolved_model_dir, "MTL_backbone.pth")

        previous_cwd = os.getcwd()
        start_time = time.perf_counter()
        try:
            os.chdir(os.path.dirname(resolved_model_dir))
            self.face_detector = FaceDetector(model_path=self.face_model_path, device=self.device)
            self.multitask_model = MultitaskPredictor(
                model_path=self.multitask_model_path,
                device=self.device,
            )
        finally:
            os.chdir(previous_cwd)

        load_seconds = time.perf_counter() - start_time
        logger.info("OpenFace facial engine loaded in %.2fs", load_seconds)

    async def initialize(self) -> None:
        return None

    async def preprocess_frame(self, frame: Any) -> Any:
        return frame

    async def detect_face(self, frame: Any) -> Dict[str, Any]:
        return {"face_detected": False}

    async def detect_mood(self, face_region: Any) -> Dict[str, Any]:
        return {}

    @staticmethod
    def _to_cpu_numpy(value: Any) -> np.ndarray:
        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def analyze_frame(self, image_bytes: bytes) -> Dict[str, Any]:
        if not image_bytes:
            self._track_no_face()
            return {"face_detected": False}

        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            self._track_no_face()
            return {"face_detected": False}

        # OpenFace's FaceDetector.get_face() calls cv2.imread() internally,
        # so it must receive a file path — not a numpy array.
        # Write the decoded image to a named temp file, pass the path, then clean up.
        tmp_path: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp_path = tmp.name
                # Encode the numpy array back to JPEG bytes and write
                ok, encoded = cv2.imencode(".jpg", image)
                if not ok:
                    self._track_no_face()
                    return {"face_detected": False}
                tmp.write(encoded.tobytes())

            cropped_face, dets = self.face_detector.get_face(tmp_path)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

        if cropped_face is None or dets is None:
            self._track_no_face()
            return {"face_detected": False}

        try:
            emotion_logits, gaze_output, au_output = self.multitask_model.predict(cropped_face)

            emotion_index = int(torch.argmax(emotion_logits, dim=1).item())

            gaze_np = self._to_cpu_numpy(gaze_output).reshape(-1)
            gaze = {
                "yaw": float(gaze_np[0]) if gaze_np.size > 0 else 0.0,
                "pitch": float(gaze_np[1]) if gaze_np.size > 1 else 0.0,
            }

            au_np = self._to_cpu_numpy(au_output).reshape(-1)
            au_intensities = {}
            for idx, au_name in enumerate(AU_ORDER):
                au_intensities[au_name] = float(au_np[idx]) if idx < au_np.size else 0.0

            self._no_face_count = 0
            return {
                "au_intensities": au_intensities,
                "gaze": gaze,
                "emotion_index": emotion_index,
                "face_detected": True,
            }
        except Exception:
            logger.exception("Facial inference failed")
            self._track_no_face()
            return {"face_detected": False}

    def _track_no_face(self) -> None:
        self._no_face_count += 1
        if self._no_face_count >= _NO_FACE_LOG_THRESHOLD:
            logger.warning(
                "Face not detected for %d consecutive frames",
                self._no_face_count,
            )