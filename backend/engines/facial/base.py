# backend/engines/facial/base.py

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional




AU_ORDER = ["AU1", "AU2", "AU4", "AU6", "AU9", "AU12", "AU25", "AU26"]


class BaseFacialEngine(ABC):
    """
    Base abstract class for all facial analysis engines.

    First implementation goal:
    - detect face
    - detect mood/emotion
    """

    engine_name = "base_facial_engine"

    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = config or {}

    @abstractmethod
    async def initialize(self) -> None:
        """
        Load models/resources.
        """
        pass

    @abstractmethod
    async def preprocess_frame(self, frame: Any) -> Any:
        """
        Prepare incoming frame for inference.
        """
        pass

    @abstractmethod
    async def detect_face(self, frame: Any) -> Dict[str, Any]:
        """
        Detect face region from frame.

        Example output:
        {
            "face_detected": True,
            "bbox": [x1, y1, x2, y2]
        }
        """
        pass

    @abstractmethod
    async def detect_mood(self, face_region: Any) -> Dict[str, Any]:
        """
        Detect mood/emotion from face.

        Example output:
        {
            "mood": "sad",
            "confidence": 0.81
        }
        """
        pass

    async def process(self, frame: Any) -> Dict[str, Any]:
        """
        Full facial pipeline.
        """

        processed_frame = await self.preprocess_frame(frame)

        face_data = await self.detect_face(processed_frame)

        if not face_data.get("face_detected"):
            return {
                "success": False,
                "error": "No face detected"
            }

        mood_data = await self.detect_mood(processed_frame)

        return {
            "success": True,
            "engine": self.engine_name,
            "face": face_data,
            "mood": mood_data
        }

