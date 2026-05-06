from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseSEREngine(ABC):
    """Base abstract class for speech emotion recognition engines."""

    engine_name = "base_ser_engine"

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config or {}

    @abstractmethod
    async def initialize(self) -> None:
        """Load models/resources."""

    @abstractmethod
    async def start(self) -> None:
        """Start background tasks or streaming resources."""

    @abstractmethod
    async def stop(self) -> None:
        """Stop background tasks or streaming resources."""

    @abstractmethod
    async def append_audio(self, chunk: bytes) -> None:
        """Append a raw audio chunk to the in-memory buffer."""

    @abstractmethod
    async def handle_transcript_commit(self, transcript: str) -> Dict[str, Any]:
        """Return aggregated emotions for a committed transcript."""


class BaseTREngine(ABC):
    """Base abstract class for transcript engines (speech-to-text)."""

    engine_name = "base_tr_engine"

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config or {}

    @abstractmethod
    async def initialize(self) -> None:
        """Load models/resources."""

    @abstractmethod
    async def start(self) -> None:
        """Start background tasks or streaming resources."""

    @abstractmethod
    async def stop(self) -> None:
        """Stop background tasks or streaming resources."""
