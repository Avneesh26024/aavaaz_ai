from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LiveSessionConfig:
    """Session configuration for the Gemini Live WebSocket API."""

    generation_config: Dict[str, Any] = field(default_factory=dict)
    realtime_input_config: Optional[Dict[str, Any]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    input_audio_transcription: Optional[Dict[str, Any]] = None
    output_audio_transcription: Optional[Dict[str, Any]] = None
    history_config: Optional[Dict[str, Any]] = None
    proactivity: Optional[Dict[str, Any]] = None
    context_window_compression: Optional[Dict[str, Any]] = None
    session_resumption: Optional[Dict[str, Any]] = None
    media_resolution: Optional[str] = None

    def build_setup_payload(
        self, model: str, system_prompt: str, response_modalities: List[str]
    ) -> Dict[str, Any]:
        generation_config = {"responseModalities": response_modalities}
        generation_config.update(self.generation_config)

        setup: Dict[str, Any] = {
            "model": f"models/{model}",
            "generationConfig": generation_config,
            "systemInstruction": {"parts": [{"text": system_prompt}]},
        }

        if self.realtime_input_config is not None:
            setup["realtimeInputConfig"] = self.realtime_input_config
        if self.tools is not None:
            setup["tools"] = self.tools
        if self.input_audio_transcription is not None:
            setup["inputAudioTranscription"] = self.input_audio_transcription
        if self.output_audio_transcription is not None:
            setup["outputAudioTranscription"] = self.output_audio_transcription
        if self.history_config is not None:
            setup["historyConfig"] = self.history_config
        if self.proactivity is not None:
            setup["proactivity"] = self.proactivity
        if self.context_window_compression is not None:
            setup["contextWindowCompression"] = self.context_window_compression
        if self.session_resumption is not None:
            setup["sessionResumption"] = self.session_resumption
        if self.media_resolution is not None:
            setup["mediaResolution"] = self.media_resolution

        return {"setup": setup}
