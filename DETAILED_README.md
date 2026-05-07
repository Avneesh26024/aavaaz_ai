# Aavaaz Engine - Detailed Architecture and Workflow

This document provides an in-depth technical explanation of the Aavaaz Engine, covering its architecture, data flow, and individual subsystem mechanics.

## 1. System Architecture

Aavaaz Engine is designed as a real-time, multimodal system comprising two main parts:
*   **Frontend (React + Vite):** A browser-based client that captures user inputs (audio and video) and renders the therapist's responses (text and audio playback).
*   **Backend (Django + Channels):** An asynchronous ASGI application that acts as the orchestration layer. It receives streaming inputs, runs machine learning models for facial and voice analysis, aggregates the data, queries a Large Language Model (Gemini), and streams Text-To-Speech (TTS) back to the client.

The communication bridge between the frontend and backend is a persistent **WebSocket** connection, enabling bidirectional, low-latency data exchange.

## 2. End-to-End Workflow

The lifecycle of a single interaction turn involves several parallel and sequential steps:

### Phase A: Capture and Streaming (Frontend)
1.  **Audio Capture:** `useAudioCapture.ts` utilizes the Web Audio API and an AudioWorklet (`pcm-processor.js`) to capture 16kHz mono audio. It converts raw PCM data into base64 strings and streams them to the backend WebSocket (`type: 'audio_chunk'`).
2.  **Video Capture:** `useWebcamCapture.ts` periodically grabs JPEG frames from the user's webcam and sends them as base64 strings to the backend (`type: 'video_frame'`).
3.  **Speech-to-Text (STT):** Simultaneously, the frontend connects to ElevenLabs Scribe via a separate WebSocket. The Scribe engine performs Voice Activity Detection (VAD). When the user finishes speaking, it emits a `COMMITTED_TRANSCRIPT` event. The frontend then sends this final text to the backend (`type: 'transcript_committed'`).

### Phase B: Processing and Fusion (Backend)
1.  **Facial Analysis (Continuous):** As video frames arrive, the backend passes them to `OpenFaceFacialEngine`. This engine detects the face, extracts Action Unit (AU) intensities (e.g., AU12 for smiling, AU4 for brow lowerer), estimates gaze (yaw/pitch), and classifies the basic emotion. The last 5 valid frames are buffered.
2.  **Voice Emotion Recognition (Continuous):** As audio chunks arrive, they are appended to a rolling byte buffer in `SenseVoiceSEREngine`. A background task periodically extracts a 4-second window from this buffer, saves it as a temporary WAV file, and runs the SenseVoice model to extract the dominant acoustic emotion and segment-level emotions.
3.  **Triggering the Pipeline:** The arrival of the `transcript_committed` message acts as the trigger for a conversational turn.
4.  **Multimodal Fusion:** The `fusion_layer.py` aggregates the buffered facial data (averaging AU intensities and determining the mode emotion). It computes:
    *   **Facial Valence:** Based on detected AU patterns (e.g., "genuine_smile", "sadness_signature") and gaze.
    *   **Acoustic Valence:** Mapped from the dominant voice emotion.
    *   **Semantic Valence:** Computed using VADER sentiment analysis on the transcript.
    *   **Cross-Modal Dissonance:** A standard deviation metric across the three valences. High dissonance with specific patterns yields "clinical flags" (e.g., "emotional_masking").

### Phase C: Response Generation
1.  **LLM Prompting:** The `PromptBuilder` formats the transcript and the fused multimodal state (valences, flags, emotions) into a JSON composite state. This is injected into a strict system prompt tailored for CBT/DBT therapy.
2.  **Gemini Execution:** The backend calls Google's Gemini API (either via GenAI streaming or the Live WebSocket API, depending on configuration).

### Phase D: Delivery and Playback
1.  **Text Delivery:** The generated text response is sent to the frontend (`type: 'therapist_response'`).
2.  **TTS Streaming:** The backend triggers the `engine.py` TTS module, which makes an HTTP stream request to ElevenLabs. As MP3 chunks arrive from ElevenLabs, the backend base64-encodes them and streams them down the WebSocket (`type: 'tts_audio_chunk'`). Finally, a `tts_complete` message is sent.
3.  **Gapless Playback:** The frontend's `useTTS.ts` hook accumulates the base64 chunks, decodes them into `AudioBuffer` objects, and schedules them sequentially on the Web Audio context for smooth, gapless playback.

## 3. Subsystem Details

### 3.1. Singleton Engine Management
To prevent massive latency spikes on connection, the heavy ML models (OpenFace and SenseVoice) are loaded globally via `singletons.py` at server startup.
*   **Facial Engine:** Completely stateless across sessions; a single instance is shared.
*   **Voice Engine:** State is session-specific (the rolling buffer), but the underlying model weights are loaded once and shared across session instances.

### 3.2. Facial Engine (OpenFace)
Located in `backend/engines/facial/engine.py`. It uses a RetinaFace model for face detection and a custom multi-task network for extracting AUs, gaze, and emotions from the cropped face region. Temporary JPEG files are used internally to interface with the OpenFace C++ bindings.

### 3.3. Voice Engine (SenseVoice)
Located in `backend/engines/voice/ser_engine.py`. It wraps the FunASR AutoModel. It processes raw 16kHz PCM audio. Because SenseVoice requires WAV files, the engine writes the rolling buffer to temporary disk files for inference.

### 3.4. Django Channels & Routing
The core orchestrator is `TherapySessionConsumer` in `backend/websocket/consumer.py`. It manages the asyncio lifecycle, tracks metrics (e.g., frames received, latency), and gracefully handles disconnects and error recovery.

## 4. Setup Accuracy Verification
The setup instructions in the main README accurately reflect the requirements:
*   Python 3.10+ and Node.js 18+.
*   Redis is explicitly required and correctly configured in `settings.py` for Django Channels channel layers.
*   The Daphne ASGI server command is correct.
*   The placement of the downloaded OpenFace weights matches the paths expected by the engine code.
*   The frontend Vite setup follows standard conventions.
