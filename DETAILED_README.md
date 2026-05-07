# Aavaaz Engine - Detailed Architecture and Workflow

This document provides an in-depth technical explanation of the Aavaaz Engine, focusing extensively on the backend orchestration, machine learning (ML) models, multimodal fusion, and the AI interaction layer.

## 1. System Architecture Overview

Aavaaz Engine is fundamentally a real-time, stateful orchestration system built on Django Channels. While the frontend acts as a thin client for capturing raw I/O (streaming PCM audio and JPEG frames via WebSockets), the heavy lifting occurs entirely within the Python backend.

The backend is responsible for concurrent stream processing, maintaining continuous ML inference loops, fusing disparate modalities into a cohesive psychological state, and driving a Large Language Model (LLM) with strict clinical constraints.

## 2. Backend Orchestration (Django Channels)

The core orchestrator is the `TherapySessionConsumer` (`backend/websocket/consumer.py`). It manages the lifecycle of a session and handles high-throughput bidirectional WebSocket communication.

### 2.1. Concurrency and State Management
*   **Event Loop:** The consumer operates asynchronously. When raw video frames or audio chunks arrive, they are immediately ingested into memory buffers.
*   **Singleton Pattern:** To eliminate the massive latency of loading PyTorch models on every connection, `singletons.py` pre-warms the ML models globally. 
    *   The **Facial Engine** is completely stateless and shared across all sessions.
    *   The **Voice Engine** maintains session-specific buffers, but the underlying PyTorch model weights are loaded into GPU/CPU memory once and shared.
*   **Non-Blocking Execution:** ML inference is computationally heavy. The consumer uses `asyncio.get_running_loop().run_in_executor()` to push synchronous blocking calls (like facial analysis) into a thread pool, ensuring the main ASGI event loop remains responsive to incoming WebSocket messages.

## 3. Machine Learning Engines

### 3.1. Facial Engine (OpenFace Integration)
The facial engine (`backend/engines/facial/engine.py`) relies on OpenFace-3.0 for deep facial analysis.
*   **Pipeline:** 
    1.  Base64 JPEGs are decoded into OpenCV matrices (`numpy`).
    2.  Because the OpenFace `FaceDetector` requires file paths, the engine rapidly writes frames to temporary disk files, runs inference, and unlinks the files.
    3.  A RetinaFace model aligns the face, and a multi-task network extracts three key metrics: Action Unit (AU) intensities, gaze (yaw/pitch), and a dominant emotion index.
*   **Smoothing Buffer:** The consumer maintains a sliding window of the last 5 valid frames. When a response is triggered, these frames are averaged (`_average_facial_buffer`). This mitigates the noise inherent in single-frame facial analysis.

### 3.2. Voice Engine (SenseVoice SER)
The speech emotion recognition (SER) engine (`backend/engines/voice/ser_engine.py`) wraps the FunASR AutoModel.
*   **Continuous Inference Loop:** Unlike the facial engine which reacts to incoming frames, the SER engine runs an active background `asyncio.Task` (`_run_loop`). 
*   **Rolling Buffer:** It appends incoming PCM audio chunks to a rolling byte array.
*   **Windowed Analysis:** Every interval (default 4 seconds), it extracts the most recent audio window, writes it to a temporary `.wav` file, and runs the SenseVoice model.
*   **Token Extraction:** The model outputs raw text containing emotion tokens (e.g., `<|SAD|>`). The engine parses these tokens to track segment-level emotions and aggregates them into a dominant acoustic emotion for the window.

## 4. Multimodal Fusion Layer

The `fusion_layer.py` is the analytical core. It translates raw ML metrics (AU floats, emotion strings) into structured clinical insights.

### 4.1. Action Unit (AU) Pattern Recognition
The fusion layer applies threshold logic to the averaged AUs to detect psychological micro-expressions:
*   `AU12` (Lip Corner Puller) > `0.15` AND `AU6` (Cheek Raiser) < `0.15` $\rightarrow$ **"forced_smile"**
*   `AU12` > `0.15` AND `AU6` > `0.15` $\rightarrow$ **"genuine_smile"**
*   `AU1` (Inner Brow Raiser) > `0.15` AND `AU4` (Brow Lowerer) > `0.15` $\rightarrow$ **"sadness_signature"**

### 4.2. Tri-Modal Valence Computation
The system computes a normalized valence score (-1.0 to 1.0) for three distinct channels:
1.  **Facial Valence:** Derived additively from detected AU patterns and gaze classification (e.g., "avoidant_downward" reduces valence).
2.  **Acoustic Valence:** Directly mapped from the SenseVoice dominant emotion (e.g., "happy" = 1.0, "sad" = -0.7).
3.  **Semantic Valence:** Computed using the VADER sentiment intensity analyzer on the finalized spoken transcript.

### 4.3. Cross-Modal Dissonance & Clinical Flags
The hallmark of the engine is detecting when a patient's words contradict their body language.
*   **Dissonance Score:** Calculated as the standard deviation of the three valences, normalized against a maximum theoretical standard deviation (`0.816`).
*   **Clinical Flags:** Boolean logic matrices combine the dissonance score, valences, and AU patterns to output hard flags. For example:
    *   *If* Dissonance > 0.6 *AND* Semantic Valence > 0.2 *AND* Facial Valence < -0.2 $\rightarrow$ **"emotional_masking"**
    *   *If* "dysphoria_marker" is present *AND* Voice is "neutral" or "sad" $\rightarrow$ **"blunted_affect"**

## 5. AI Prompting and LLM Generation

The final phase uses the fused multimodal state to generate a therapeutic response.

### 5.1. The Composite State
The backend bundles the raw metrics, valences, dissonance score, and clinical flags into a massive JSON object representing the patient's real-time state.

### 5.2. Deterministic Prompting
The `PromptBuilder` injects this JSON into a strict system prompt. The prompt is highly engineered to force the LLM (Gemini) to act on the fusion layer's findings:
> *"If dissonance score > 0.5, explicitly acknowledge that the patient's nonverbal signals contradict their words."*
> *"If clinical_flags contains emotional_masking... do not take the transcript at face value."*

### 5.3. Streaming Delivery
The backend calls the Gemini API (supporting both REST GenAI and Live WebSocket APIs). As text chunks are generated, the backend immediately requests MP3 audio chunks from ElevenLabs' streaming TTS endpoint (`backend/tts/engine.py`), forwarding the raw binary chunks to the frontend for gapless playback.
