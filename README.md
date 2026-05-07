# Aavaaz Engine

**Live Demo(FRONTEND ONLY):** [https://aavaaz-ai.vercel.app](https://aavaaz-ai.vercel.app)

> **Note**: For a deep dive into the architecture, end-to-end workflow, and subsystem mechanics, please refer to the [Detailed Architecture and Workflow README](DETAILED_README.md).

Aavaaz Engine is a multimodal virtual therapist system that blends facial cues, vocal emotion, and live transcripts to generate empathetic CBT/DBT-style responses in real time. The system streams audio/video from the browser to a Django Channels backend, fuses signals, calls a Gemini LLM, and streams ElevenLabs TTS audio back to the client.

## Architecture Overview

- **Frontend (React)**
  - Lightweight client for streaming raw A/V to the backend, handling Scribe STT, and gapless TTS playback.
- **Backend (Django + Channels)**
  - WebSocket pipeline for video frames, audio chunks, and committed transcripts.
  - REST endpoints for Scribe tokens and health checks.
- **Engines + Fusion**
  - Facial analysis via OpenFace.
  - Speech emotion recognition via SenseVoice.
  - Fusion layer computes valence and cross-modal dissonance.
- **LLM + TTS**
  - Gemini clients (Live WebSocket or GenAI streaming).
  - ElevenLabs TTS streaming endpoint.

## Runtime Flow (End-to-End)

1. **Session start (frontend)**
   - Connects backend WebSocket.
   - Fetches a short-lived Scribe token.
   - Starts microphone capture (PCM for SER) and webcam frame capture (JPEG).
2. **Streaming inputs**
   - Video frames -> backend facial engine.
   - Audio chunks -> backend SER engine.
   - Committed transcripts -> backend pipeline trigger.
3. **Backend pipeline (on transcript commit)**
   - Aggregate SER window.
   - Build multimodal composite state with fusion layer.
   - Call Gemini LLM to generate response text.
   - Stream ElevenLabs TTS audio back to frontend.
4. **Frontend playback**
   - Shows therapist response text.
   - Plays TTS stream gaplessly; supports replay.
5. **Session end**
   - Stops capture, closes Scribe and backend sockets.

## Deep Module Walkthrough

### WebSocket Orchestration (Backend)
- **Consumer**: `backend/websocket/consumer.py` handles connect/disconnect, routes message types, and runs the full pipeline on `transcript_committed`.
- **Routing**: WebSocket URL is `ws://<host>/ws/session/` via `backend/websocket/router.py` and ASGI routing in `Aavaaz/asgi.py`.

### Facial Engine (OpenFace)
- **Engine**: `backend/engines/facial/engine.py` loads OpenFace models and processes JPEG frames.
- **Output**: AU intensities, gaze, and emotion index; buffers last few frames for smoothing.

### Voice SER (SenseVoice)
- **Engine**: `backend/engines/voice/ser_engine.py` runs a rolling inference loop over recent PCM audio.
- **Aggregation**: Returns dominant emotion and segment breakdown on transcript commit.

### Fusion Layer
- **Module**: `backend/fusion/fusion_layer.py` detects AU patterns, computes valence (facial/acoustic/semantic), and cross-modal dissonance.
- **Composite State**: Bundles facial, voice, and fusion signals for the LLM prompt.

### LLM + Prompting
- **Prompt**: `backend/llm/prompt.py` builds a system prompt and serializes the composite state as JSON.
- **Clients**: `backend/llm/client.py` provides Gemini Live WebSocket and GenAI streaming clients.

### TTS Streaming
- **Engine**: `backend/tts/engine.py` streams MP3 chunks from ElevenLabs and forwards base64 chunks to the frontend.



## Key Endpoints

- **WebSocket**: `ws://<host>/ws/session/`
- **Scribe token**: `GET /api/scribe-token/`
- **Health**: `GET /api/health/`

## Environment Variables

- `GEMINI_API_KEY`
- `GEMINI_MODEL` (default: `gemini-3-flash-preview`)
- `LLM_MODE` (`genai` or `live`)
- `ELEVENLABS_API_KEY`
- `ELEVENLABS_VOICE_ID`
- `SENSEVOICE_MODEL_DIR` (optional override)
- `SENSEVOICE_VAD_MODEL_DIR` (optional override)
- `SENSEVOICE_REMOTE_CODE` (optional override)
- `TTS_LOG_ENABLED` (optional)
- `TTS_LOG_PATH` (optional, can include `{session_id}`)

Frontend env (see `frontend/.env.example`):
- `VITE_BACKEND_WS_URL`
- `VITE_BACKEND_API_URL`
- `VITE_ELEVENLABS_VOICE_ID`
- `VITE_DEBUG`
- `VITE_TTS_STREAMING`

## Full Setup

### Prerequisites

- Python 3.10+ (3.11 recommended)
- Node.js 18+
- Redis (default `127.0.0.1:6379` in Django settings)
- GPU + CUDA recommended for OpenFace and SenseVoice (CPU works but is slower)

### Backend Setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

3. Configure environment:

```bash
cp .env.example .env
```

Fill in `ELEVENLABS_API_KEY`, `ELEVENLABS_VOICE_ID`, and `GEMINI_API_KEY`. If you have custom SenseVoice paths, set `SENSEVOICE_MODEL_DIR`, `SENSEVOICE_VAD_MODEL_DIR`, and `SENSEVOICE_REMOTE_CODE`.

4. Run migrations:

```bash
python manage.py migrate
```

5. Start the backend (Channels + ASGI):

```bash
daphne -b 0.0.0.0 -p 8000 Aavaaz.asgi:application
```

### Facial Weights (OpenFace)

The facial engine expects weights in `backend/engines/facial/weights/`.

If the weights are not present locally, download them from the links in
`backend/engines/facial/weights/README.md` and place them in:

```
backend/engines/facial/weights/
```

Required files include:

- `Alignment_RetinaFace.pth`
- `MTL_backbone.pth`
- `mobilenet0.25_Final.pth`
- `stage2_epoch_7_loss_1.1606_acc_0.5589.pth`

### Frontend Setup

1. Install dependencies:

```bash
cd frontend
npm install
```

2. Configure environment:

```bash
cp .env.example .env
```

3. Start the frontend:

```bash
npm run dev
```

### Verify

- Backend health: `GET http://localhost:8000/api/health/`
- Frontend: open the Vite dev URL (printed by `npm run dev`)
- WebSocket: `ws://localhost:8000/ws/session/`
