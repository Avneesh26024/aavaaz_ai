// src/hooks/useTTS.ts
//
// Receives base64-encoded MP3 chunks from the Django WebSocket backend
// (type: "tts_audio_chunk") and plays them gaplessly via Web Audio API.
// ElevenLabs HTTP streaming happens on the backend; this hook only plays
// what the backend forwards over WebSocket.

import { useCallback, useRef, useState } from 'react';

// Minimum bytes to attempt a decode pass (MP3 frames ~417–1044 bytes)
const MIN_DECODE_BYTES = 8192;

export interface TTSApi {
  beginUtterance: (utteranceId: string) => void;
  /** Called when backend sends a tts_audio_chunk message */
  receiveChunk: (base64: string) => void;
  /** Called when backend sends tts_complete */
  receiveComplete: () => void;
  isPlaying: boolean;
  canReplay: (utteranceId: string) => boolean;
  replay: (utteranceId: string) => void;
}

export function useTTS(onPlaybackStart: () => void, onPlaybackEnd: () => void): TTSApi {
  const [isPlaying, setIsPlaying] = useState(false);

  const audioContextRef = useRef<AudioContext | null>(null);
  const nextStartTimeRef = useRef<number>(0);
  const pendingSourcesRef = useRef<number>(0);
  const isPlayingRef = useRef(false);
  const streamEndedRef = useRef(false);
  const decodingRef = useRef(false);
  const forceDecodeRef = useRef(false);
  const activeUtteranceIdRef = useRef<string | null>(null);
  const buffersByUtteranceRef = useRef<Record<string, AudioBuffer[]>>({});
  const [replayableMap, setReplayableMap] = useState<Record<string, true>>({});

  // Accumulated undecoded MP3 bytes across chunks
  const accumulatedRef = useRef<Uint8Array>(new Uint8Array(0));

  const streamingEnabled = import.meta.env.VITE_TTS_STREAMING === 'true';

  // ── AudioContext ─────────────────────────────────────────────────────────
  const getAudioContext = useCallback((): AudioContext => {
    if (!audioContextRef.current || audioContextRef.current.state === 'closed') {
      audioContextRef.current = new AudioContext();
    }
    if (audioContextRef.current.state === 'suspended') {
      void audioContextRef.current.resume();
    }
    return audioContextRef.current;
  }, []);

  const checkAndMarkEnd = useCallback(() => {
    // End playback only when stream is complete AND all scheduled sources have finished
    if (streamEndedRef.current && pendingSourcesRef.current === 0) {
      isPlayingRef.current = false;
      setIsPlaying(false);
      onPlaybackEnd();
    }
  }, [onPlaybackEnd]);

  // Schedule a decoded AudioBuffer for gapless sequential playback
  const scheduleBuffer = useCallback(
    (buffer: AudioBuffer, utteranceId: string) => {
      const ctx = getAudioContext();
      const source = ctx.createBufferSource();
      source.buffer = buffer;
      source.connect(ctx.destination);

      if (!buffersByUtteranceRef.current[utteranceId]) {
        buffersByUtteranceRef.current[utteranceId] = [];
      }
      buffersByUtteranceRef.current[utteranceId].push(buffer);

      const now = ctx.currentTime;
      const startAt = Math.max(now, nextStartTimeRef.current);
      source.start(startAt);
      nextStartTimeRef.current = startAt + buffer.duration;

      pendingSourcesRef.current += 1;
      source.onended = () => {
        pendingSourcesRef.current -= 1;
        checkAndMarkEnd();
      };
    },
    [getAudioContext, checkAndMarkEnd],
  );

  // Try to decode the current accumulation buffer into an AudioBuffer
  const tryDecode = useCallback(
    async (force = false): Promise<void> => {
      if (force) forceDecodeRef.current = true;
      if (decodingRef.current) return;

      const data = accumulatedRef.current;
      const shouldForce = forceDecodeRef.current;
      if (!shouldForce && data.byteLength < MIN_DECODE_BYTES) return;
      if (data.byteLength === 0) return;

      decodingRef.current = true;
      const ctx = getAudioContext();
      const utteranceId = activeUtteranceIdRef.current;
      if (!utteranceId) {
        decodingRef.current = false;
        return;
      }

      try {
        // slice so decodeAudioData can detach the buffer without losing our ref
        const copy = data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
        const audioBuffer = await ctx.decodeAudioData(copy as ArrayBuffer);
        scheduleBuffer(audioBuffer, utteranceId);
        accumulatedRef.current = new Uint8Array(0);
        forceDecodeRef.current = false;
      } catch {
        // Incomplete MP3 frame — keep accumulating until more data arrives
      } finally {
        decodingRef.current = false;
      }

      // If more data arrived while decoding, try again
      const nextData = accumulatedRef.current;
      if (forceDecodeRef.current || nextData.byteLength >= MIN_DECODE_BYTES) {
        void tryDecode(false);
      }
    },
    [getAudioContext, scheduleBuffer],
  );

  // ── Public API ────────────────────────────────────────────────────────────

  const beginUtterance = useCallback((utteranceId: string) => {
    activeUtteranceIdRef.current = utteranceId;
    accumulatedRef.current = new Uint8Array(0);
    streamEndedRef.current = false;
    forceDecodeRef.current = false;
    buffersByUtteranceRef.current[utteranceId] = [];
    setReplayableMap((prev) => {
      if (!prev[utteranceId]) return prev;
      const next = { ...prev };
      delete next[utteranceId];
      return next;
    });
  }, []);

  const receiveChunk = useCallback(
    (base64: string) => {
      const utteranceId = activeUtteranceIdRef.current;
      if (!utteranceId) return;

      if (!isPlayingRef.current && accumulatedRef.current.byteLength === 0) {
        streamEndedRef.current = false;
      }

      // Decode base64 → Uint8Array
      const binary = atob(base64);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);

      // Append to accumulation buffer
      const prev = accumulatedRef.current;
      const merged = new Uint8Array(prev.byteLength + bytes.byteLength);
      merged.set(prev, 0);
      merged.set(bytes, prev.byteLength);
      accumulatedRef.current = merged;

      if (streamingEnabled) {
        // Signal playback start on first chunk
        if (!isPlayingRef.current) {
          streamEndedRef.current = false;
          isPlayingRef.current = true;
          setIsPlaying(true);
          onPlaybackStart();
          // Reset scheduling clock for this utterance
          const ctx = getAudioContext();
          nextStartTimeRef.current = ctx.currentTime;
        }

        void tryDecode(false);
      }
    },
    [getAudioContext, onPlaybackStart, streamingEnabled, tryDecode],
  );

  const receiveComplete = useCallback(() => {
    const utteranceId = activeUtteranceIdRef.current;
    if (!utteranceId) return;

    streamEndedRef.current = true;
    if (!streamingEnabled) {
      if (!isPlayingRef.current) {
        isPlayingRef.current = true;
        setIsPlaying(true);
        onPlaybackStart();
        const ctx = getAudioContext();
        nextStartTimeRef.current = ctx.currentTime;
      }
    }

    // Force-decode whatever bytes remain in the accumulation buffer
    void tryDecode(true).then(() => {
      if ((buffersByUtteranceRef.current[utteranceId] ?? []).length > 0) {
        setReplayableMap((prev) => ({ ...prev, [utteranceId]: true }));
      }
      // If no sources were ever scheduled (e.g. empty response), end immediately
      checkAndMarkEnd();
    });
  }, [checkAndMarkEnd, getAudioContext, onPlaybackStart, streamingEnabled, tryDecode]);

  const replay = useCallback((utteranceId: string) => {
    const buffers = buffersByUtteranceRef.current[utteranceId] ?? [];
    if (buffers.length === 0) return;

    if (!isPlayingRef.current) {
      isPlayingRef.current = true;
      setIsPlaying(true);
      onPlaybackStart();
    }

    const ctx = getAudioContext();
    nextStartTimeRef.current = ctx.currentTime;
    streamEndedRef.current = true;
    buffers.forEach((buffer) => scheduleBuffer(buffer, utteranceId));
  }, [getAudioContext, onPlaybackStart, scheduleBuffer]);

  const canReplay = useCallback(
    (utteranceId: string) => Boolean(replayableMap[utteranceId]),
    [replayableMap],
  );

  return { beginUtterance, receiveChunk, receiveComplete, isPlaying, canReplay, replay };
}
