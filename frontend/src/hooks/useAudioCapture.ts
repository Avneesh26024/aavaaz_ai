// src/hooks/useAudioCapture.ts
import { useCallback, useRef } from 'react';

export interface AudioCaptureApi {
  start: () => Promise<void>;
  stop: () => void;
}

/**
 * onChunk receives a base64-encoded Uint8Array (raw Int16 PCM bytes).
 * The same chunk is sent to both the Django backend and ElevenLabs Scribe.
 */
export function useAudioCapture(
  onChunk: (base64: string) => void,
  isPaused: () => boolean,
): AudioCaptureApi {
  const audioContextRef = useRef<AudioContext | null>(null);
  const workletNodeRef = useRef<AudioWorkletNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const onChunkRef = useRef(onChunk);
  onChunkRef.current = onChunk;

  const start = useCallback(async () => {
    if (audioContextRef.current) return; // already running

    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        sampleRate: 16000,
        echoCancellation: true,
        noiseSuppression: true,
      },
      video: false,
    });
    streamRef.current = stream;

    const audioContext = new AudioContext({ sampleRate: 16000 });
    audioContextRef.current = audioContext;

    // Load the worklet from the public directory (served as static asset)
    await audioContext.audioWorklet.addModule('/pcm-processor.js');

    const workletNode = new AudioWorkletNode(audioContext, 'pcm-processor');
    workletNodeRef.current = workletNode;

    workletNode.port.onmessage = (event: MessageEvent<Int16Array>) => {
      if (isPaused()) return;

      const int16 = event.data;
      // Convert Int16Array → Uint8Array (raw little-endian bytes)
      const uint8 = new Uint8Array(int16.buffer, int16.byteOffset, int16.byteLength);

      // Base64 encode
      let binary = '';
      for (let i = 0; i < uint8.length; i++) {
        binary += String.fromCharCode(uint8[i]);
      }
      const base64 = btoa(binary);
      onChunkRef.current(base64);
    };

    const source = audioContext.createMediaStreamSource(stream);
    sourceRef.current = source;
    source.connect(workletNode);
    // Do NOT connect worklet to audioContext.destination (would cause feedback)
  }, [isPaused]);

  const stop = useCallback(() => {
    workletNodeRef.current?.disconnect();
    workletNodeRef.current = null;

    sourceRef.current?.disconnect();
    sourceRef.current = null;

    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;

    audioContextRef.current?.close();
    audioContextRef.current = null;
  }, []);

  return { start, stop };
}
