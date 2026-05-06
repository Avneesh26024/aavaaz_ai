// src/hooks/useBackendSocket.ts
import { useCallback, useRef } from 'react';

export interface BackendSocketHandlers {
  onTherapistResponse: (text: string) => void;
  onTTSChunk: (base64: string) => void;
  onTTSComplete: () => void;
  onOpen?: () => void;
  onError: (message: string) => void;
  onDisconnect: () => void;
}

export interface BackendSocketApi {
  connect: () => void;
  disconnect: () => void;
  sendVideoFrame: (base64: string) => void;
  sendAudioChunk: (base64: string) => void;
  sendTranscriptCommit: (transcript: string) => void;
  isConnected: () => boolean;
}

type IncomingMessage =
  | { type: 'therapist_response'; text: string }
  | { type: 'tts_audio_chunk'; data: string }
  | { type: 'tts_complete' }
  | { type: 'error'; message: string };

export function useBackendSocket(handlers: BackendSocketHandlers): BackendSocketApi {
  const wsRef = useRef<WebSocket | null>(null);
  const handlersRef = useRef(handlers);
  handlersRef.current = handlers;

  const send = useCallback((payload: object) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(payload));
    }
  }, []);

  const connect = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState < WebSocket.CLOSING) {
      return;
    }

    const url = import.meta.env.VITE_BACKEND_WS_URL ?? 'ws://localhost:8000/ws/session/';
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('[BackendSocket] connected');
      handlersRef.current.onOpen?.();
    };

    ws.onmessage = (event: MessageEvent<string>) => {
      try {
        const msg = JSON.parse(event.data) as IncomingMessage;
        const h = handlersRef.current;
        switch (msg.type) {
          case 'therapist_response':
            if (msg.text) h.onTherapistResponse(msg.text);
            break;
          case 'tts_audio_chunk':
            if (msg.data) h.onTTSChunk(msg.data);
            break;
          case 'tts_complete':
            h.onTTSComplete();
            break;
          case 'error':
            if (msg.message) h.onError(msg.message);
            break;
        }
      } catch {
        // ignore malformed messages
      }
    };

    ws.onerror = () => {
      handlersRef.current.onError('WebSocket connection error');
    };

    ws.onclose = () => {
      console.log('[BackendSocket] disconnected');
      handlersRef.current.onDisconnect();
    };
  }, []);

  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  }, []);

  const sendVideoFrame = useCallback(
    (base64: string) => send({ type: 'video_frame', data: base64 }),
    [send],
  );

  const sendAudioChunk = useCallback(
    (base64: string) => send({ type: 'audio_chunk', data: base64 }),
    [send],
  );

  const sendTranscriptCommit = useCallback(
    (transcript: string) => send({ type: 'transcript_committed', transcript }),
    [send],
  );

  const isConnected = useCallback(
    (): boolean => wsRef.current?.readyState === WebSocket.OPEN,
    [],
  );

  return {
    connect,
    disconnect,
    sendVideoFrame,
    sendAudioChunk,
    sendTranscriptCommit,
    isConnected,
  };
}
