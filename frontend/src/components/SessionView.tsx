// src/components/SessionView.tsx
import { useCallback, useRef, useState } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { Scribe, RealtimeEvents, type CommitStrategy } from '@elevenlabs/client';

import type { SessionStatus } from '../types/session';
import { useBackendSocket } from '../hooks/useBackendSocket';
import { useAudioCapture } from '../hooks/useAudioCapture';
import { useWebcamCapture } from '../hooks/useWebcamCapture';
import { useTTS } from '../hooks/useTTS';
import { DebugPanel, type DebugStats } from './DebugPanel';

import { WebcamFeed } from './WebcamFeed';
import { ConversationFeed, type ConversationTurn } from './ConversationFeed';

import styles from './SessionView.module.css';

type ScribeConnection = ReturnType<typeof Scribe.connect>;

const initialDebugStats = (): DebugStats => ({
  backendWsState: 'CLOSED',
  scribeState: 'CLOSED',
  framesPerMin: 0,
  audioChunksPerMin: 0,
  transcriptsCommitted: 0,
  ttsChunksReceived: 0,
  lastError: null,
  lastTranscript: null,
  lastResponsePreview: null,
});

export function SessionView() {
  const [status, setStatus] = useState<SessionStatus>('idle');
  const [partialText, setPartialText] = useState('');
  const [conversation, setConversation] = useState<ConversationTurn[]>([]);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [debugStats, setDebugStats] = useState<DebugStats>(initialDebugStats);
  const [speakingId, setSpeakingId] = useState<string | null>(null);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const isTTSPlayingRef = useRef(false);
  const sessionActiveRef = useRef(false);
  const scribeConnectionRef = useRef<ScribeConnection | null>(null);
  const activeTtsIdRef = useRef<string | null>(null);

  // ── Rate counters for debug panel ────────────────────────────────────────────
  const frameCountRef = useRef(0);
  const audioCountRef = useRef(0);
  const rateIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const startRateTracking = useCallback(() => {
    if (rateIntervalRef.current) return;
    rateIntervalRef.current = setInterval(() => {
      setDebugStats((prev) => ({
        ...prev,
        framesPerMin: frameCountRef.current * 60,
        audioChunksPerMin: audioCountRef.current * 60,
      }));
      frameCountRef.current = 0;
      audioCountRef.current = 0;
    }, 1000);
  }, []);

  const stopRateTracking = useCallback(() => {
    if (rateIntervalRef.current) {
      clearInterval(rateIntervalRef.current);
      rateIntervalRef.current = null;
    }
  }, []);

  // ── TTS ─────────────────────────────────────────────────────────────────────
  const handlePlaybackStart = useCallback(() => {
    isTTSPlayingRef.current = true;
    setSpeakingId(activeTtsIdRef.current);
  }, []);
  const handlePlaybackEnd = useCallback(() => {
    isTTSPlayingRef.current = false;
    setSpeakingId(null);
  }, []);
  const { beginUtterance, receiveChunk, receiveComplete, canReplay, replay } = useTTS(
    handlePlaybackStart,
    handlePlaybackEnd,
  );

  // ── Backend WebSocket ────────────────────────────────────────────────────────
  const backendSocket = useBackendSocket({
    onTherapistResponse: useCallback((text: string) => {
      const responseId = uuidv4();
      activeTtsIdRef.current = responseId;
      beginUtterance(responseId);
      setConversation((prev) => [
        ...prev,
        { id: responseId, role: 'assistant', text, timestamp: Date.now() },
      ]);
      setErrorMessage(null);
      setDebugStats((p) => ({ ...p, lastResponsePreview: text, lastError: null }));
    }, [beginUtterance]),
    onTTSChunk: useCallback((base64: string) => {
      receiveChunk(base64);
      setDebugStats((p) => ({ ...p, ttsChunksReceived: p.ttsChunksReceived + 1 }));
    }, [receiveChunk]),
    onTTSComplete: useCallback(() => { receiveComplete(); }, [receiveComplete]),
    onOpen: useCallback(() => {
      setDebugStats((p) => ({ ...p, backendWsState: 'OPEN' }));
    }, []),
    onError: useCallback((msg: string) => {
      setErrorMessage(msg);
      setDebugStats((p) => ({ ...p, lastError: msg }));
    }, []),
    onDisconnect: useCallback(() => {
      setDebugStats((p) => ({ ...p, backendWsState: 'CLOSED' }));
      if (sessionActiveRef.current) {
        sessionActiveRef.current = false;
        setStatus('ended');
      }
    }, []),
  });

  const backendSocketRef = useRef(backendSocket);
  backendSocketRef.current = backendSocket;

  // ── Audio capture ────────────────────────────────────────────────────────────
  const isPaused = useCallback(() => isTTSPlayingRef.current, []);

  const handleAudioChunk = useCallback((base64: string) => {
    if (!sessionActiveRef.current || isTTSPlayingRef.current) return;
    audioCountRef.current += 1;
    backendSocketRef.current.sendAudioChunk(base64);
  }, []);

  const audioCapture = useAudioCapture(handleAudioChunk, isPaused);

  // ── Webcam ────────────────────────────────────────────────────────────────────
  const { startFrameCapture, stopFrameCapture } = useWebcamCapture();

  const handleVideoReady = useCallback((videoEl: HTMLVideoElement) => {
    videoRef.current = videoEl;
  }, []);

  const shouldAcceptTranscript = useCallback((text: string): boolean => {
    const cleaned = text.trim();
    if (!cleaned) return false;
    if (cleaned.endsWith('-') || cleaned.endsWith('...')) return false;
    const words = cleaned.split(/\s+/).filter(Boolean);
    if (words.length < 4 && cleaned.length < 20) return false;
    return true;
  }, []);

  // ── Scribe connection (microphone mode — same as working demo) ────────────────
  const connectScribe = useCallback(async (token: string) => {
    const connection = Scribe.connect({
      token,
      modelId: 'scribe_v2_realtime',
      commitStrategy: 'vad' as CommitStrategy,
      vadSilenceThresholdSecs: 2.0,
      microphone: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });

    scribeConnectionRef.current = connection;

    connection.on(RealtimeEvents.OPEN, () => {
      console.log('[Scribe] Connection opened');
      setDebugStats((p) => ({ ...p, scribeState: 'OPEN' }));
    });

    connection.on(RealtimeEvents.PARTIAL_TRANSCRIPT, (event: { text: string }) => {
      setPartialText(event.text);
    });

    connection.on(RealtimeEvents.COMMITTED_TRANSCRIPT, (event: { text: string }) => {
      setPartialText('');
      if (!sessionActiveRef.current) return;
      if (!shouldAcceptTranscript(event.text)) return;
      console.log('[Scribe] Committed:', event.text);
      const transcriptId = uuidv4();
      setConversation((prev) => [
        ...prev,
        { id: transcriptId, role: 'user', text: event.text, timestamp: Date.now() },
      ]);
      setDebugStats((p) => ({
        ...p,
        transcriptsCommitted: p.transcriptsCommitted + 1,
        lastTranscript: event.text,
      }));
      backendSocketRef.current.sendTranscriptCommit(event.text);
    });

    connection.on(RealtimeEvents.ERROR, (err: { error: string }) => {
      const msg = `Scribe: ${err?.error ?? JSON.stringify(err)}`;
      console.error('[Scribe] Error:', err);
      setErrorMessage(msg);
      setDebugStats((p) => ({ ...p, scribeState: 'ERROR', lastError: msg }));
    });

    connection.on(RealtimeEvents.CLOSE, () => {
      console.log('[Scribe] Connection closed');
      setDebugStats((p) => ({ ...p, scribeState: 'CLOSED' }));
    });

    return connection;
  }, []);

  // ── Session start ─────────────────────────────────────────────────────────────
  const handleStartSession = useCallback(async () => {
    setStatus('starting');
    setErrorMessage(null);
    sessionActiveRef.current = true;
    setPartialText('');
    setConversation([]);
    setDebugStats(initialDebugStats);
    frameCountRef.current = 0;
    audioCountRef.current = 0;

    try {
      backendSocket.connect();
      startRateTracking();

      // Fetch Scribe token and connect
      try {
        const res = await fetch('/api/scribe-token/');
        if (res.ok) {
          const json = (await res.json()) as { token: string };
          console.log('[Session] Scribe token received, connecting...');
          await connectScribe(json.token);
        } else {
          const body = await res.text();
          console.warn('[Session] Scribe token fetch failed:', res.status, body);
          setDebugStats((p) => ({
            ...p,
            lastError: `Scribe token ${res.status}: ${body.slice(0, 80)}`,
          }));
        }
      } catch (err) {
        console.warn('[Session] Scribe setup failed (continuing without STT):', err);
        setDebugStats((p) => ({
          ...p,
          lastError: `Scribe setup: ${err instanceof Error ? err.message : String(err)}`,
        }));
      }

      // Start AudioWorklet for SenseVoice SER
      await audioCapture.start();

      // Start webcam frame capture
      if (videoRef.current) {
        startFrameCapture(videoRef.current, (base64) => {
          if (sessionActiveRef.current) {
            frameCountRef.current += 1;
            backendSocketRef.current.sendVideoFrame(base64);
          }
        });
      }

      setStatus('active');
    } catch (err) {
      console.error('[Session] Failed to start:', err);
      const msg = err instanceof Error ? err.message : 'Failed to start session';
      setErrorMessage(msg);
      setDebugStats((p) => ({ ...p, lastError: msg }));
      setStatus('error');
      sessionActiveRef.current = false;
      stopRateTracking();
    }
  }, [audioCapture, backendSocket, connectScribe, startFrameCapture, startRateTracking, stopRateTracking]);

  // ── Session end ───────────────────────────────────────────────────────────────
  const handleEndSession = useCallback(() => {
    setStatus('ending');
    sessionActiveRef.current = false;
    stopRateTracking();

    audioCapture.stop();
    stopFrameCapture();

    if (scribeConnectionRef.current) {
      scribeConnectionRef.current.close();
      scribeConnectionRef.current = null;
    }

    backendSocket.disconnect();
    setDebugStats((p) => ({ ...p, backendWsState: 'CLOSED', scribeState: 'CLOSED' }));
    setStatus('ended');
  }, [audioCapture, backendSocket, stopFrameCapture, stopRateTracking]);

  // ── Render ────────────────────────────────────────────────────────────────────
  return (
    <div className={styles.root}>
      <div className={styles.layout}>
        <div className={styles.left}>
          <div className={styles.brand}>
            <div className={styles.brandDot} />
            <span className={styles.brandName}>Aavaaz</span>
          </div>
          <WebcamFeed
            status={status}
            onVideoReady={handleVideoReady}
            onStartSession={handleStartSession}
            onEndSession={handleEndSession}
          />
          <section className={styles.aboutCard}>
            <span className={styles.aboutLabel}>About Us</span>
            <p className={styles.aboutBody}>
              Aavaaz is a multimodal virtual therapist that blends facial cues, vocal emotion,
              and live transcripts to deliver empathetic, CBT/DBT‑style support in real time.
            </p>
            <p className={styles.aboutBody}>
              During a session, we stream audio and video over WebSocket, commit speech into
              transcripts, and generate grounded responses that return as text plus expressive
              TTS audio you can replay turn by turn.
            </p>
            <p className={styles.aboutBody}>
              Our fusion layer analyzes affect from facial AUs, acoustic emotion, and language
              sentiment to surface signals like valence and dissonance, helping the assistant
              respond with context and care.
            </p>
          </section>
        </div>

        <div className={styles.right}>
          {errorMessage && (
            <div className={styles.errorBanner}>⚠ {errorMessage}</div>
          )}
          <ConversationFeed
            turns={conversation}
            partialText={partialText}
            speakingId={speakingId}
            canReplay={canReplay}
            onReplay={replay}
          />
        </div>
      </div>

      <DebugPanel stats={debugStats} />
    </div>
  );
}
